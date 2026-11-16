#include "slang-ir-metal-portable-ray-tracing-stages.h"

#include "slang-ir-metal-portable-ray-tracing-candidates.h"
#include "slang-ir-portable-ray-tracing-util.h"
#include "slang-ir-util.h"

namespace Slang
{
namespace PortableRT
{
namespace Metal
{

void StageRequirements::combine(const StageRequirements& other)
{
    mask |= other.mask;
    minDistance |= other.minDistance;
    rayTime |= other.rayTime;
    rayFlags |= other.rayFlags;
    dispatchRaysIndex |= other.dispatchRaysIndex;
    dispatchRaysDimensions |= other.dispatchRaysDimensions;
}

IRType* getRayPayloadStorageType(IRBuilder& builder, IRType* payloadType)
{
    return isEmptyType(payloadType) ? builder.getVoidType() : payloadType;
}

/// Builds an IR pointer to stage data using the concrete type expected by the selected shader.
/// Metal function-table signatures pass payload, callable data, and attributes as thread void*
/// so independently compiled shaders agree on the argument types. The selected adapter knows
/// the concrete type from the shader's input contract and restores that type here.
///
/// Consider a miss shader whose Context.Payload is `struct Payload { float value; }`.
/// The generated caller and adapter conceptually do the following (other arguments omitted):
///     // Caller: pass the address of the actual payload through the common table signature.
///     Payload payload;
///     missTable[index]((thread void*)&payload, ...);
///     // Selected miss adapter: recover the pointer type needed by input.payload.value.
///     thread Payload* typedPayload = (thread Payload*)payloadAddress;
///
/// `erasedAddress` is the IR value for payloadAddress, and `dataType` is Payload. The returned
/// pointer lets later input lowering pass the same payload storage to invoke and its helpers.
/// For nonempty data, the caller must supply storage of the matching type in the thread address
/// space; this function does not discover or validate the storage's type at runtime.
/// Void and empty structs need no transported bytes, so those cases ignore erasedAddress.
IRInst* getStageDataReference(IRBuilder& builder, IRInst* erasedAddress, IRType* dataType)
{
    if (as<IRVoidType>(dataType))
    {
        // An ordinary helper can still have a reference parameter whose value type is void.
        // Keep that pointer type with a null value: there is no data to access, and creating a
        // local variable or a store with value type void would be invalid IR.
        return builder.getNullPtrValue(builder.getPtrType(dataType, AddressSpace::ThreadLocal));
    }
    if (isEmptyType(dataType))
    {
        // Consider Payload = ApplicationEmpty, where `struct ApplicationEmpty {}` has no fields.
        // Helpers still expect a reference to that specific struct, so supply an initialized
        // local of the correct type. There are no payload bytes to recover from erasedAddress.
        // Later empty-type legalization removes this local and the empty arguments using it.
        auto value = builder.emitVar(dataType);
        builder.emitStore(value, builder.emitDefaultConstruct(dataType));
        return value;
    }
    // Nonempty data needs an address operand. This checks the presence of the IR instruction,
    // not whether the pointer it represents is non-null at runtime.
    SLANG_RELEASE_ASSERT(erasedAddress);
    // Cast thread void* to thread dataType*. Only the pointer's pointee type changes; its
    // address and address space stay the same. This emits no load, store, or copy of the data.
    // For example, writes through the recovered Payload* update the caller's original payload.
    return builder.emitBitCast(
        builder.getPtrType(dataType, AddressSpace::ThreadLocal),
        erasedAddress);
}

IRInst* getRayPayloadReference(
    IRBuilder& builder,
    const RayData& carrier,
    IRInst* rayData,
    IRType* payloadType)
{
    if (carrier.payloadKey)
        return builder.emitFieldAddress(rayData, carrier.payloadKey);

    SLANG_RELEASE_ASSERT(isEmptyType(payloadType));
    return getStageDataReference(builder, nullptr, payloadType);
}

namespace
{

/// Visits ordinary function bodies once, following only their specialized direct calls. The
/// selected static invoke is still callable code; no native-stage decoration is needed to find it.
void collectOperations(IRFunc* function, HashSet<IRFunc*>& visited, List<IRInst*>& operations)
{
    if (!visited.add(function))
        return;
    for (auto block : function->getBlocks())
    {
        for (auto inst : block->getOrdinaryInsts())
        {
            operations.add(inst);
            if (auto call = as<IRCall>(inst))
            {
                if (auto callee = as<IRFunc>(call->getCallee()))
                    collectOperations(callee, visited, operations);
            }
        }
    }
}

/// Identifies a built-in attribute view from its checked input contract, not its attribute type.
/// A procedural BoundingBoxPrimitive<TriangleData> is not a native triangle primitive.
bool isBuiltinAttributeView(IRInst* operation)
{
    auto inputType = operation->getOperand(0)->getDataType();
    if (auto pointer = as<IRPtrTypeBase>(inputType))
        inputType = pointer->getValueType();
    auto input = getResolvedInstForDecorations(inputType)
                     ->findDecoration<IRPortableRTStageInputDecoration>();
    SLANG_RELEASE_ASSERT(input);
    auto primitive = getBuiltinRole(input->getPrimitiveType());
    return primitive == KnownBuiltinDeclName::RayTracingTrianglePrimitive ||
           primitive == KnownBuiltinDeclName::RayTracingCurvePrimitive;
}

enum class ParameterKind
{
    Data,
    HitAttributes,
    Distance,
    HitKind,
    Barycentrics,
    FrontFacing,
    CurveParameter,
    WorldOrigin,
    WorldDirection,
    PrimitiveIndex,
    GeometryIndex,
    InstanceCount,
    InstanceIndices,
    InstanceIDs,
    ObjectOrigin,
    ObjectDirection,
    ObjectToWorld,
    WorldToObject,
    Record,
    Descriptor,
    Records,
    MinDistance,
    RayTime,
    RayFlags,
    DispatchIndex,
    DispatchDimensions,
    GlobalResources,
};

struct ParameterInfo
{
    ParameterKind kind;
    IRType* type;
    const char* name;
    IROp operation = kIROp_Invalid;
};

/// Defines the table signature once for both resource construction and adapter parameters.
/// Its ordering is the physical visible-function ABI consumed by Metal trace emission.
List<ParameterInfo> getVisibleParameters(
    IRBuilder& builder,
    const StageRequirements& requirements,
    IRType* descriptorPointerType)
{
    List<ParameterInfo> result;
    auto floatType = builder.getFloatType();
    auto uintType = builder.getUIntType();
    auto float3Type = builder.getVectorType(floatType, 3);
    auto matrixType = builder.getMatrixType(
        floatType,
        builder.getIntValue(builder.getIntType(), 4),
        builder.getIntValue(builder.getIntType(), 3),
        builder.getIntValue(builder.getIntType(), kMatrixLayoutMode_RowMajor));
    result.add(
        {ParameterKind::Data,
         builder.getPtrType(builder.getVoidType(), AddressSpace::ThreadLocal),
         "payload"});
    if (requirements.has(MetalRTStageRequirement::HitAttributes))
        result.add(
            {ParameterKind::HitAttributes,
             builder.getPtrType(builder.getVoidType(), AddressSpace::ThreadLocal),
             "attributes"});
    if (requirements.has(MetalRTStageRequirement::Distance) ||
        requirements.has(MetalRTStageRequirement::ObjectSpaceRay))
        result.add(
            {ParameterKind::Distance, floatType, "distance", kIROp_MetalRTGetCurrentDistance});
    if (requirements.has(MetalRTStageRequirement::HitKind))
        result.add({ParameterKind::HitKind, uintType, "hitKind", kIROp_MetalRTGetHitKind});
    if (requirements.has(MetalRTStageRequirement::TriangleBarycentricCoord))
        result.add(
            {ParameterKind::Barycentrics,
             builder.getVectorType(floatType, 2),
             "barycentricCoord",
             kIROp_PortableRTGetTriangleBarycentricCoord});
    if (requirements.has(MetalRTStageRequirement::TriangleFrontFacing))
        result.add(
            {ParameterKind::FrontFacing,
             builder.getBoolType(),
             "frontFacing",
             kIROp_MetalRTGetTriangleFrontFacing});
    if (requirements.has(MetalRTStageRequirement::CurveParameter))
        result.add(
            {ParameterKind::CurveParameter,
             floatType,
             "curveParameter",
             kIROp_PortableRTGetCurveParameter});
    if (requirements.has(MetalRTStageRequirement::WorldSpaceOrigin))
        result.add(
            {ParameterKind::WorldOrigin,
             float3Type,
             "worldSpaceOrigin",
             kIROp_MetalRTGetWorldSpaceOrigin});
    if (requirements.has(MetalRTStageRequirement::WorldSpaceDirection))
        result.add(
            {ParameterKind::WorldDirection,
             float3Type,
             "worldSpaceDirection",
             kIROp_MetalRTGetWorldSpaceDirection});
    if (requirements.has(MetalRTStageRequirement::PrimitiveIndex))
        result.add(
            {ParameterKind::PrimitiveIndex,
             uintType,
             "primitiveIndex",
             kIROp_MetalRTGetPrimitiveIndex});
    if (requirements.has(MetalRTStageRequirement::GeometryIndex))
        result.add(
            {ParameterKind::GeometryIndex,
             uintType,
             "geometryIndex",
             kIROp_MetalRTGetGeometryIndex});
    if (requirements.has(MetalRTStageRequirement::InstanceCount))
        result.add(
            {ParameterKind::InstanceCount,
             uintType,
             "instanceCount",
             kIROp_MetalRTGetInstanceCount});
    if (requirements.has(MetalRTStageRequirement::InstanceIndexAtLevel))
        result.add(
            {ParameterKind::InstanceIndices,
             builder.getPtrType(uintType, AddressSpace::ThreadLocal),
             "instanceIndices",
             kIROp_MetalRTGetInstanceIndex});
    if (requirements.has(MetalRTStageRequirement::InstanceIDAtLevel))
        result.add(
            {ParameterKind::InstanceIDs,
             builder.getPtrType(uintType, AddressSpace::ThreadLocal),
             "instanceIDs",
             kIROp_MetalRTGetInstanceID});
    if (requirements.has(MetalRTStageRequirement::ObjectSpaceRay))
    {
        result.add({ParameterKind::ObjectOrigin, float3Type, "objectSpaceOrigin"});
        result.add({ParameterKind::ObjectDirection, float3Type, "objectSpaceDirection"});
    }
    if (requirements.has(MetalRTStageRequirement::ObjectToWorld))
        result.add(
            {ParameterKind::ObjectToWorld,
             matrixType,
             "objectToWorld",
             kIROp_MetalRTGetObjectToWorld});
    if (requirements.has(MetalRTStageRequirement::WorldToObject))
        result.add(
            {ParameterKind::WorldToObject,
             matrixType,
             "worldToObject",
             kIROp_MetalRTGetWorldToObject});
    if (requirements.has(MetalRTStageRequirement::Record))
        result.add(
            {ParameterKind::Record,
             builder.getPtrType(builder.getUInt8Type(), AddressSpace::Global),
             "recordData"});
    if (requirements.has(MetalRTStageRequirement::CallableDispatch))
        result.add({ParameterKind::Descriptor, descriptorPointerType, "descriptorResources"});
    result.add({ParameterKind::MinDistance, floatType, "minDistance", kIROp_MetalRTGetMinDistance});
    result.add({ParameterKind::RayTime, floatType, "rayTime", kIROp_MetalRTGetRayTime});
    result.add({ParameterKind::RayFlags, uintType, "rayFlags", kIROp_MetalRTGetRayFlags});
    auto uint3 = builder.getVectorType(uintType, 3);
    result.add(
        {ParameterKind::DispatchIndex,
         uint3,
         "dispatchRaysIndex",
         kIROp_MetalRTGetDispatchRaysIndex});
    result.add(
        {ParameterKind::DispatchDimensions,
         uint3,
         "dispatchRaysDimensions",
         kIROp_MetalRTGetDispatchRaysDimensions});
    result.add(
        {ParameterKind::Records, builder.getPtrType(uintType, AddressSpace::Global), "records"});
    result.add(
        {ParameterKind::GlobalResources,
         builder.getPtrType(builder.getUInt8Type(), AddressSpace::ThreadLocal),
         "globalResources"});
    return result;
}

/// Defines callable parameter order once for the physical table and each adapter. The emitter's
/// MetalRTCallDispatch consumes this ABI; source data types may differ because the first pointer
/// is erased until the selected adapter knows its Context.CallableData.
List<ParameterInfo> getCallableParameters(
    IRBuilder& builder,
    const StageRequirements& requirements,
    IRType* descriptorPointerType)
{
    SLANG_RELEASE_ASSERT(
        requirements.dispatchRaysIndex && requirements.dispatchRaysDimensions &&
        requirements.has(MetalRTStageRequirement::Record) &&
        requirements.has(MetalRTStageRequirement::CallableDispatch));
    List<ParameterInfo> result;
    result.add(
        {ParameterKind::Data,
         builder.getPtrType(builder.getVoidType(), AddressSpace::ThreadLocal),
         "data"});
    auto uint3 = builder.getVectorType(builder.getUIntType(), 3);
    result.add(
        {ParameterKind::DispatchIndex,
         uint3,
         "dispatchRaysIndex",
         kIROp_MetalRTGetDispatchRaysIndex});
    result.add(
        {ParameterKind::DispatchDimensions,
         uint3,
         "dispatchRaysDimensions",
         kIROp_MetalRTGetDispatchRaysDimensions});
    result.add(
        {ParameterKind::Record,
         builder.getPtrType(builder.getUInt8Type(), AddressSpace::Global),
         "recordData"});
    result.add({ParameterKind::Descriptor, descriptorPointerType, "descriptorResources"});
    result.add(
        {ParameterKind::Records,
         builder.getPtrType(builder.getUIntType(), AddressSpace::Global),
         "records"});
    result.add(
        {ParameterKind::GlobalResources,
         builder.getPtrType(builder.getUInt8Type(), AddressSpace::ThreadLocal),
         "globalResources"});
    return result;
}

/// Emits the one logical input argument without introducing payload or record storage. Source
/// parameter lowering may already have represented `in Input<C>` as a borrowed empty struct.
void emitInvoke(IRBuilder& builder, IRFunc* invoke)
{
    SLANG_RELEASE_ASSERT(invoke->getParamCount() == 1);
    auto type = invoke->getParamType(0);
    IRInst* input;
    if (auto pointer = as<IRPtrTypeBase>(type))
    {
        input = builder.emitVar(pointer->getValueType());
        builder.emitStore(input, builder.emitDefaultConstruct(pointer->getValueType()));
    }
    else
        input = builder.emitDefaultConstruct(type);
    builder.emitCallInst(invoke->getResultType(), invoke, 1, &input);
}

} // namespace

bool isBuiltinHitAttributeView(IRInst* operation)
{
    return isBuiltinAttributeView(operation);
}

bool isStageInputOperation(IROp op)
{
    switch (op)
    {
    case kIROp_PortableRTGetPayload:
    case kIROp_PortableRTGetCallableData:
    case kIROp_PortableRTGetRecord:
    case kIROp_PortableRTGetHitAttributes:
    case kIROp_PortableRTGetTriangleBarycentricCoord:
    case kIROp_PortableRTGetCurveParameter:
    case kIROp_MetalRTGetMinDistance:
    case kIROp_MetalRTGetCurrentDistance:
    case kIROp_MetalRTGetHitKind:
    case kIROp_MetalRTGetRayFlags:
    case kIROp_MetalRTGetPrimitiveIndex:
    case kIROp_MetalRTGetGeometryIndex:
    case kIROp_MetalRTGetWorldSpaceOrigin:
    case kIROp_MetalRTGetWorldSpaceDirection:
    case kIROp_MetalRTGetDispatchRaysIndex:
    case kIROp_MetalRTGetDispatchRaysDimensions:
    case kIROp_MetalRTGetRayTime:
    case kIROp_MetalRTGetInstanceCount:
    case kIROp_MetalRTGetInstanceIndex:
    case kIROp_MetalRTGetInstanceID:
    case kIROp_MetalRTGetObjectSpaceRay:
    case kIROp_MetalRTGetObjectToWorld:
    case kIROp_MetalRTGetWorldToObject:
    case kIROp_MetalRTGetTriangleFrontFacing:
        return true;
    default:
        return false;
    }
}

void collectStageInputOperations(IRFunc* invoke, List<IRInst*>& operations)
{
    HashSet<IRFunc*> visited;
    List<IRInst*> reachable;
    collectOperations(invoke, visited, reachable);
    for (auto operation : reachable)
        if (isStageInputOperation(operation->getOp()))
            operations.add(operation);
}

StageRequirements getStageContractRequirements(Stage stage)
{
    StageRequirements result;
    result.add(MetalRTStageRequirement::Record);
    result.add(MetalRTStageRequirement::CallableDispatch);
    result.dispatchRaysIndex = true;
    result.dispatchRaysDimensions = true;
    if (stage == Stage::Callable)
        return result;

    result.minDistance = true;
    result.rayTime = true;
    result.rayFlags = true;
    result.add(MetalRTStageRequirement::Distance);
    result.add(MetalRTStageRequirement::WorldSpaceOrigin);
    result.add(MetalRTStageRequirement::WorldSpaceDirection);
    if (stage == Stage::Miss)
        return result;

    SLANG_RELEASE_ASSERT(
        stage == Stage::ClosestHit || stage == Stage::AnyHit || stage == Stage::Intersection);
    result.add(MetalRTStageRequirement::HitAttributes);
    result.add(MetalRTStageRequirement::TriangleBarycentricCoord);
    result.add(MetalRTStageRequirement::TriangleFrontFacing);
    result.add(MetalRTStageRequirement::CurveParameter);
    result.add(MetalRTStageRequirement::HitKind);
    result.add(MetalRTStageRequirement::PrimitiveIndex);
    result.add(MetalRTStageRequirement::GeometryIndex);
    result.add(MetalRTStageRequirement::ObjectSpaceRay);
    result.add(MetalRTStageRequirement::ObjectToWorld);
    result.add(MetalRTStageRequirement::WorldToObject);
    result.add(MetalRTStageRequirement::InstanceCount);
    result.add(MetalRTStageRequirement::InstanceIndexAtLevel);
    result.add(MetalRTStageRequirement::InstanceIDAtLevel);
    return result;
}

IRFuncType* getVisibleFunctionSignature(
    IRBuilder& builder,
    const StageRequirements& requirements,
    IRType* descriptorPointerType)
{
    List<IRType*> types;
    for (auto parameter : getVisibleParameters(builder, requirements, descriptorPointerType))
        types.add(parameter.type);
    return builder.getFuncType(types, builder.getVoidType());
}

IRFuncType* getCallableFunctionSignature(
    IRBuilder& builder,
    const StageRequirements& requirements,
    IRType* descriptorPointerType)
{
    List<IRType*> types;
    for (auto parameter : getCallableParameters(builder, requirements, descriptorPointerType))
        types.add(parameter.type);
    return builder.getFuncType(types, builder.getVoidType());
}

/// Creates the Metal function-table wrapper for a selected miss or closest-hit shader.
/// The wrapper receives physical arguments such as a payload address and hit distance, then
/// calls the source invoke method. This function also records which wrapper values supply reads
/// of the logical shader input. Later, lowerVisibleStageInputs passes those values through invoke
/// and its helpers and replaces the input-read instructions with the corresponding values.
VisibleAdapter createVisibleAdapter(
    IRModule* module,
    const SchemaStage& stage,
    const StageRequirements& tableRequirements,
    UnownedStringSlice symbolName,
    IRType* descriptorPointerType)
{
    SLANG_RELEASE_ASSERT(stage.stage == Stage::ClosestHit || stage.stage == Stage::Miss);

    // Create the exported function with the signature expected by its visible function table.
    // Both the table type and the parameters below use getVisibleParameters, so their types and
    // order agree. Mark this as a Metal visible function and keep its selected export name.
    IRBuilder builder(module);
    builder.setInsertInto(module->getModuleInst());
    VisibleAdapter result;
    result.function = builder.createFunc();
    auto functionType =
        getVisibleFunctionSignature(builder, tableRequirements, descriptorPointerType);
    result.function->setFullType(functionType);
    builder.addNameHintDecoration(result.function, symbolName);
    builder.addExportDecoration(result.function, symbolName);
    builder.addExternCppDecoration(result.function, symbolName);
    builder.addKeepAliveDecoration(result.function);
    IRInst* tableOperands[] = {
        functionType,
        builder.getIntValue(builder.getIntType(), IRIntegerValue(stage.stage))};
    auto tableType = builder.getType(kIROp_MetalVisibleFunctionTable, 2, tableOperands);
    builder.addDecoration(
        result.function,
        kIROp_MetalVisibleFunctionDecoration,
        tableOperands[1],
        tableType);

    builder.setInsertInto(result.function);
    builder.emitBlock();

    // Create the incoming parameters and keep two lookups while constructing this wrapper:
    //     parameters[Data] = %payload              // find an argument by its role
    //     values[MetalRTGetCurrentDistance] = %distance // find the value for an input read
    // The first lookup also covers arguments that need a cast or load, such as the payload and
    // record addresses. The second covers arguments associated with a getter instruction.
    // `seen` tracks (getter opcode, getter result type) pairs already saved in result.bindings.
    // These are C++ construction data; result.bindings is the list returned for later input
    // lowering. None of these dictionaries or lists becomes a shader buffer.
    Dictionary<ParameterKind, IRInst*> parameters;
    Dictionary<IROp, IRInst*> values;
    HashSet<KeyValuePair<IROp, IRType*>> seen;
    for (auto parameter : getVisibleParameters(builder, tableRequirements, descriptorPointerType))
    {
        auto value = builder.emitParam(parameter.type);
        builder.addNameHintDecoration(value, UnownedTerminatedStringSlice(parameter.name));
        // Global-context lowering uses this marker to find the incoming shared resources.
        if (parameter.kind == ParameterKind::GlobalResources)
            builder.addDecoration(value, kIROp_MetalRTGlobalResourceContextDecoration);
        parameters.add(parameter.kind, value);
        if (parameter.operation != kIROp_Invalid)
            values.add(parameter.operation, value);
        // These two uint3 arguments describe the originating ray-generation launch. Save both
        // values even when the source body does not read them yet. Consider
        // Miss.invoke calling a helper that contains CallShader. Later, lowerCallable creates
        // MetalRTGetDispatchRaysIndex/Dimensions in that helper to pass to the callable shader.
        // The binding saved here says, for example:
        //     (MetalRTGetDispatchRaysIndex, uint3) -> %dispatchRaysIndex
        // lowerVisibleStageInputs then passes this existing argument through invoke and the
        // helper, replacing the new getter. The wrapper needs no extra parameter.
        // Other getters are recorded by the body scan below, which supplies their result type.
        // For example, an instance-index getter returns uint, but its argument is a uint*.
        if (stage.isPresent && (parameter.operation == kIROp_MetalRTGetDispatchRaysIndex ||
                                parameter.operation == kIROp_MetalRTGetDispatchRaysDimensions))
        {
            seen.add({parameter.operation, parameter.type});
            result.bindings.add({parameter.operation, parameter.type, value});
        }
    }

    // A missing stage needs only the function signature and an empty body.
    if (!stage.isPresent)
    {
        builder.emitReturn();
        return result;
    }
    SLANG_RELEASE_ASSERT(stage.invoke && stage.input);

    // Keep the descriptor and whole records-buffer arguments in the returned description.
    if (auto descriptor = parameters.tryGetValue(ParameterKind::Descriptor))
        result.descriptorResources = *descriptor;
    result.records = parameters.getValue(ParameterKind::Records);

    // Find input reads in invoke and the ordinary helper functions it calls. For each kind of
    // read, construct its replacement value inside this wrapper. Leave the source instructions
    // in place for now: lowerVisibleStageInputs will forward these values through their callers.
    HashSet<IRFunc*> visited;
    List<IRInst*> operations;
    collectOperations(stage.invoke, visited, operations);
    for (auto operation : operations)
    {
        auto op = operation->getOp();
        auto type = operation->getDataType();
        IRInst* value = nullptr;
        // Recover a typed pointer to the caller's payload so writes update the same storage.
        if (op == kIROp_PortableRTGetPayload)
            value = getStageDataReference(
                builder,
                parameters.getValue(ParameterKind::Data),
                cast<IRPtrTypeBase>(type)->getValueType());
        else if (auto bound = values.tryGetValue(op))
            value = *bound;
        else if (op == kIROp_PortableRTGetRecord)
        {
            // Read this shader's Record value from the incoming record-data address. Empty
            // records have no bytes to load, so construct their value without using the address.
            if (isEmptyType(type))
                value = as<IRVoidType>(type) ? builder.getVoidValue()
                                             : builder.emitDefaultConstruct(type);
            else
            {
                auto address = parameters.getValue(ParameterKind::Record);
                // This is application buffer storage, not a pointer to an already-legalized
                // internal struct. UserPointer lets existing Metal storage lowering pack its
                // vectors/matrices consistently with reflected structured-buffer field offsets.
                value = builder.emitLoad(builder.emitBitCast(
                    builder.getPtrType(type, AddressSpace::UserPointer),
                    address));
            }
        }
        else if (op == kIROp_PortableRTGetHitAttributes)
        {
            // Triangle/curve attribute views carry no data themselves; their getters use the
            // separate barycentric/front-facing/curve parameters. Procedural attributes have an
            // application-defined type, which we load through the incoming attributes pointer.
            if (isBuiltinAttributeView(operation))
                value = builder.emitDefaultConstruct(type);
            else
            {
                value = builder.emitLoad(getStageDataReference(
                    builder,
                    parameters.getValue(ParameterKind::HitAttributes),
                    type));
            }
        }
        else if (op == kIROp_MetalRTGetObjectSpaceRay)
        {
            // The signature passes the ray components separately. Assemble the RayDesc value
            // expected by this getter from origin, minimum distance, direction and hit distance.
            IRInst* fields[] = {
                parameters.getValue(ParameterKind::ObjectOrigin),
                values.getValue(kIROp_MetalRTGetMinDistance),
                parameters.getValue(ParameterKind::ObjectDirection),
                values.getValue(kIROp_MetalRTGetCurrentDistance)};
            value = builder.emitMakeStruct(type, SLANG_COUNT_OF(fields), fields);
        }
        // Save one rule per opcode/result type, even if several helpers contain the same read.
        // For example, all float MetalRTGetCurrentDistance reads use this wrapper's %distance.
        if (value && seen.add({op, type}))
            result.bindings.add({op, type, value});
    }

    // Call the original method with its empty logical input. Later input lowering adds the
    // arguments needed by its body and helpers, using the replacement rules recorded above.
    emitInvoke(builder, stage.invoke);
    builder.emitReturn();
    return result;
}

VisibleAdapter createCallableAdapter(
    IRModule* module,
    const SchemaStage& stage,
    const StageRequirements& tableRequirements,
    UnownedStringSlice symbolName,
    IRType* descriptorPointerType)
{
    SLANG_RELEASE_ASSERT(
        stage.stage == Stage::Callable && stage.isPresent && stage.invoke && stage.input);
    IRBuilder builder(module);
    builder.setInsertInto(module->getModuleInst());
    VisibleAdapter result;
    result.function = builder.createFunc();
    auto functionType =
        getCallableFunctionSignature(builder, tableRequirements, descriptorPointerType);
    result.function->setFullType(functionType);
    builder.addNameHintDecoration(result.function, symbolName);
    builder.addExportDecoration(result.function, symbolName);
    builder.addExternCppDecoration(result.function, symbolName);
    builder.addKeepAliveDecoration(result.function);
    IRInst* operands[] = {
        functionType,
        builder.getIntValue(builder.getIntType(), IRIntegerValue(Stage::Callable))};
    auto tableType = builder.getType(kIROp_MetalVisibleFunctionTable, 2, operands);
    builder.addDecoration(
        result.function,
        kIROp_MetalVisibleFunctionDecoration,
        operands[1],
        tableType);
    builder.setInsertInto(result.function);
    builder.emitBlock();
    Dictionary<ParameterKind, IRInst*> parameters;
    Dictionary<IROp, IRInst*> values;
    HashSet<KeyValuePair<IROp, IRType*>> seen;
    for (auto parameter : getCallableParameters(builder, tableRequirements, descriptorPointerType))
    {
        auto value = builder.emitParam(parameter.type);
        builder.addNameHintDecoration(value, UnownedTerminatedStringSlice(parameter.name));
        if (parameter.kind == ParameterKind::GlobalResources)
            builder.addDecoration(value, kIROp_MetalRTGlobalResourceContextDecoration);
        parameters.add(parameter.kind, value);
        if (parameter.operation != kIROp_Invalid)
            values.add(parameter.operation, value);
        // Nested dispatch can create launch queries after this adapter's body is inspected.
        // Their fixed uint3 result types already match these invocation parameters.
        if (parameter.operation == kIROp_MetalRTGetDispatchRaysIndex ||
            parameter.operation == kIROp_MetalRTGetDispatchRaysDimensions)
        {
            seen.add({parameter.operation, parameter.type});
            result.bindings.add({parameter.operation, parameter.type, value});
        }
    }
    result.descriptorResources = parameters.getValue(ParameterKind::Descriptor);
    result.records = parameters.getValue(ParameterKind::Records);
    auto data = getStageDataReference(
        builder,
        parameters.getValue(ParameterKind::Data),
        stage.input->getPayloadType());
    values.add(kIROp_PortableRTGetCallableData, data);
    HashSet<IRFunc*> visited;
    List<IRInst*> operations;
    collectOperations(stage.invoke, visited, operations);
    for (auto operation : operations)
    {
        auto op = operation->getOp();
        auto type = operation->getDataType();
        IRInst* value = nullptr;
        if (auto bound = values.tryGetValue(op))
            value = *bound;
        else if (op == kIROp_PortableRTGetRecord)
        {
            if (isEmptyType(type))
                value = as<IRVoidType>(type) ? builder.getVoidValue()
                                             : builder.emitDefaultConstruct(type);
            else
                value = builder.emitLoad(builder.emitBitCast(
                    builder.getPtrType(type, AddressSpace::UserPointer),
                    parameters.getValue(ParameterKind::Record)));
        }
        if (value && seen.add({op, type}))
            result.bindings.add({op, type, value});
    }
    emitInvoke(builder, stage.invoke);
    builder.emitReturn();
    return result;
}

void lowerVisibleStageInputs(IRModule* module, const List<VisibleAdapter>& adapters)
{
    // Different schemas can bind the same logical getter through different physical pointer or
    // path types. Partition by those canonical types rather than erasing their address spaces.
    using Query = KeyValuePair<IROp, IRType*>;
    using BindingKey = KeyValuePair<Query, IRType*>;
    List<BindingKey> keys;
    HashSet<BindingKey> seen;
    for (const auto& adapter : adapters)
        for (const auto& binding : adapter.bindings)
        {
            BindingKey key({binding.op, binding.resultType}, binding.value->getDataType());
            if (seen.add(key))
                keys.add(key);
        }
    for (const auto& key : keys)
    {
        PortableRTInvocationParameter parameter(module, key.value, "invocationValue");
        HashSet<IRFunc*> visited;
        List<IRInst*> operations;
        for (const auto& adapter : adapters)
            for (const auto& binding : adapter.bindings)
                if (binding.op == key.key.key && binding.resultType == key.key.value &&
                    binding.value->getDataType() == key.value)
                {
                    parameter.bind(adapter.function, binding.value);
                    collectOperations(adapter.function, visited, operations);
                }
        // Threading replaces call instructions while walking callers. Keep only the accessors
        // being lowered, not snapshots of calls that those updates can deallocate.
        List<IRInst*> queries;
        for (auto operation : operations)
        {
            if (operation->getOp() == key.key.key && operation->getDataType() == key.key.value)
                queries.add(operation);
        }
        for (auto operation : queries)
        {
            if (operation->getOp() != kIROp_MetalRTGetInstanceIndex &&
                operation->getOp() != kIROp_MetalRTGetInstanceID)
            {
                parameter.legalize(operation);
                continue;
            }
            auto value = parameter.addParameterAndUpdateCallers(getParentFunc(operation));
            // Every hit-stage ABI passes a thread pointer plus a count. Ordinary instancing
            // supplies one element, and its scalar source getter indexes that element at zero.
            IRBuilder builder(module);
            SLANG_RELEASE_ASSERT(as<IRPtrType>(value->getDataType()));
            builder.setInsertBefore(operation);
            operation->replaceUsesWith(
                builder.emitLoad(builder.emitGetOffsetPtr(value, operation->getOperand(0))));
            operation->removeAndDeallocate();
        }
    }
}

} // namespace Metal
} // namespace PortableRT

List<PortableRT::Metal::VisibleAdapter> createMetalStageAdapters(
    IRModule* module,
    TargetProgram*,
    List<IRFunc*>& entryPoints,
    bool& requiresResultTypeLowering,
    DiagnosticSink* sink)
{
    using namespace PortableRT::Metal;

    struct SelectedStage
    {
        PortableRT::SchemaStage stage;
        String name;
    };

    List<VisibleAdapter> result;
    List<IRFunc*> nativeEntries;
    List<SelectedStage> selectedStages;
    IRBuilder builder(module);
    for (auto invoke : entryPoints)
    {
        auto input = PortableRT::getEntryPointInputInfo(invoke);
        if (!input)
        {
            nativeEntries.add(invoke);
            continue;
        }
        auto entry = invoke->findDecoration<IREntryPointDecoration>();
        SLANG_RELEASE_ASSERT(entry);

        // Selecting Miss supplies a checked invoke and Context without selecting a program
        // schema. The same source contract fixes the VFT signature in the caller's compilation;
        // only this adapter needs the implementation and its logical input decoration.
        SelectedStage selected;
        selected.stage.stage = entry->getProfile().getStage();
        selected.stage.isPresent = true;
        selected.stage.invoke = invoke;
        selected.stage.input = input;
        selected.name = entry->getName()->getStringSlice();
        selectedStages.add(_Move(selected));

        // The final adapter owns the selected export. Keep the original function callable,
        // without an entry-point signature or its reflected shader-record parameter layout.
        PortableRT::demoteStageImplementation(invoke);
        if (auto keepAlive = invoke->findDecoration<IRKeepAliveDecoration>())
            keepAlive->removeAndDeallocate();
        if (auto layout = invoke->findDecoration<IRLayoutDecoration>())
            layout->removeAndDeallocate();
    }

    // Candidate adapters clone parts of the source call graph. Remove remaining entry markers
    // before cloning, so linked implementations used as helpers also remain ordinary functions.
    PortableRT::demoteStageImplementations(module);

    for (const auto& selected : selectedStages)
    {
        const auto& stage = selected.stage;
        auto input = stage.input;
        auto symbolName = selected.name.getUnownedSlice();
        VisibleAdapter root;
        if (stage.stage == Stage::AnyHit || stage.stage == Stage::Intersection)
        {
            root = createSelectedCandidateAdapter(
                module,
                stage,
                symbolName,
                requiresResultTypeLowering,
                sink);
        }
        else
        {
            auto requirements = getStageContractRequirements(stage.stage);
            auto descriptorPointer =
                builder.getPtrType(builder.getUIntType(), AddressSpace::Uniform);
            if (stage.stage == Stage::Callable)
            {
                root = createCallableAdapter(
                    module,
                    stage,
                    requirements,
                    symbolName,
                    descriptorPointer);
            }
            else
            {
                SLANG_RELEASE_ASSERT(
                    stage.stage == Stage::ClosestHit || stage.stage == Stage::Miss);
                root = createVisibleAdapter(
                    module,
                    stage,
                    requirements,
                    symbolName,
                    descriptorPointer);
            }
        }
        result.add(_Move(root));
    }
    entryPoints = _Move(nativeEntries);
    return result;
}

} // namespace Slang
