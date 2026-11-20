#include "slang-ir-metal-portable-ray-tracing-stages.h"

#include "slang-ir-any-value-marshalling.h"
#include "slang-ir-metal-portable-ray-tracing-candidates.h"
#include "slang-ir-portable-ray-tracing-util.h"
#include "slang-ir-util.h"
#include "slang-target-program.h"

namespace Slang
{
namespace PortableRT
{
namespace Metal
{

RayDataHeader createRayDataHeader(
    IRModule* module,
    const MetalRTCandidateResources& candidateResources)
{
    IRBuilder builder(module);
    builder.setInsertInto(module);
    RayDataHeader result;
    result.type = builder.createStructType();
    builder.addNameHintDecoration(result.type, toSlice("PortableRTRayDataHeader"));
    auto addField = [&](IRType* type, const char* name)
    {
        auto key = builder.createStructKey();
        builder.addNameHintDecoration(key, UnownedTerminatedStringSlice(name));
        builder.createStructField(result.type, key, type);
        return key;
    };
    auto uintType = builder.getUIntType();
    auto uintPointer = builder.getPtrType(uintType, AddressSpace::Global);
    result.recordsKey = addField(
        builder.getPtrType(
            uintType,
            AccessQualifier::Read,
            AddressSpace::Global,
            builder.getDefaultBufferLayoutType()),
        "records");
    result.instanceContributionMetadataKey = addField(uintPointer, "instanceContributionMetadata");
    result.candidateResourcesKey = addField(
        builder.getPtrType(candidateResources.type, AddressSpace::Uniform),
        "candidateResources");
    result.sbtOffsetKey = addField(uintType, "sbtOffset");
    result.sbtStrideKey = addField(uintType, "sbtStride");
    result.hitRecordStrideKey = addField(uintType, "hitRecordStride");
    result.committedAttributesOffsetKey = addField(uintType, "committedAttributesOffset");
    result.minDistanceKey = addField(builder.getFloatType(), "minDistance");
    result.rayTimeKey = addField(builder.getFloatType(), "time");
    result.rayFlagsKey = addField(uintType, "rayFlags");
    auto uint3 = builder.getVectorType(uintType, 3);
    result.dispatchRaysIndexKey = addField(uint3, "dispatchIndex");
    result.dispatchRaysDimensionsKey = addField(uint3, "dispatchDimensions");
    result.customHitKindKey = addField(uintType, "customHitKind");
    return result;
}

RayData createRayDataPrefix(IRModule* module, const RayDataHeader& header, IRType* payloadType)
{
    IRBuilder builder(module);
    builder.setInsertInto(module);
    RayData result;
    result.type = builder.createStructType();
    builder.addNameHintDecoration(result.type, toSlice("PortableRTRayData"));
    result.header = &header;
    result.headerKey = builder.createStructKey();
    builder.addNameHintDecoration(result.headerKey, toSlice("header"));
    builder.createStructField(result.type, result.headerKey, header.type);
    if (!isEmptyType(payloadType))
    {
        result.payloadKey = builder.createStructKey();
        builder.addNameHintDecoration(result.payloadKey, toSlice("payload"));
        builder.createStructField(result.type, result.payloadKey, payloadType);
    }
    return result;
}

IRType* getRayPayloadStorageType(IRBuilder& builder, IRType* payloadType)
{
    return isEmptyType(payloadType) ? builder.getVoidType() : payloadType;
}

/// Builds an IR pointer to stage data using the concrete type expected by the selected shader.
/// Invocation contexts store payload, callable data, and attributes as thread void*
/// so independently compiled shaders agree on the field types. The selected adapter knows
/// the concrete type from the shader's input contract and restores that type here.
///
/// Consider a miss shader whose Context.Payload is `struct Payload { float value; }`.
/// The generated caller and adapter conceptually do the following (other fields omitted):
///     // Caller: pass the address of the actual payload through the common table signature.
///     Payload payload;
///     MissInvocationContext context = { (thread void*)&payload, ... };
///     missTable[index](&context);
///     // Selected miss adapter receives &context as invocation and restores the payload type.
///     thread Payload* typedPayload = (thread Payload*)invocation->payload;
///
/// `untypedDataPointer` is the loaded invocation->payload field, and `dataType` is Payload.
/// The returned pointer lets later input lowering pass the same payload storage to invoke and
/// its helpers.
/// For nonempty data, the caller must supply storage of the matching type in the thread address
/// space; this function does not discover or validate the storage's type at runtime.
/// Void and empty structs need no transported bytes, so those cases ignore untypedDataPointer.
IRInst* recoverStageDataPointer(IRBuilder& builder, IRInst* untypedDataPointer, IRType* dataType)
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
        // local of the correct type. There are no payload bytes to recover from untypedDataPointer.
        // Later empty-type legalization removes this local and the empty arguments using it.
        auto value = builder.emitVar(dataType);
        builder.emitStore(value, builder.emitDefaultConstruct(dataType));
        return value;
    }
    // Nonempty data needs an address operand. This checks the presence of the IR instruction,
    // not whether the pointer it represents is non-null at runtime.
    SLANG_RELEASE_ASSERT(untypedDataPointer);
    // Cast thread void* to thread dataType*. Only the pointer's pointee type changes; its
    // address and address space stay the same. This emits no load, store, or copy of the data.
    // For example, writes through the recovered Payload* update the caller's original payload.
    return builder.emitBitCast(
        builder.getPtrType(dataType, AddressSpace::ThreadLocal),
        untypedDataPointer);
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
    return recoverStageDataPointer(builder, nullptr, payloadType);
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

/// Defines one physical context field and the logical getter it directly supplies, if any.
struct InvocationFieldInfo
{
    MetalRTInvocationField role;
    IRType* type;
    const char* name;
    IROp operation = kIROp_Invalid;
};

/// Defines the fixed field types and order shared by independently compiled callers and stages.
/// The getter opcode identifies fields that directly supply a logical input property.
List<InvocationFieldInfo> getVisibleContextFields(
    IRBuilder& builder,
    const StageRequirements& requirements,
    IRType* descriptorPointerType)
{
    List<InvocationFieldInfo> result;
    auto floatType = builder.getFloatType();
    auto uintType = builder.getUIntType();
    auto float3Type = builder.getVectorType(floatType, 3);
    auto matrixType = builder.getMatrixType(
        floatType,
        builder.getIntValue(builder.getIntType(), 4),
        builder.getIntValue(builder.getIntType(), 3),
        builder.getIntValue(builder.getIntType(), kMatrixLayoutMode_RowMajor));
    result.add(
        {MetalRTInvocationField::Data,
         builder.getPtrType(builder.getVoidType(), AddressSpace::ThreadLocal),
         "payload"});
    if (requirements.has(MetalRTStageRequirement::HitAttributes))
        result.add(
            {MetalRTInvocationField::HitAttributes,
             builder.getPtrType(builder.getVoidType(), AddressSpace::ThreadLocal),
             "attributes"});
    if (requirements.has(MetalRTStageRequirement::Distance) ||
        requirements.has(MetalRTStageRequirement::ObjectSpaceRay))
        result.add(
            {MetalRTInvocationField::Distance,
             floatType,
             "distance",
             kIROp_MetalRTGetCurrentDistance});
    if (requirements.has(MetalRTStageRequirement::HitKind))
        result.add({MetalRTInvocationField::HitKind, uintType, "hitKind", kIROp_MetalRTGetHitKind});
    if (requirements.has(MetalRTStageRequirement::TriangleBarycentricCoord))
        result.add(
            {MetalRTInvocationField::Barycentrics,
             builder.getVectorType(floatType, 2),
             "barycentricCoord",
             kIROp_PortableRTGetTriangleBarycentricCoord});
    if (requirements.has(MetalRTStageRequirement::TriangleFrontFacing))
        result.add(
            {MetalRTInvocationField::FrontFacing,
             builder.getBoolType(),
             "frontFacing",
             kIROp_MetalRTGetTriangleFrontFacing});
    if (requirements.has(MetalRTStageRequirement::CurveParameter))
        result.add(
            {MetalRTInvocationField::CurveParameter,
             floatType,
             "curveParameter",
             kIROp_PortableRTGetCurveParameter});
    if (requirements.has(MetalRTStageRequirement::WorldSpaceOrigin))
        result.add(
            {MetalRTInvocationField::WorldOrigin,
             float3Type,
             "worldSpaceOrigin",
             kIROp_MetalRTGetWorldSpaceOrigin});
    if (requirements.has(MetalRTStageRequirement::WorldSpaceDirection))
        result.add(
            {MetalRTInvocationField::WorldDirection,
             float3Type,
             "worldSpaceDirection",
             kIROp_MetalRTGetWorldSpaceDirection});
    if (requirements.has(MetalRTStageRequirement::PrimitiveIndex))
        result.add(
            {MetalRTInvocationField::PrimitiveIndex,
             uintType,
             "primitiveIndex",
             kIROp_MetalRTGetPrimitiveIndex});
    if (requirements.has(MetalRTStageRequirement::GeometryIndex))
        result.add(
            {MetalRTInvocationField::GeometryIndex,
             uintType,
             "geometryIndex",
             kIROp_MetalRTGetGeometryIndex});
    if (requirements.has(MetalRTStageRequirement::InstanceCount))
        result.add(
            {MetalRTInvocationField::InstanceCount,
             uintType,
             "instanceCount",
             kIROp_MetalRTGetInstanceCount});
    if (requirements.has(MetalRTStageRequirement::InstanceIndexAtLevel))
        result.add(
            {MetalRTInvocationField::InstanceIndices,
             builder.getPtrType(uintType, AddressSpace::ThreadLocal),
             "instanceIndices",
             kIROp_MetalRTGetInstanceIndex});
    if (requirements.has(MetalRTStageRequirement::InstanceIDAtLevel))
        result.add(
            {MetalRTInvocationField::InstanceIDs,
             builder.getPtrType(uintType, AddressSpace::ThreadLocal),
             "instanceIDs",
             kIROp_MetalRTGetInstanceID});
    if (requirements.has(MetalRTStageRequirement::ObjectSpaceRay))
    {
        result.add({MetalRTInvocationField::ObjectOrigin, float3Type, "objectSpaceOrigin"});
        result.add({MetalRTInvocationField::ObjectDirection, float3Type, "objectSpaceDirection"});
    }
    if (requirements.has(MetalRTStageRequirement::ObjectToWorld))
        result.add(
            {MetalRTInvocationField::ObjectToWorld,
             matrixType,
             "objectToWorld",
             kIROp_MetalRTGetObjectToWorld});
    if (requirements.has(MetalRTStageRequirement::WorldToObject))
        result.add(
            {MetalRTInvocationField::WorldToObject,
             matrixType,
             "worldToObject",
             kIROp_MetalRTGetWorldToObject});
    if (requirements.has(MetalRTStageRequirement::Record))
        result.add(
            {MetalRTInvocationField::Record,
             builder.getPtrType(builder.getUInt8Type(), AddressSpace::Global),
             "recordData"});
    if (requirements.has(MetalRTStageRequirement::CallableDispatch))
        result.add(
            {MetalRTInvocationField::Descriptor, descriptorPointerType, "descriptorResources"});
    result.add(
        {MetalRTInvocationField::MinDistance,
         floatType,
         "minDistance",
         kIROp_MetalRTGetMinDistance});
    result.add({MetalRTInvocationField::RayTime, floatType, "rayTime", kIROp_MetalRTGetRayTime});
    result.add({MetalRTInvocationField::RayFlags, uintType, "rayFlags", kIROp_MetalRTGetRayFlags});
    auto uint3 = builder.getVectorType(uintType, 3);
    result.add(
        {MetalRTInvocationField::DispatchIndex,
         uint3,
         "dispatchRaysIndex",
         kIROp_MetalRTGetDispatchRaysIndex});
    result.add(
        {MetalRTInvocationField::DispatchDimensions,
         uint3,
         "dispatchRaysDimensions",
         kIROp_MetalRTGetDispatchRaysDimensions});
    result.add(
        {MetalRTInvocationField::Records,
         builder.getPtrType(
             uintType,
             AccessQualifier::Read,
             AddressSpace::Global,
             builder.getDefaultBufferLayoutType()),
         "records"});
    result.add(
        {MetalRTInvocationField::GlobalResources,
         builder.getPtrType(builder.getUInt8Type(), AddressSpace::ThreadLocal),
         "globalResources"});
    return result;
}

/// Defines the callable context independently of the source CallableData and Record types.
/// The selected receiver restores those types from the erased addresses.
List<InvocationFieldInfo> getCallableContextFields(
    IRBuilder& builder,
    const StageRequirements& requirements,
    IRType* descriptorPointerType)
{
    SLANG_RELEASE_ASSERT(
        requirements.has(MetalRTStageRequirement::Record) &&
        requirements.has(MetalRTStageRequirement::CallableDispatch));
    List<InvocationFieldInfo> result;
    result.add(
        {MetalRTInvocationField::Data,
         builder.getPtrType(builder.getVoidType(), AddressSpace::ThreadLocal),
         "data"});
    auto uint3 = builder.getVectorType(builder.getUIntType(), 3);
    result.add(
        {MetalRTInvocationField::DispatchIndex,
         uint3,
         "dispatchRaysIndex",
         kIROp_MetalRTGetDispatchRaysIndex});
    result.add(
        {MetalRTInvocationField::DispatchDimensions,
         uint3,
         "dispatchRaysDimensions",
         kIROp_MetalRTGetDispatchRaysDimensions});
    result.add(
        {MetalRTInvocationField::Record,
         builder.getPtrType(builder.getUInt8Type(), AddressSpace::Global),
         "recordData"});
    result.add({MetalRTInvocationField::Descriptor, descriptorPointerType, "descriptorResources"});
    result.add(
        {MetalRTInvocationField::Records,
         builder.getPtrType(
             builder.getUIntType(),
             AccessQualifier::Read,
             AddressSpace::Global,
             builder.getDefaultBufferLayoutType()),
         "records"});
    result.add(
        {MetalRTInvocationField::GlobalResources,
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
    if (stage == Stage::Callable)
        return result;

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

IRStructType* InvocationContexts::get(Stage stage) const
{
    switch (stage)
    {
    case Stage::ClosestHit:
        return closestHit;
    case Stage::Miss:
        return miss;
    case Stage::Callable:
        return callable;
    default:
        SLANG_UNEXPECTED("Not a post-traversal or callable stage");
    }
}

/// Builds an ordinary struct with a fixed ABI. Field roles survive later renaming and let the
/// emitter fill the context while the receiver loads only properties used by its shader.
IRStructType* createInvocationContext(IRModule* module, Stage stage, const char* name)
{
    IRBuilder builder(module);
    builder.setInsertInto(module);
    auto type = builder.createStructType();
    builder.addNameHintDecoration(type, UnownedTerminatedStringSlice(name));
    auto descriptorPointer = builder.getPtrType(builder.getUIntType(), AddressSpace::Uniform);
    auto requirements = getStageContractRequirements(stage);
    auto fields = stage == Stage::Callable
                      ? getCallableContextFields(builder, requirements, descriptorPointer)
                      : getVisibleContextFields(builder, requirements, descriptorPointer);
    for (const auto& field : fields)
    {
        auto key = builder.createStructKey();
        builder.addNameHintDecoration(key, UnownedTerminatedStringSlice(field.name));
        builder.addDecoration(
            key,
            kIROp_MetalRTInvocationFieldDecoration,
            builder.getIntValue(builder.getIntType(), IRIntegerValue(field.role)),
            builder.getIntValue(builder.getIntType(), IRIntegerValue(field.operation)));
        builder.createStructField(type, key, field.type);
    }
    return type;
}

InvocationContexts createInvocationContexts(IRModule* module)
{
    return {
        createInvocationContext(module, Stage::ClosestHit, "ClosestHitInvocationContext"),
        createInvocationContext(module, Stage::Miss, "MissInvocationContext"),
        createInvocationContext(module, Stage::Callable, "CallableInvocationContext")};
}

IRFuncType* getInvocationFunctionSignature(IRBuilder& builder, IRStructType* contextType)
{
    IRType* parameter = builder.getPtrType(contextType, AddressSpace::ThreadLocal);
    return builder.getFuncType(1, &parameter, builder.getVoidType());
}

/// Emits IR loads from the invocation context passed to a Metal visible-function wrapper.
/// An invocation context is a compiler-generated per-thread struct containing payload/Record
/// addresses, built-in values, and resource pointers for the current shader invocation. Its
/// layout is fixed for each stage so separately compiled callers and wrappers agree.
///
/// Consider a Miss::invoke that reads input.distance. The generated code is conceptually:
///     // Caller: fill the context and pass its address through the visible-function table.
///     MissInvocationContext context = { payloadAddress, rayTMax, ... };
///     missFunctions[index](&context);
///     // Selected wrapper: read the field needed by the source shader.
///     void miss(thread MissInvocationContext* invocation)
///     {
///         Miss_invoke(invocation->distance, ...);
///     }
///
/// This helper emits the field-address/load instructions for those wrapper reads. A getter
/// opcode selects a directly mapped field such as distance; a field role selects components
/// used to build values such as objectSpaceRay.
struct InvocationContextReader
{
    IRBuilder& builder;
    IRInst* context;
    IRStructType* type;

    IRInst* load(IRStructField* field)
    {
        return builder.emitLoad(builder.emitFieldAddress(
            builder.getPtrType(field->getFieldType(), AddressSpace::ThreadLocal),
            context,
            field->getKey()));
    }

    IRInst* load(MetalRTInvocationField role)
    {
        for (auto field : type->getFields())
        {
            auto info = field->getKey()->findDecoration<IRMetalRTInvocationFieldDecoration>();
            SLANG_RELEASE_ASSERT(info);
            if (getIntVal(info->getRole()) == IRIntegerValue(role))
                return load(field);
        }
        SLANG_UNEXPECTED("Required invocation field is absent");
    }

    IRInst* tryLoadInput(IROp op)
    {
        for (auto field : type->getFields())
        {
            auto info = field->getKey()->findDecoration<IRMetalRTInvocationFieldDecoration>();
            SLANG_RELEASE_ASSERT(info);
            if (getIntVal(info->getOperation()) == IRIntegerValue(op))
                return load(field);
        }
        return nullptr;
    }
};

/// Creates the Metal function-table wrapper for a selected miss, closest-hit, or callable shader.
/// The wrapper receives a thread pointer to the fixed invocation context and loads the fields
/// used by the source invoke method. This function also records which wrapper values supply reads
/// of the logical shader input. Later, lowerVisibleStageInputs passes those values through invoke
/// and its helpers and replaces the input-read instructions with the corresponding values.
VisibleAdapter createVisibleAdapter(
    IRModule* module,
    TargetRequest* target,
    const SchemaStage& stage,
    IRStructType* contextType,
    UnownedStringSlice symbolName)
{
    SLANG_RELEASE_ASSERT(
        stage.stage == Stage::ClosestHit || stage.stage == Stage::Miss ||
        stage.stage == Stage::Callable);
    SLANG_RELEASE_ASSERT(stage.isPresent && stage.invoke && stage.input);

    // Create the exported function with the signature expected by its visible function table.
    // The table and its receiver share the canonical context type. Its field layout is fixed
    // by stage kind, including fields this particular shader does not use.
    IRBuilder builder(module);
    builder.setInsertInto(module->getModuleInst());
    VisibleAdapter result;
    result.function = builder.createFunc();
    auto functionType = getInvocationFunctionSignature(builder, contextType);
    result.function->setFullType(functionType);
    builder.addNameHintDecoration(result.function, symbolName);
    builder.addExportDecoration(result.function, symbolName);
    builder.addExternCppDecoration(result.function, symbolName);
    builder.addKeepAliveDecoration(result.function);
    // Mark the wrapper [[visible]] and record its stage and function-table signature.
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

    auto context = builder.emitParam(builder.getPtrType(contextType, AddressSpace::ThreadLocal));
    builder.addNameHintDecoration(context, toSlice("invocation"));
    InvocationContextReader inputs{builder, context, contextType};
    HashSet<KeyValuePair<IROp, IRType*>> seen;

    // Lowering trace/callShader later will require us to pass the dispatch index and dimensions,
    // even though the current invoke doesn't read them explicitly, so we also need to save those
    // values now.
    IROp launchOps[] = {kIROp_MetalRTGetDispatchRaysIndex, kIROp_MetalRTGetDispatchRaysDimensions};
    for (auto op : launchOps)
    {
        auto value = inputs.tryLoadInput(op);
        SLANG_RELEASE_ASSERT(value);
        seen.add({op, value->getDataType()});
        result.bindings.add({op, value->getDataType(), value});
    }

    // Nested dispatch also needs the program descriptor and the complete records buffer.
    result.descriptorResources = inputs.load(MetalRTInvocationField::Descriptor);
    result.records = inputs.load(MetalRTInvocationField::Records);

    // Find input reads in invoke and the ordinary helper functions it calls. For each kind of
    // read, construct its replacement value inside this wrapper. Leave the source instructions
    // in place for now: lowerVisibleStageInputs will forward these values through their callers.
    List<IRInst*> operations;
    collectStageInputOperations(stage.invoke, operations);
    for (auto operation : operations)
    {
        auto op = operation->getOp();
        auto type = operation->getDataType();
        if (!seen.add({op, type}))
            continue;
        IRInst* value = nullptr;
        // The caller passes payload/callable data as void*. Recover the pointer's concrete type.
        if (op == kIROp_PortableRTGetPayload || op == kIROp_PortableRTGetCallableData)
            value = recoverStageDataPointer(
                builder,
                inputs.load(MetalRTInvocationField::Data),
                cast<IRPtrTypeBase>(type)->getValueType());
        else if (auto bound = inputs.tryLoadInput(op))
            value = bound;
        else if (op == kIROp_PortableRTGetRecord)
        {
            // Read this shader's Record value from the incoming record-data address. Empty
            // records have no bytes to load, so construct their value without using the address.
            if (isEmptyType(type))
                value = as<IRVoidType>(type) ? builder.getVoidValue()
                                             : builder.emitDefaultConstruct(type);
            else
            {
                auto address = inputs.load(MetalRTInvocationField::Record);
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
            // separate barycentric/front-facing/curve fields. Procedural attributes have an
            // application-defined type packed in the shared committed-attribute storage. Its
            // packing is independent of schema order and native Metal float3 padding.
            if (isBuiltinAttributeView(operation))
                value = builder.emitDefaultConstruct(type);
            else
            {
                if (isEmptyType(type))
                    value = builder.emitDefaultConstruct(type);
                else
                {
                    auto packedSize = getAnyValueSize(type, target);
                    SLANG_RELEASE_ASSERT(packedSize > 0);
                    auto packedType = builder.getAnyValueType(packedSize);
                    auto address = builder.emitBitCast(
                        builder.getPtrType(packedType, AddressSpace::ThreadLocal),
                        inputs.load(MetalRTInvocationField::HitAttributes));
                    value = builder.emitUnpackAnyValue(type, builder.emitLoad(address));
                }
            }
        }
        else if (op == kIROp_MetalRTGetObjectSpaceRay)
        {
            // The context stores the ray components separately. Assemble the RayDesc value
            // expected by this getter from origin, minimum distance, direction and hit distance.
            IRInst* fields[] = {
                inputs.load(MetalRTInvocationField::ObjectOrigin),
                inputs.load(MetalRTInvocationField::MinDistance),
                inputs.load(MetalRTInvocationField::ObjectDirection),
                inputs.load(MetalRTInvocationField::Distance)};
            value = builder.emitMakeStruct(type, SLANG_COUNT_OF(fields), fields);
        }
        // Save one rule per opcode/result type, even if several helpers contain the same read.
        // For example, all float MetalRTGetCurrentDistance reads use the same context field load.
        if (value)
            result.bindings.add({op, type, value});
    }

    // Call the original method with its empty logical input. Later input lowering adds the
    // arguments needed by its body and helpers, using the replacement rules recorded above.
    emitInvoke(builder, stage.invoke);
    builder.emitReturn();
    return result;
}

void lowerVisibleStageInputs(IRModule* module, const List<VisibleAdapter>& adapters)
{
    // Different stage contexts can bind the same logical getter through different pointer or
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
    TargetProgram* targetProgram,
    List<IRFunc*>& entryPoints,
    const PortableRT::Metal::RayDataHeader& header,
    const MetalRTCandidateResources& candidateResources,
    const PortableRT::Metal::InvocationContexts& invocationContexts,
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
        auto symbolName = selected.name.getUnownedSlice();
        VisibleAdapter root;
        if (stage.stage == Stage::AnyHit || stage.stage == Stage::Intersection)
        {
            root = createSelectedCandidateAdapter(
                module,
                targetProgram->getTargetReq(),
                stage,
                header,
                candidateResources,
                symbolName,
                requiresResultTypeLowering,
                sink);
        }
        else
        {
            root = createVisibleAdapter(
                module,
                targetProgram->getTargetReq(),
                stage,
                invocationContexts.get(stage.stage),
                symbolName);
        }
        result.add(_Move(root));
    }
    entryPoints = _Move(nativeEntries);
    return result;
}

} // namespace Slang
