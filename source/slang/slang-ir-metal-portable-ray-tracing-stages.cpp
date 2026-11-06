#include "slang-ir-metal-portable-ray-tracing-stages.h"

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

IRInst* getStageDataReference(IRBuilder& builder, IRInst* erasedAddress, IRType* dataType)
{
    if (as<IRVoidType>(dataType))
    {
        // A retained helper may still take a nominal void reference. It has no storage, so use
        // a typed null pointer instead of constructing an invalid void local or store.
        return builder.getNullPtrValue(builder.getPtrType(dataType, AddressSpace::ThreadLocal));
    }
    if (isEmptyType(dataType))
    {
        // Consider Payload = ApplicationEmpty. Its nominal reference is still meaningful to
        // ordinary source calls even though transport has no bytes. Empty-type legalization
        // removes this local token without dereferencing the erased ABI pointer.
        auto value = builder.emitVar(dataType);
        builder.emitStore(value, builder.emitDefaultConstruct(dataType));
        return value;
    }
    SLANG_RELEASE_ASSERT(erasedAddress);
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
    InstanceIndex,
    InstanceID,
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
    if (requirements.has(MetalRTStageRequirement::InstanceIndex))
        result.add(
            {ParameterKind::InstanceIndex,
             uintType,
             "instanceIndex",
             kIROp_MetalRTGetInstanceIndex});
    if (requirements.has(MetalRTStageRequirement::InstanceID))
        result.add({ParameterKind::InstanceID, uintType, "instanceID", kIROp_MetalRTGetInstanceID});
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

StageRequirements getStageContractRequirements(Stage stage, bool multiLevel)
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
    if (multiLevel)
    {
        result.add(MetalRTStageRequirement::InstanceCount);
        result.add(MetalRTStageRequirement::InstanceIndexAtLevel);
        result.add(MetalRTStageRequirement::InstanceIDAtLevel);
    }
    else
    {
        result.add(MetalRTStageRequirement::InstanceIndex);
        result.add(MetalRTStageRequirement::InstanceID);
    }
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

VisibleAdapter createVisibleAdapter(
    IRModule* module,
    const SchemaStage& stage,
    const StageRequirements& tableRequirements,
    UnownedStringSlice symbolName,
    IRType* descriptorPointerType)
{
    SLANG_RELEASE_ASSERT(stage.stage == Stage::ClosestHit || stage.stage == Stage::Miss);
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
    Dictionary<ParameterKind, IRInst*> parameters;
    Dictionary<IROp, IRInst*> values;
    for (auto parameter : getVisibleParameters(builder, tableRequirements, descriptorPointerType))
    {
        auto value = builder.emitParam(parameter.type);
        builder.addNameHintDecoration(value, UnownedTerminatedStringSlice(parameter.name));
        if (parameter.kind == ParameterKind::GlobalResources)
            builder.addDecoration(value, kIROp_MetalRTGlobalResourceContextDecoration);
        parameters.add(parameter.kind, value);
        if (parameter.operation != kIROp_Invalid)
            values.add(parameter.operation, value);
    }
    if (!stage.isPresent)
    {
        // A placeholder occupies a dense VFT position but has no source body or state access.
        builder.emitReturn();
        return result;
    }
    SLANG_RELEASE_ASSERT(stage.invoke && stage.input);
    if (auto descriptor = parameters.tryGetValue(ParameterKind::Descriptor))
        result.descriptorResources = *descriptor;
    result.records = parameters.getValue(ParameterKind::Records);

    HashSet<IRFunc*> visited;
    List<IRInst*> operations;
    collectOperations(stage.invoke, visited, operations);
    HashSet<KeyValuePair<IROp, IRType*>> seen;
    for (auto operation : operations)
    {
        auto op = operation->getOp();
        auto type = operation->getDataType();
        IRInst* value = nullptr;
        if (op == kIROp_PortableRTGetPayload)
            value = getStageDataReference(
                builder,
                parameters.getValue(ParameterKind::Data),
                cast<IRPtrTypeBase>(type)->getValueType());
        else if (auto bound = values.tryGetValue(op))
            value = *bound;
        else if (op == kIROp_PortableRTGetRecord)
        {
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
            IRInst* fields[] = {
                parameters.getValue(ParameterKind::ObjectOrigin),
                values.getValue(kIROp_MetalRTGetMinDistance),
                parameters.getValue(ParameterKind::ObjectDirection),
                values.getValue(kIROp_MetalRTGetCurrentDistance)};
            value = builder.emitMakeStruct(type, SLANG_COUNT_OF(fields), fields);
        }
        if (value && seen.add({op, type}))
            result.bindings.add({op, type, value});
    }
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
    for (auto parameter : getCallableParameters(builder, tableRequirements, descriptorPointerType))
    {
        auto value = builder.emitParam(parameter.type);
        builder.addNameHintDecoration(value, UnownedTerminatedStringSlice(parameter.name));
        if (parameter.kind == ParameterKind::GlobalResources)
            builder.addDecoration(value, kIROp_MetalRTGlobalResourceContextDecoration);
        parameters.add(parameter.kind, value);
        if (parameter.operation != kIROp_Invalid)
            values.add(parameter.operation, value);
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
    HashSet<KeyValuePair<IROp, IRType*>> seen;
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
                parameter.lower(operation);
                continue;
            }
            auto value = parameter.get(getParentFunc(operation));
            if (value->getDataType() == operation->getDataType())
            {
                // Only the ordinary two-level getter uses the scalar ABI, and it supplies level 0.
                SLANG_RELEASE_ASSERT(cast<IRIntLit>(operation->getOperand(0))->getValue() == 0);
                operation->replaceUsesWith(value);
            }
            else
            {
                // MultiLevelAccelerationStructure<1> has an empty path and works on targets
                // predating Metal's array_ref type. All multilevel VFTs therefore pass plain
                // thread pointers plus a count, while native traversal owns their storage.
                IRBuilder builder(module);
                SLANG_RELEASE_ASSERT(as<IRPtrType>(value->getDataType()));
                builder.setInsertBefore(operation);
                operation->replaceUsesWith(
                    builder.emitLoad(builder.emitGetOffsetPtr(value, operation->getOperand(0))));
            }
            operation->removeAndDeallocate();
        }
    }
}

} // namespace Metal
} // namespace PortableRT
} // namespace Slang
