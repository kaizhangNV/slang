#include "slang-ir-metal-portable-ray-tracing-stages.h"

#include "slang-ir-metal-portable-ray-tracing-instance-path.h"
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

IRInst* getRayPayloadReference(
    IRBuilder& builder,
    const RayData& carrier,
    IRInst* rayData,
    IRType* payloadType)
{
    if (carrier.payloadKey)
        return builder.emitFieldAddress(rayData, carrier.payloadKey);

    // Consider a payload-less trace that reaches a stage with `Payload = ApplicationEmpty`.
    // Both transport zero bytes, but ordinary helper calls still use ApplicationEmpty values.
    // Keep that nominal reference until normal empty-type legalization removes it; casting the
    // carrier to a different source empty type would incorrectly imply physical payload storage.
    SLANG_RELEASE_ASSERT(isEmptyType(payloadType));
    if (as<IRVoidType>(payloadType))
    {
        // Unlike a pointer to an empty struct, void* is an opaque pointer and remains legal
        // after empty-type legalization. It carries no payload address: allocating/storing a
        // void local would create invalid IR that the emitter cannot turn into native storage.
        return builder.getNullPtrValue(builder.getPtrType(payloadType, AddressSpace::ThreadLocal));
    }
    auto value = builder.emitVar(payloadType);
    builder.emitStore(value, builder.emitDefaultConstruct(payloadType));
    return value;
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
    RayData,
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
    DispatchIndex,
    DispatchDimensions,
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
    IRType* descriptorPointerType,
    IRType* instancePathParamType)
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
        {ParameterKind::RayData,
         builder.getPtrType(builder.getVoidType(), AddressSpace::ThreadLocal),
         "rayData"});
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
             instancePathParamType,
             "instanceIndices",
             kIROp_MetalRTGetInstanceIndex});
    if (requirements.has(MetalRTStageRequirement::InstanceIDAtLevel))
        result.add(
            {ParameterKind::InstanceIDs,
             instancePathParamType,
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
    List<ParameterInfo> result;
    result.add(
        {ParameterKind::RayData,
         builder.getPtrType(builder.getVoidType(), AddressSpace::ThreadLocal),
         "data"});
    auto uint3 = builder.getVectorType(builder.getUIntType(), 3);
    if (requirements.dispatchRaysIndex)
        result.add(
            {ParameterKind::DispatchIndex,
             uint3,
             "dispatchRaysIndex",
             kIROp_MetalRTGetDispatchRaysIndex});
    if (requirements.dispatchRaysDimensions)
        result.add(
            {ParameterKind::DispatchDimensions,
             uint3,
             "dispatchRaysDimensions",
             kIROp_MetalRTGetDispatchRaysDimensions});
    if (requirements.has(MetalRTStageRequirement::Record))
        result.add(
            {ParameterKind::Record,
             builder.getPtrType(builder.getUInt8Type(), AddressSpace::Global),
             "recordData"});
    result.add({ParameterKind::Descriptor, descriptorPointerType, "descriptorResources"});
    result.add(
        {ParameterKind::Records,
         builder.getPtrType(builder.getUIntType(), AddressSpace::Global),
         "records"});
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

/// Loads a field whose presence was decided when the descriptor created its canonical carrier.
void bindCarrierField(
    IRBuilder& builder,
    Dictionary<IROp, IRInst*>& values,
    IRInst* carrier,
    IRStructKey* key,
    IROp operation)
{
    if (key)
        values.add(operation, builder.emitLoad(builder.emitFieldAddress(carrier, key)));
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

StageRequirements collectStageRequirements(IRFunc* invoke, bool multiLevel)
{
    StageRequirements result;
    if (!invoke)
        return result;
    HashSet<IRFunc*> visited;
    List<IRInst*> operations;
    collectOperations(invoke, visited, operations);
    for (auto operation : operations)
    {
        switch (operation->getOp())
        {
        case kIROp_PortableRTGetRecord:
            if (!isEmptyType(operation->getDataType()))
                result.add(MetalRTStageRequirement::Record);
            break;
        case kIROp_PortableRTGetHitAttributes:
            if (!isBuiltinAttributeView(operation))
                result.add(MetalRTStageRequirement::HitAttributes);
            break;
        case kIROp_PortableRTGetTriangleBarycentricCoord:
            result.add(MetalRTStageRequirement::TriangleBarycentricCoord);
            break;
        case kIROp_PortableRTGetCurveParameter:
            result.add(MetalRTStageRequirement::CurveParameter);
            break;
        case kIROp_MetalRTGetTriangleFrontFacing:
            result.add(MetalRTStageRequirement::TriangleFrontFacing);
            break;
        case kIROp_MetalRTGetMinDistance:
            result.minDistance = true;
            break;
        case kIROp_MetalRTGetRayTime:
            result.rayTime = true;
            break;
        case kIROp_MetalRTGetRayFlags:
            result.rayFlags = true;
            break;
        case kIROp_MetalRTGetDispatchRaysIndex:
            result.dispatchRaysIndex = true;
            break;
        case kIROp_MetalRTGetDispatchRaysDimensions:
            result.dispatchRaysDimensions = true;
            break;
        case kIROp_MetalRTGetCurrentDistance:
            result.add(MetalRTStageRequirement::Distance);
            break;
        case kIROp_MetalRTGetHitKind:
            result.add(MetalRTStageRequirement::HitKind);
            break;
        case kIROp_MetalRTGetWorldSpaceOrigin:
            result.add(MetalRTStageRequirement::WorldSpaceOrigin);
            break;
        case kIROp_MetalRTGetWorldSpaceDirection:
            result.add(MetalRTStageRequirement::WorldSpaceDirection);
            break;
        case kIROp_MetalRTGetPrimitiveIndex:
            result.add(MetalRTStageRequirement::PrimitiveIndex);
            break;
        case kIROp_MetalRTGetGeometryIndex:
            result.add(MetalRTStageRequirement::GeometryIndex);
            break;
        case kIROp_MetalRTGetInstanceCount:
            result.add(MetalRTStageRequirement::InstanceCount);
            break;
        case kIROp_MetalRTGetInstanceIndex:
            result.add(
                multiLevel ? MetalRTStageRequirement::InstanceIndexAtLevel
                           : MetalRTStageRequirement::InstanceIndex);
            break;
        case kIROp_MetalRTGetInstanceID:
            result.add(
                multiLevel ? MetalRTStageRequirement::InstanceIDAtLevel
                           : MetalRTStageRequirement::InstanceID);
            break;
        case kIROp_MetalRTGetObjectSpaceRay:
            result.add(MetalRTStageRequirement::ObjectSpaceRay);
            result.minDistance = true;
            break;
        case kIROp_MetalRTGetObjectToWorld:
            result.add(MetalRTStageRequirement::ObjectToWorld);
            break;
        case kIROp_MetalRTGetWorldToObject:
            result.add(MetalRTStageRequirement::WorldToObject);
            break;
        case kIROp_MetalRTCallShader:
            result.add(MetalRTStageRequirement::CallableDispatch);
            break;
        default:
            break;
        }
    }
    return result;
}

IRFuncType* getVisibleFunctionSignature(
    IRBuilder& builder,
    const StageRequirements& requirements,
    IRType* descriptorPointerType,
    IRType* instancePathParamType)
{
    List<IRType*> types;
    for (auto parameter :
         getVisibleParameters(builder, requirements, descriptorPointerType, instancePathParamType))
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
    const RayData& carrier,
    UnownedStringSlice symbolName,
    IRType* descriptorPointerType,
    IRType* instancePathParamType)
{
    SLANG_RELEASE_ASSERT(stage.stage == Stage::ClosestHit || stage.stage == Stage::Miss);
    IRBuilder builder(module);
    builder.setInsertInto(module->getModuleInst());
    VisibleAdapter result;
    result.function = builder.createFunc();
    auto functionType = getVisibleFunctionSignature(
        builder,
        tableRequirements,
        descriptorPointerType,
        instancePathParamType);
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
    for (auto parameter : getVisibleParameters(
             builder,
             tableRequirements,
             descriptorPointerType,
             instancePathParamType))
    {
        auto value = builder.emitParam(parameter.type);
        builder.addNameHintDecoration(value, UnownedTerminatedStringSlice(parameter.name));
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
    auto rayData = builder.emitBitCast(
        builder.getPtrType(carrier.type, AddressSpace::ThreadLocal),
        parameters.getValue(ParameterKind::RayData));
    bindCarrierField(builder, values, rayData, carrier.minDistanceKey, kIROp_MetalRTGetMinDistance);
    bindCarrierField(builder, values, rayData, carrier.rayTimeKey, kIROp_MetalRTGetRayTime);
    bindCarrierField(builder, values, rayData, carrier.rayFlagsKey, kIROp_MetalRTGetRayFlags);
    bindCarrierField(
        builder,
        values,
        rayData,
        carrier.dispatchRaysIndexKey,
        kIROp_MetalRTGetDispatchRaysIndex);
    bindCarrierField(
        builder,
        values,
        rayData,
        carrier.dispatchRaysDimensionsKey,
        kIROp_MetalRTGetDispatchRaysDimensions);
    if (auto descriptor = parameters.tryGetValue(ParameterKind::Descriptor))
        result.descriptorResources = *descriptor;
    if (carrier.header && carrier.headerKey && carrier.header->recordsKey)
    {
        auto header = builder.emitFieldAddress(rayData, carrier.headerKey);
        result.records =
            builder.emitLoad(builder.emitFieldAddress(header, carrier.header->recordsKey));
    }

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
            value = getRayPayloadReference(
                builder,
                carrier,
                rayData,
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
                auto key = carrier.customAttributeKeys.getValue(type);
                value = builder.emitLoad(builder.emitFieldAddress(rayData, key));
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
    if (getBuiltinRole(stage.input->getPrimitiveType()) ==
            KnownBuiltinDeclName::RayTracingBoundingBoxPrimitive &&
        carrier.customHitKindKey)
    {
        auto hitKind =
            builder.emitLoad(builder.emitFieldAddress(rayData, carrier.customHitKindKey));
        for (auto& binding : result.bindings)
            if (binding.op == kIROp_MetalRTGetHitKind)
                binding.value = hitKind;
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
        parameters.add(parameter.kind, value);
        if (parameter.operation != kIROp_Invalid)
            values.add(parameter.operation, value);
    }
    result.descriptorResources = parameters.getValue(ParameterKind::Descriptor);
    result.records = parameters.getValue(ParameterKind::Records);
    auto dataType = stage.input->getPayloadType();
    IRInst* data;
    if (as<IRVoidType>(dataType))
    {
        // A kept generic helper may still take the nominal void reference. It has no storage,
        // so use a typed null pointer rather than producing an invalid void local/store.
        data = builder.getNullPtrValue(builder.getPtrType(dataType, AddressSpace::ThreadLocal));
    }
    else if (isEmptyType(dataType))
    {
        // The source view has no bits. Preserve its nominal reference for ordinary helper calls;
        // empty-type legalization removes that reference without dereferencing the erased ABI.
        data = builder.emitVar(dataType);
        builder.emitStore(data, builder.emitDefaultConstruct(dataType));
    }
    else
        data = builder.emitBitCast(
            builder.getPtrType(dataType, AddressSpace::ThreadLocal),
            parameters.getValue(ParameterKind::RayData));
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
        IRFunc* pathElement = nullptr;
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
                IRBuilder builder(module);
                if (!pathElement)
                {
                    SLANG_RELEASE_ASSERT(
                        value->getDataType()->getOp() == kIROp_MetalRTInstancePath);
                    pathElement =
                        createInstancePathOperation(module, InstancePathOperation::Element);
                }
                builder.setInsertBefore(operation);
                IRInst* arguments[] = {value, operation->getOperand(0)};
                operation->replaceUsesWith(
                    builder.emitCallInst(operation->getDataType(), pathElement, 2, arguments));
            }
            operation->removeAndDeallocate();
        }
    }
}

} // namespace Metal
} // namespace PortableRT
} // namespace Slang
