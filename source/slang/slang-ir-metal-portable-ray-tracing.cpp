#include "slang-ir-metal-portable-ray-tracing.h"

#include "slang-ir-call-graph.h"
#include "slang-ir-layout.h"
#include "slang-ir-metal-portable-ray-tracing-acceleration.h"
#include "slang-ir-metal-portable-ray-tracing-candidates.h"
#include "slang-ir-metal-portable-ray-tracing-instance-path.h"
#include "slang-ir-metal-portable-ray-tracing-metadata.h"
#include "slang-ir-metal-portable-ray-tracing-resources.h"
#include "slang-ir-metal-portable-ray-tracing-stages.h"
#include "slang-ir-metal-portable-ray-tracing-standalone.h"
#include "slang-ir-portable-ray-tracing-schema.h"
#include "slang-ir-portable-ray-tracing-util.h"
#include "slang-ir-util.h"
#include "slang-rich-diagnostics.h"
#include "slang-target-program.h"

namespace Slang
{
namespace
{
using namespace PortableRT;
using namespace PortableRT::Metal;

/// Keeps target-local products together while the canonical schema still owns source identity.
/// A descriptor has one physical set of tables, even when different trace calls use different
/// payload types. Each payload type gets its own carrier; all carriers start with the same header.
struct Program : RefObject
{
    IRTraceProgramDescriptorType* sourceType = nullptr;
    Index index = 0;
    Schema schema;
    MetalRTDescriptorResources resources;
    UInt tags = UInt(MetalRTTag::Instancing);
    IRIntegerValue maxLevels = 0;
    MetalRTGeometryKind geometry = MetalRTGeometryKind::Unknown;
    StageRequirements missRequirements;
    StageRequirements hitRequirements;
    StageRequirements candidateRequirements;
    StageRequirements callableRequirements;
    StageRequirements allRequirements;
    RayDataHeader header;
    Dictionary<IRType*, RayData> carriers;
    List<VisibleAdapter> adapters;
    CandidateAdapters candidates;
    IRIntegerValue hitStride = kMetalRTRecordHeaderSize;
    IRIntegerValue missStride = kMetalRTRecordHeaderSize;
    IRIntegerValue callableStride = kMetalRTRecordHeaderSize;
    IRFunc* instanceContribution = nullptr;
    IRType* instancePathType = nullptr;
    bool hasCandidates = false;
};

/// Creates a field once and returns its canonical key for all subsequent loads and stores.
IRStructKey* addField(IRBuilder& builder, IRStructType* type, IRType* fieldType, const char* name)
{
    auto key = builder.createStructKey();
    builder.addNameHintDecoration(key, UnownedTerminatedStringSlice(name));
    builder.createStructField(type, key, fieldType);
    return key;
}

/// Gives every payload carrier the same prefix so candidate dispatch can find the host's record
/// before it knows which typed stage to call. No pointer to the caller's payload crosses traversal.
RayDataHeader createHeader(IRModule* module)
{
    IRBuilder builder(module);
    builder.setInsertInto(module->getModuleInst());
    RayDataHeader result;
    result.type = builder.createStructType();
    builder.addNameHintDecoration(result.type, toSlice("PortableRTRayDataHeader"));
    result.recordsKey = addField(
        builder,
        result.type,
        builder.getPtrType(builder.getUIntType(), AddressSpace::Global),
        "records");
    result.sbtOffsetKey = addField(builder, result.type, builder.getUIntType(), "sbtOffset");
    result.sbtStrideKey = addField(builder, result.type, builder.getUIntType(), "sbtStride");
    return result;
}

/// Constructs invocation storage from reachable requirements, rather than making every shader
/// carry every optional builtin. The exact payload IR type is the cache key: a trace and its
/// compatible stage therefore use one carrier without reconstructing or comparing source syntax.
/// Empty source payloads share canonical void as their storage key. In particular, the library's
/// payload-less overload and an application's empty payload need the same procedural attributes.
RayData& getCarrier(IRModule* module, Program& program, IRType* payloadType)
{
    IRBuilder builder(module);
    payloadType = getRayPayloadStorageType(builder, payloadType);
    if (auto found = program.carriers.tryGetValue(payloadType))
        return *found;
    builder.setInsertInto(module->getModuleInst());
    RayData result;
    result.type = builder.createStructType();
    builder.addNameHintDecoration(result.type, toSlice("PortableRTRayData"));
    result.header = &program.header;
    result.headerKey = addField(builder, result.type, program.header.type, "header");
    if (!as<IRVoidType>(payloadType))
        result.payloadKey = addField(builder, result.type, payloadType, "payload");
    auto& requirements = program.allRequirements;
    if (requirements.minDistance || requirements.has(MetalRTStageRequirement::ObjectSpaceRay))
        result.minDistanceKey =
            addField(builder, result.type, builder.getFloatType(), "minDistance");
    if (requirements.rayTime)
        result.rayTimeKey = addField(builder, result.type, builder.getFloatType(), "time");
    if (requirements.rayFlags)
        result.rayFlagsKey = addField(builder, result.type, builder.getUIntType(), "rayFlags");
    auto uint3 = builder.getVectorType(builder.getUIntType(), 3);
    if (requirements.dispatchRaysIndex)
        result.dispatchRaysIndexKey = addField(builder, result.type, uint3, "dispatchIndex");
    if (requirements.dispatchRaysDimensions)
        result.dispatchRaysDimensionsKey =
            addField(builder, result.type, uint3, "dispatchDimensions");
    for (auto& group : program.schema.hitGroups)
    {
        if (getRayPayloadStorageType(builder, group.payloadType) != payloadType ||
            getBuiltinRole(group.primitiveType) !=
                KnownBuiltinDeclName::RayTracingBoundingBoxPrimitive)
            continue;
        if (!result.customAttributeKeys.containsKey(group.attributesType))
            result.customAttributeKeys.add(
                group.attributesType,
                addField(builder, result.type, group.attributesType, "customAttributes"));
        if (!result.customHitKindKey)
            result.customHitKindKey =
                addField(builder, result.type, builder.getUIntType(), "customHitKind");
    }
    program.carriers.add(payloadType, result);
    return *program.carriers.tryGetValue(payloadType);
}

/// Maps the closed source primitive family to the native geometry annotation. Attributes are
/// deliberately not used for this classification: a custom primitive may use any attribute type.
MetalRTGeometryKind getGeometry(IRType* primitive)
{
    switch (getBuiltinRole(primitive))
    {
    case KnownBuiltinDeclName::RayTracingTrianglePrimitive:
        return MetalRTGeometryKind::Triangle;
    case KnownBuiltinDeclName::RayTracingCurvePrimitive:
        return MetalRTGeometryKind::Curve;
    case KnownBuiltinDeclName::RayTracingBoundingBoxPrimitive:
        return MetalRTGeometryKind::BoundingBox;
    default:
        SLANG_UNEXPECTED("expected a checked portable ray-tracing primitive");
    }
}

/// Computes the host stride using the packed C layout that Metal structured-buffer reflection
/// exposes. Natural IR layout is different: its bool occupies four bytes and its nested structs
/// need not have C tail padding. Generated record pointers use ordinary buffer storage lowering
/// so the emitted packed vectors and MSL scalar types agree with this layout.
bool getRecordStride(
    TargetRequest* target,
    IRType* type,
    IRInst* useSite,
    DiagnosticSink* sink,
    IRIntegerValue& outStride)
{
    if (as<IRVoidType>(type))
    {
        outStride = kMetalRTRecordHeaderSize;
        return true;
    }
    IRSizeAndAlignment layout;
    if (SLANG_FAILED(getSizeAndAlignment(target, IRTypeLayoutRules::getC(), type, &layout)) ||
        layout.size == IRSizeAndAlignment::kIndeterminateSize || layout.size < 0 ||
        getMetalRTRecordStride(UInt64(layout.size)) > UINT32_MAX ||
        layout.alignment > int(kMetalRTRecordAlignment))
    {
        sink->diagnose(Diagnostics::MetalRayTracingRecordLayout{.location = useSite->sourceLoc});
        return false;
    }
    outStride = IRIntegerValue(getMetalRTRecordStride(UInt64(layout.size)));
    return true;
}

/// Collects the schema's own requirements before indirect dispatch propagates requirements from
/// its callees. Resource signatures and adapters are created only after that propagation finishes.
RefPtr<Program> createProgram(
    IRModule* module,
    TargetRequest* target,
    IRTraceProgramDescriptorType* type,
    Index programIndex,
    IRInst* useSite,
    DiagnosticSink* sink)
{
    RefPtr<Program> result = new Program();
    auto& program = *result;
    program.sourceType = type;
    program.index = programIndex;
    SLANG_RELEASE_ASSERT(readSchema(type, program.schema));
    TraceContextRequirements contextRequirements;
    if (!getTraceContextRequirements(program.schema, target, useSite, sink, contextRequirements))
        return nullptr;
    program.tags = contextRequirements.tags;
    program.maxLevels = contextRequirements.maxLevels;
    program.header = createHeader(module);
    auto accelerationStructure =
        cast<IRRaytracingAccelerationStructureType>(program.schema.accelerationStructure);
    bool hasInstancePath = accelerationStructure->getOperandCount() != 0;
    bool firstGeometry = true;
    for (auto& group : program.schema.hitGroups)
    {
        auto geometry = getGeometry(group.primitiveType);
        program.geometry =
            firstGeometry
                ? geometry
                : (program.geometry == geometry ? geometry : MetalRTGeometryKind::Unknown);
        firstGeometry = false;
        auto hitRequirements = collectStageRequirements(group.closestHit.invoke, hasInstancePath);
        program.hitRequirements.combine(hitRequirements);
        auto candidateRequirements = collectStageRequirements(group.anyHit.invoke, hasInstancePath);
        candidateRequirements.combine(
            collectStageRequirements(group.intersection.invoke, hasInstancePath));
        program.candidateRequirements.combine(candidateRequirements);
        program.hasCandidates |= group.anyHit.isPresent || group.intersection.isPresent;
        // A triangle's hit kind is derived from its front-facing state. Procedural ReportHit
        // also needs the trace flags even when the source never explicitly reads rayFlags.
        if (geometry == MetalRTGeometryKind::Triangle &&
            (candidateRequirements.has(MetalRTStageRequirement::HitKind) ||
             hitRequirements.has(MetalRTStageRequirement::HitKind)))
            program.tags |= UInt(MetalRTTag::TriangleData);
        if (geometry == MetalRTGeometryKind::BoundingBox && group.intersection.isPresent)
            program.candidateRequirements.rayFlags = true;
        IRIntegerValue stride;
        if (!getRecordStride(target, group.recordType, useSite, sink, stride))
            return nullptr;
        program.hitStride = Math::Max(program.hitStride, stride);
    }
    for (auto& miss : program.schema.missShaders)
    {
        program.missRequirements.combine(collectStageRequirements(miss.invoke));
        IRIntegerValue stride;
        if (!getRecordStride(target, miss.input->getRecordType(), useSite, sink, stride))
            return nullptr;
        program.missStride = Math::Max(program.missStride, stride);
    }
    for (auto& callable : program.schema.callableShaders)
    {
        program.callableRequirements.combine(collectStageRequirements(callable.invoke));
        IRIntegerValue stride;
        if (!getRecordStride(target, callable.input->getRecordType(), useSite, sink, stride))
            return nullptr;
        program.callableStride = Math::Max(program.callableStride, stride);
    }
    program.allRequirements.combine(program.hitRequirements);
    program.allRequirements.combine(program.missRequirements);
    program.allRequirements.combine(program.candidateRequirements);
    // Candidate signatures are shared by primitive kind. A procedural group's hitKind query
    // can therefore add front_facing to a triangle callback even if that triangle stage does
    // not query hitKind itself. Infer tags from the generated signature's requirements too.
    if (program.candidateRequirements.has(MetalRTStageRequirement::HitKind))
        for (const auto& group : program.schema.hitGroups)
            if (getGeometry(group.primitiveType) == MetalRTGeometryKind::Triangle)
                program.tags |= UInt(MetalRTTag::TriangleData);
    if (program.allRequirements.has(MetalRTStageRequirement::TriangleBarycentricCoord) ||
        program.allRequirements.has(MetalRTStageRequirement::TriangleFrontFacing))
        program.tags |= UInt(MetalRTTag::TriangleData);
    if (program.allRequirements.has(MetalRTStageRequirement::CurveParameter))
        program.tags |= UInt(MetalRTTag::CurveData);
    if ((program.tags & UInt(MetalRTTag::Instancing)) &&
        (program.allRequirements.has(MetalRTStageRequirement::WorldSpaceOrigin) ||
         program.allRequirements.has(MetalRTStageRequirement::WorldSpaceDirection) ||
         program.allRequirements.has(MetalRTStageRequirement::ObjectToWorld) ||
         program.allRequirements.has(MetalRTStageRequirement::WorldToObject) ||
         program.hitRequirements.has(MetalRTStageRequirement::ObjectSpaceRay)))
        program.tags |= UInt(MetalRTTag::WorldSpaceData);

    IRBuilder builder(module);
    program.instancePathType =
        hasInstancePath ? builder.getType(kIROp_MetalRTInstancePath) : builder.getUIntType();
    return result;
}

/// Freezes physical resource signatures only after nested dispatch has propagated launch needs.
void createProgramResources(
    IRModule* module,
    Program& program,
    Dictionary<IRType*, IRFunc*>& instanceContributions)
{
    IRBuilder builder(module);
    auto descriptorPointer = builder.getPtrType(builder.getUIntType(), AddressSpace::Uniform);
    auto instancePathType = program.instancePathType;
    program.resources = createMetalRTDescriptorResources(
        module,
        program.sourceType,
        program.tags,
        program.maxLevels,
        getVisibleFunctionSignature(
            builder,
            program.missRequirements,
            descriptorPointer,
            instancePathType),
        getVisibleFunctionSignature(
            builder,
            program.hitRequirements,
            descriptorPointer,
            instancePathType),
        getCallableFunctionSignature(builder, program.callableRequirements, descriptorPointer));
    addMetalRTProgramMetadata(
        module,
        program.resources.resourcesStruct,
        program.tags,
        program.maxLevels,
        program.hitStride,
        program.missStride,
        program.callableStride);
    if (program.tags & UInt(MetalRTTag::Instancing))
    {
        if (!instanceContributions.containsKey(instancePathType))
            instanceContributions.add(
                instancePathType,
                createInstanceContribution(module, instancePathType));
        program.instanceContribution = instanceContributions.getValue(instancePathType);
    }
}

/// Creates every source-stage root after trace/call lowering has inserted its launch queries.
/// Binding sooner would miss queries introduced by nested dispatch inside an ordinary helper.
void createProgramAdapters(IRModule* module, Program& program, DiagnosticSink* sink)
{
    IRBuilder builder(module);
    auto descriptorPointer = builder.getPtrType(builder.getUIntType(), AddressSpace::Uniform);
    auto instancePathType = program.instancePathType;

    // Candidate callbacks can be needed even when ClosestHit is absent. Create their exact
    // payload carriers before either family of adapters starts threading invocation values.
    for (auto& group : program.schema.hitGroups)
        getCarrier(module, program, group.payloadType);
    StringBuilder candidatePrefix;
    candidatePrefix << "__slang_portable_rt_" << program.index << "_candidate";
    program.candidates = createCandidateAdapters(
        module,
        program.schema,
        program.carriers,
        program.header,
        program.candidateRequirements,
        program.tags,
        program.maxLevels,
        program.hitStride,
        program.instanceContribution,
        instancePathType,
        candidatePrefix.getUnownedSlice(),
        sink);
    for (auto& pair : program.candidates.intersectionFunctions)
        addMetalRTFunctionMetadata(
            module,
            pair.second,
            program.resources.resourcesStruct,
            UInt(pair.first),
            program.candidateRequirements.mask);

    auto createAdapter = [&](const SchemaStage& stage,
                             IRType* payload,
                             const StageRequirements& requirements,
                             MetalRTDescriptorResource tableRole,
                             const char* stageName,
                             Index index)
    {
        StringBuilder symbol;
        symbol << "__slang_portable_rt_" << program.index << "_" << stageName << "_" << index;
        auto adapter = createVisibleAdapter(
            module,
            stage,
            requirements,
            getCarrier(module, program, payload),
            symbol.getUnownedSlice(),
            descriptorPointer,
            instancePathType);
        SLANG_RELEASE_ASSERT(
            adapter.function->findDecoration<IRMetalVisibleFunctionDecoration>()->getTableType() ==
            program.resources.getField(tableRole)->getFieldType());
        program.adapters.add(adapter);
        addMetalRTFunctionMetadata(
            module,
            adapter.function,
            program.resources.resourcesStruct,
            UInt(index),
            requirements.mask);
    };
    for (Index i = 0; i < program.schema.hitGroups.getCount(); ++i)
    {
        auto& group = program.schema.hitGroups[i];
        createAdapter(
            group.closestHit,
            group.payloadType,
            program.hitRequirements,
            MetalRTDescriptorResource::ClosestHitVisibleFunctionTable,
            "closesthit",
            i);
    }
    for (Index i = 0; i < program.schema.missShaders.getCount(); ++i)
    {
        auto& miss = program.schema.missShaders[i];
        createAdapter(
            miss,
            miss.input->getPayloadType(),
            program.missRequirements,
            MetalRTDescriptorResource::MissVisibleFunctionTable,
            "miss",
            i);
    }
    for (Index i = 0; i < program.schema.callableShaders.getCount(); ++i)
    {
        StringBuilder symbol;
        symbol << "__slang_portable_rt_" << program.index << "_callable_" << i;
        auto adapter = createCallableAdapter(
            module,
            program.schema.callableShaders[i],
            program.callableRequirements,
            symbol.getUnownedSlice(),
            descriptorPointer);
        SLANG_RELEASE_ASSERT(
            adapter.function->findDecoration<IRMetalVisibleFunctionDecoration>()->getTableType() ==
            program.resources.getField(MetalRTDescriptorResource::CallableVisibleFunctionTable)
                ->getFieldType());
        program.adapters.add(adapter);
        addMetalRTFunctionMetadata(
            module,
            adapter.function,
            program.resources.resourcesStruct,
            UInt(i),
            program.callableRequirements.mask);
    }
}

/// Loads a named resource from the descriptor using the physical field's exact type. The
/// descriptor is still opaque at this point; the following physicalization step replaces that
/// type everywhere, including helper signatures and nested global resources.
IRInst* loadResource(IRBuilder& builder, IRInst* descriptor, IRStructField* field)
{
    auto addressType = builder.getPtrType(field->getFieldType(), AddressSpace::Uniform);
    return builder.emitLoad(builder.emitFieldAddress(addressType, descriptor, field->getKey()));
}

/// Expands the trace boundary, leaving traversal and the post-trace choice explicit for emission.
/// The payload is copied into ray_data before traversal and back afterward; stages all reference
/// that invocation's carrier rather than independent copies of the logical input object.
void lowerTrace(IRModule* module, IRMetalTraceRay* trace, Program& program)
{
    auto payloadType = cast<IRPtrTypeBase>(trace->getPayload()->getDataType())->getValueType();
    auto& carrier = getCarrier(module, program, payloadType);
    IRBuilder builder(module);
    builder.setInsertBefore(trace);
    auto load = [&](MetalRTDescriptorResource role)
    { return loadResource(builder, trace->getDescriptor(), program.resources.getField(role)); };
    auto records = load(MetalRTDescriptorResource::Records);
    auto rayData = builder.emitVar(carrier.type);
    builder.addNameHintDecoration(rayData, toSlice("rayData"));
    IRInst* payload = nullptr;
    if (carrier.payloadKey)
    {
        payload = builder.emitFieldAddress(rayData, carrier.payloadKey);
        builder.emitStore(payload, builder.emitLoad(trace->getPayload()));
    }
    auto header = builder.emitFieldAddress(rayData, carrier.headerKey);
    builder.emitStore(builder.emitFieldAddress(header, carrier.header->recordsKey), records);
    builder.emitStore(
        builder.emitFieldAddress(header, carrier.header->sbtOffsetKey),
        trace->getSbtOffset());
    builder.emitStore(
        builder.emitFieldAddress(header, carrier.header->sbtStrideKey),
        trace->getSbtStride());
    auto time = trace->getOperandCount() == 13 ? trace->getTime()
                                               : builder.getFloatValue(builder.getFloatType(), 0);
    if (carrier.minDistanceKey)
        builder.emitStore(
            builder.emitFieldAddress(rayData, carrier.minDistanceKey),
            trace->getRayTMin());
    if (carrier.rayFlagsKey)
        builder.emitStore(
            builder.emitFieldAddress(rayData, carrier.rayFlagsKey),
            trace->getRayFlags());
    if (carrier.rayTimeKey)
        builder.emitStore(builder.emitFieldAddress(rayData, carrier.rayTimeKey), time);
    auto uint3 = builder.getVectorType(builder.getUIntType(), 3);
    if (carrier.dispatchRaysIndexKey)
    {
        auto index =
            builder.emitIntrinsicInst(uint3, kIROp_MetalRTGetDispatchRaysIndex, 0, nullptr);
        builder.emitStore(builder.emitFieldAddress(rayData, carrier.dispatchRaysIndexKey), index);
    }
    if (carrier.dispatchRaysDimensionsKey)
    {
        auto dimensions =
            builder.emitIntrinsicInst(uint3, kIROp_MetalRTGetDispatchRaysDimensions, 0, nullptr);
        builder.emitStore(
            builder.emitFieldAddress(rayData, carrier.dispatchRaysDimensionsKey),
            dimensions);
    }
    auto integer = [&](IRIntegerValue value)
    { return builder.getIntValue(builder.getIntType(), value); };
    IRInst* operands[] = {
        integer(program.tags),
        integer(program.maxLevels),
        integer(program.missRequirements.mask),
        integer(program.hitRequirements.mask),
        integer(IRIntegerValue(program.geometry)),
        builder.getBoolValue(program.hasCandidates),
        builder.getBoolValue(program.schema.missShaders.getCount() != 0),
        builder.getBoolValue(program.schema.hitGroups.getCount() != 0),
        trace->getRayOrigin(),
        trace->getRayDirection(),
        trace->getRayTMin(),
        trace->getRayTMax(),
        time,
        trace->getRayFlags(),
        trace->getInstanceMask(),
        trace->getSbtOffset(),
        trace->getSbtStride(),
        trace->getMissIndex(),
        trace->getAccelerationStructure(),
        load(MetalRTDescriptorResource::IntersectionFunctionTable),
        load(MetalRTDescriptorResource::MissVisibleFunctionTable),
        load(MetalRTDescriptorResource::ClosestHitVisibleFunctionTable),
        trace->getDescriptor(),
        records,
        builder.getIntValue(builder.getUIntType(), program.hitStride),
        builder.getIntValue(builder.getUIntType(), program.missStride),
        rayData,
        builder.getBoolValue(false),
        builder.getBoolValue(false),
        builder.emitDefaultConstruct(
            builder.getPtrType(builder.getUIntType(), AddressSpace::ThreadLocal)),
        program.instanceContribution ? static_cast<IRInst*>(program.instanceContribution)
                                     : integer(0),
    };
    builder.emitIntrinsicInst(
        builder.getVoidType(),
        kIROp_MetalRTTraceDispatch,
        SLANG_COUNT_OF(operands),
        operands);
    if (payload)
        builder.emitStore(trace->getPayload(), builder.emitLoad(payload));
    trace->removeAndDeallocate();
}

/// Expands callable dispatch using the same host record format as trace dispatch. Empty source
/// data still needs a concrete pointer for Metal's indirect ABI, but has no bits to copy back.
void lowerCallable(IRModule* module, IRMetalRTCallShader* call, Program& program)
{
    IRBuilder builder(module);
    builder.setInsertBefore(call);
    auto descriptor = call->getDescriptor();
    auto records = loadResource(
        builder,
        descriptor,
        program.resources.getField(MetalRTDescriptorResource::Records));
    auto uint3 = builder.getVectorType(builder.getUIntType(), 3);
    auto zero = builder.emitDefaultConstruct(uint3);
    auto& requirements = program.callableRequirements;
    auto index =
        requirements.dispatchRaysIndex
            ? builder.emitIntrinsicInst(uint3, kIROp_MetalRTGetDispatchRaysIndex, 0, nullptr)
            : zero;
    auto dimensions =
        requirements.dispatchRaysDimensions
            ? builder.emitIntrinsicInst(uint3, kIROp_MetalRTGetDispatchRaysDimensions, 0, nullptr)
            : zero;
    auto data = call->getData();
    auto dataType = cast<IRPtrTypeBase>(data->getDataType())->getValueType();
    if (isEmptyType(dataType))
    {
        data = builder.emitVar(builder.getUIntType());
        builder.emitStore(data, builder.getIntValue(builder.getUIntType(), 0));
    }
    IRInst* operands[] = {
        call->getCallableIndex(),
        data,
        index,
        dimensions,
        builder.getBoolValue(requirements.dispatchRaysIndex),
        builder.getBoolValue(requirements.dispatchRaysDimensions),
        descriptor,
        records,
        builder.getIntValue(builder.getUIntType(), program.callableStride),
        builder.getBoolValue(requirements.has(MetalRTStageRequirement::Record)),
        descriptor->getDataType(),
        program.resources.getField(MetalRTDescriptorResource::CallableVisibleFunctionTable)
            ->getKey(),
        builder.getBoolValue(false),
        builder.emitDefaultConstruct(
            builder.getPtrType(builder.getUIntType(), AddressSpace::ThreadLocal))};
    builder.emitIntrinsicInst(
        builder.getVoidType(),
        kIROp_MetalRTCallDispatch,
        SLANG_COUNT_OF(operands),
        operands);
    call->removeAndDeallocate();
}

/// Collects source dispatch instructions after ordinary witness specialization. Both operations
/// retain their canonical descriptor type, including a descriptor used only for callable dispatch.
void collectDispatches(
    IRInst* parent,
    List<IRMetalTraceRay*>& traces,
    List<IRMetalRTCallShader*>& calls)
{
    for (auto child : parent->getChildren())
    {
        if (auto trace = as<IRMetalTraceRay>(child))
            traces.add(trace);
        else if (auto call = as<IRMetalRTCallShader>(child))
            calls.add(call);
        collectDispatches(child, traces, calls);
    }
}

/// Propagates the two launch values across actual indirect-dispatch dependencies. Consider a
/// closest-hit method that calls a callable reading dispatchRaysIndex: the source closest-hit
/// contains no such getter yet, but lowerCallable will create one. Follow that descriptor edge
/// before fixing the caller's carrier/table signature, rather than giving every program all inputs.
void propagateLaunchRequirements(
    Dictionary<IRTraceProgramDescriptorType*, RefPtr<Program>>& programs,
    const List<IRMetalTraceRay*>& traces,
    const List<IRMetalRTCallShader*>& calls)
{
    struct StageUse
    {
        IRFunc* function;
        StageRequirements* requirements;
    };
    List<StageUse> stages;
    List<IRFunc*> roots;
    HashSet<IRFunc*> uniqueRoots;
    auto add = [&](const SchemaStage& stage, StageRequirements* requirements)
    {
        if (!stage.invoke)
            return;
        stages.add({stage.invoke, requirements});
        if (uniqueRoots.add(stage.invoke))
            roots.add(stage.invoke);
    };
    for (auto& pair : programs)
    {
        auto program = pair.second.Ptr();
        for (const auto& group : program->schema.hitGroups)
        {
            add(group.closestHit, &program->hitRequirements);
            add(group.anyHit, &program->candidateRequirements);
            add(group.intersection, &program->candidateRequirements);
        }
        for (const auto& stage : program->schema.missShaders)
            add(stage, &program->missRequirements);
        for (const auto& stage : program->schema.callableShaders)
            add(stage, &program->callableRequirements);
    }
    Dictionary<IRInst*, HashSet<IRFunc*>> references;
    buildEntryPointReferenceGraph(references, roots.getArrayView());
    struct Dependency
    {
        IRFunc* caller;
        Program* target;
        bool callable;
    };
    List<Dependency> dependencies;
    for (auto trace : traces)
        dependencies.add(
            {getParentFunc(trace),
             programs
                 .getValue(
                     cast<IRTraceProgramDescriptorType>(trace->getDescriptor()->getDataType()))
                 .Ptr(),
             false});
    for (auto call : calls)
        dependencies.add(
            {getParentFunc(call),
             programs
                 .getValue(cast<IRTraceProgramDescriptorType>(call->getDescriptor()->getDataType()))
                 .Ptr(),
             true});
    bool changed;
    do
    {
        changed = false;
        for (const auto& dependency : dependencies)
        {
            auto callers = references.tryGetValue(dependency.caller);
            if (!callers)
                continue;
            auto& required = dependency.callable ? dependency.target->callableRequirements
                                                 : dependency.target->allRequirements;
            for (auto& stage : stages)
                if (callers->contains(stage.function))
                {
                    if (required.dispatchRaysIndex && !stage.requirements->dispatchRaysIndex)
                    {
                        stage.requirements->dispatchRaysIndex = true;
                        changed = true;
                    }
                    if (required.dispatchRaysDimensions &&
                        !stage.requirements->dispatchRaysDimensions)
                    {
                        stage.requirements->dispatchRaysDimensions = true;
                        changed = true;
                    }
                }
        }
        for (auto& pair : programs)
        {
            auto& program = *pair.second;
            program.allRequirements.combine(program.hitRequirements);
            program.allRequirements.combine(program.missRequirements);
            program.allRequirements.combine(program.candidateRequirements);
        }
    } while (changed);
}

/// Adds a Metal launch builtin only at a kernel boundary. Intermediate ordinary functions receive
/// the value from their caller through the shared invocation-parameter utility; they need not
/// inline.
class LaunchParameter : public PortableRTInvocationParameter
{
public:
    LaunchParameter(IRModule* module, IRType* type, const char* name, const char* semantic)
        : PortableRTInvocationParameter(module, type, name), m_semantic(semantic)
    {
    }

protected:
    IRInst* createParameter(IRFunc* function) override
    {
        auto parameter = PortableRTInvocationParameter::createParameter(function);
        if (auto entry = function->findDecoration<IREntryPointDecoration>())
        {
            SLANG_RELEASE_ASSERT(entry->getProfile().getStage() == Stage::Compute);
            IRBuilder builder(m_module);
            builder.addTargetSystemValueDecoration(
                parameter,
                UnownedTerminatedStringSlice(m_semantic));
        }
        return parameter;
    }

private:
    const char* m_semantic;
};

/// Collects launch queries left after stage-input lowering. Visible functions already obtain
/// these values from ray_data; the remaining queries originate in a trace's caller.
void collectLaunchQueries(IRInst* parent, List<IRInst*>& indices, List<IRInst*>& dimensions)
{
    for (auto child : parent->getChildren())
    {
        if (child->getOp() == kIROp_MetalRTGetDispatchRaysIndex)
            indices.add(child);
        else if (child->getOp() == kIROp_MetalRTGetDispatchRaysDimensions)
            dimensions.add(child);
        collectLaunchQueries(child, indices, dimensions);
    }
}

void lowerLaunchQueries(IRModule* module)
{
    List<IRInst*> indices;
    List<IRInst*> dimensions;
    collectLaunchQueries(module->getModuleInst(), indices, dimensions);
    IRBuilder builder(module);
    auto type = builder.getVectorType(builder.getUIntType(), 3);
    LaunchParameter indexParameter(module, type, "dispatchIndex", "thread_position_in_grid");
    LaunchParameter dimensionsParameter(module, type, "dispatchDimensions", "threads_per_grid");
    for (auto index : indices)
        indexParameter.lower(index);
    for (auto dimension : dimensions)
        dimensionsParameter.lower(dimension);
}

/// Reports only live operations that require invocation state. Exported helper libraries can
/// retain these operations for later linking by serializing their modules, but native Metal
/// output needs a selected stage or schema to define the arguments that provide that state.
void diagnoseUnboundOperations(IRInst* parent, DiagnosticSink* sink)
{
    for (auto inst : parent->getChildren())
    {
        auto op = inst->getOp();
        if (isStageInputOperation(op) || op == kIROp_MetalRTIgnoreHit ||
            op == kIROp_MetalRTAcceptHitAndEndSearch || op == kIROp_MetalRTReportHit ||
            op == kIROp_MetalRTCallShader || op == kIROp_MetalTraceRay)
        {
            sink->diagnose(Diagnostics::MetalRayTracingUnboundInvocation{
                .operation = UnownedStringSlice(getIROpInfo(op).name),
                .location = inst->sourceLoc});
        }
        diagnoseUnboundOperations(inst, sink);
    }
}

} // namespace

void diagnoseUnboundMetalPortableRayTracing(IRModule* module, DiagnosticSink* sink)
{
    diagnoseUnboundOperations(module->getModuleInst(), sink);
}

void lowerMetalPortableRayTracing(
    IRModule* module,
    TargetProgram* targetProgram,
    List<IRFunc*>& entryPoints,
    DiagnosticSink* sink,
    bool& requiresResultTypeLowering)
{
    List<IRMetalTraceRay*> traces;
    List<IRMetalRTCallShader*> calls;
    collectDispatches(module->getModuleInst(), traces, calls);
    Dictionary<IRInst*, HashSet<IRFunc*>> referencingEntries;
    buildEntryPointReferenceGraph(referencingEntries, module);
    HashSet<IRFunc*> physicalRayGenerationEntries;
    for (auto trace : traces)
    {
        if (auto roots = referencingEntries.tryGetValue(getParentFunc(trace)))
            for (auto root : *roots)
                physicalRayGenerationEntries.add(root);
    }
    for (auto call : calls)
    {
        if (auto roots = referencingEntries.tryGetValue(getParentFunc(call)))
            for (auto root : *roots)
                physicalRayGenerationEntries.add(root);
    }
    IRBuilder builder(module);
    for (auto entry : entryPoints)
    {
        auto decoration = entry->findDecoration<IREntryPointDecoration>();
        if (!decoration || decoration->getProfile().getStage() != Stage::RayGeneration)
            continue;
        if (!physicalRayGenerationEntries.contains(entry))
        {
            sink->diagnose(Diagnostics::UnsupportedTargetIntrinsic{
                .operation = "Metal ray-generation entry without portable ray-tracing dispatch",
                .location = entry->sourceLoc});
            continue;
        }
        // Capability checking admits the logical ray-generation stage. Only now, after witness
        // specialization exposes even an interface-dispatched trace, is its physical stage known.
        decoration->setOperand(
            0,
            builder.getIntValue(builder.getIntType(), Profile(Stage::Compute).raw));
    }
    Dictionary<IRTraceProgramDescriptorType*, RefPtr<Program>> programs;
    Dictionary<IRType*, IRFunc*> instanceContributions;
    auto discover = [&](IRInst* descriptor, IRInst* useSite)
    {
        auto descriptorType = cast<IRTraceProgramDescriptorType>(descriptor->getDataType());
        if (!programs.containsKey(descriptorType))
            programs.add(
                descriptorType,
                createProgram(
                    module,
                    targetProgram->getTargetReq(),
                    descriptorType,
                    programs.getCount(),
                    useSite,
                    sink));
    };
    for (auto trace : traces)
        discover(trace->getDescriptor(), trace);
    for (auto call : calls)
        discover(call->getDescriptor(), call);
    for (auto& pair : programs)
        if (!pair.second)
            return;
    propagateLaunchRequirements(programs, traces, calls);
    List<AccelerationStructureUse> accelerationStructures;
    for (auto trace : traces)
    {
        auto type = cast<IRTraceProgramDescriptorType>(trace->getDescriptor()->getDataType());
        accelerationStructures.add(
            {trace->getAccelerationStructure(), programs.getValue(type)->tags, trace});
    }
    if (!physicalizeAccelerationStructures(module, accelerationStructures, sink))
        return;
    for (auto& pair : programs)
        createProgramResources(module, *pair.second, instanceContributions);
    for (auto trace : traces)
    {
        auto type = cast<IRTraceProgramDescriptorType>(trace->getDescriptor()->getDataType());
        lowerTrace(module, trace, *programs.tryGetValue(type)->Ptr());
    }
    for (auto call : calls)
    {
        auto type = cast<IRTraceProgramDescriptorType>(call->getDescriptor()->getDataType());
        lowerCallable(module, call, *programs.tryGetValue(type)->Ptr());
    }
    auto adapters = prepareMetalStandaloneStages(module, targetProgram, entryPoints, sink);
    List<MetalRTDescriptorResources> resources;
    for (auto& pair : programs)
    {
        auto& program = *pair.second;
        createProgramAdapters(module, program, sink);
        adapters.addRange(program.adapters);
        adapters.addRange(program.candidates.invocationRoots);
        requiresResultTypeLowering |= program.candidates.producesResultTypes;
        resources.add(program.resources);
    }
    lowerVisibleStageInputs(module, adapters);
    lowerLaunchQueries(module);
    physicalizeMetalRTDescriptorResources(module, resources);
}

} // namespace Slang
