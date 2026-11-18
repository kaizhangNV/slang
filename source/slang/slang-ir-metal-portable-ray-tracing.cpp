#include "slang-ir-metal-portable-ray-tracing.h"

#include "slang-ir-any-value-marshalling.h"
#include "slang-ir-layout.h"
#include "slang-ir-metal-portable-ray-tracing-acceleration.h"
#include "slang-ir-metal-portable-ray-tracing-candidates.h"
#include "slang-ir-metal-portable-ray-tracing-instance-path.h"
#include "slang-ir-metal-portable-ray-tracing-resources.h"
#include "slang-ir-metal-portable-ray-tracing-stages.h"
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

/// Stores one trace allocation. Native callbacks know only the header/payload prefix; the
/// committed-attribute offset points into the extra storage sized for this schema's payload.
/// Attributes use the existing AnyValue representation, so differently sized procedural types
/// share one uint-aligned region without depending on native MSL struct packing.
struct RayDataAllocation
{
    RayData prefix;
    IRStructType* type = nullptr;
    IRStructKey* prefixKey = nullptr;
    IRStructKey* attributesKey = nullptr;
};

/// Holds the physical resources and record strides for one fixed descriptor specialization.
/// The host supplies six eight-byte fields: tables, miss/closest-hit/callable VFTs, records,
/// and inline candidateResources. tables[rayType] selects native candidate callbacks. The hit
/// record's first two uints independently select closest-hit and procedural AnyHit VFT slots;
/// neither slot is a schema ordinal. All records retain a 16-byte header before application data.
struct TraceProgramDescriptor : RefObject
{
    IRTraceProgramDescriptorType* sourceType = nullptr;
    Schema schema;
    MetalRTDescriptorResources resources;
    UInt tags = UInt(MetalRTTag::Instancing);
    IRIntegerValue maxLevels = 0;
    MetalRTGeometryKind geometry = MetalRTGeometryKind::Unknown;
    StageRequirements missRequirements;
    StageRequirements hitRequirements;
    StageRequirements callableRequirements;
    const RayDataHeader* header = nullptr;
    Dictionary<IRType*, RayDataAllocation> carriers;
    TargetRequest* target = nullptr;
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

/// Allocates the common prefix and enough packed storage for every procedural attribute type
/// compatible with this payload. The maximum is independent of host-selected VFT slot numbers.
RayDataAllocation& getCarrier(
    IRModule* module,
    TraceProgramDescriptor& programDescriptor,
    IRType* payloadType)
{
    IRBuilder builder(module);
    payloadType = getRayPayloadStorageType(builder, payloadType);
    if (auto found = programDescriptor.carriers.tryGetValue(payloadType))
        return *found;
    RayDataAllocation result;
    result.prefix = createRayDataPrefix(module, *programDescriptor.header, payloadType);
    builder.setInsertInto(module);
    result.type = builder.createStructType();
    builder.addNameHintDecoration(result.type, toSlice("PortableRTRayDataAllocation"));
    result.prefixKey = addField(builder, result.type, result.prefix.type, "prefix");
    SlangInt attributeSize = 0;
    for (const auto& group : programDescriptor.schema.hitGroups)
    {
        if (getRayPayloadStorageType(builder, group.payloadType) != payloadType ||
            getBuiltinRole(group.primitiveType) !=
                KnownBuiltinDeclName::RayTracingBoundingBoxPrimitive)
            continue;
        auto size = getAnyValueSize(group.attributesType, programDescriptor.target);
        SLANG_RELEASE_ASSERT(size >= 0);
        attributeSize = Math::Max(attributeSize, size);
    }
    if (attributeSize)
        result.attributesKey = addField(
            builder,
            result.type,
            builder.getAnyValueType(attributeSize),
            "committedAttributes");
    programDescriptor.carriers.add(payloadType, result);
    return *programDescriptor.carriers.tryGetValue(payloadType);
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

/// Prepares the Metal traversal settings, stage arguments, and record strides for one schema.
/// `type` is a TraceProgramDescriptor<Schema> whose associated types have already been resolved
/// by specialization. createDescriptorResources uses the returned TraceProgramDescriptor to build
/// descriptor resource types. lowerTrace and lowerCallable use it to lower their operations.
/// Deriving these choices from the schema lets ray generation and separately compiled stage
/// implementations agree on their arguments, even when an implementation does not use them.
///
/// Consider a schema whose miss list contains two shaders with these Record types:
///
///     struct SmallRecord { uint value; }
///     struct LargeRecord { uint values[5]; }
///
/// Their application data occupies 4 and 20 bytes. With the 16-byte record header and alignment,
/// their individual strides are 32 and 48 bytes. The whole miss section uses the larger stride,
/// 48 bytes, so a runtime miss-record index always advances by the same amount. The hit and
/// callable sections each get their own maximum stride.
///
/// This function also creates the common ray-data header type. Payload storage, descriptor
/// resources, and traversal callbacks are built later from this information. Shader invoke
/// bodies are compiled when their entries are selected separately. An unsupported traversal
/// setting or Record layout is diagnosed at useSite and returns nullptr.
RefPtr<TraceProgramDescriptor> createTraceProgramDescriptor(
    IRModule* module,
    TargetRequest* target,
    IRTraceProgramDescriptorType* type,
    const RayDataHeader& header,
    IRInst* useSite,
    DiagnosticSink* sink)
{
    RefPtr<TraceProgramDescriptor> result = new TraceProgramDescriptor();
    auto& programDescriptor = *result;
    programDescriptor.sourceType = type;
    programDescriptor.header = &header;
    programDescriptor.target = target;
    // Read the concrete acceleration-structure and motion types, each hit group's Payload,
    // Record, Primitive and Attributes, and the ordered miss/callable Record lists. The hit
    // groups also say whether closest-hit, any-hit and intersection shaders are present.
    readSchemaContract(type, programDescriptor.schema);

    // Translate the declared acceleration-structure topology and motion mode into Metal options
    // such as instancing and primitive_motion, plus a native traversal-depth limit. Check that
    // the target supports these choices.
    TraceContextRequirements contextRequirements;
    if (!getTraceContextRequirements(
            programDescriptor.schema,
            target,
            useSite,
            sink,
            contextRequirements))
        return nullptr;
    programDescriptor.tags = contextRequirements.tags;
    programDescriptor.maxLevels = contextRequirements.maxLevels;
    // Fix the values passed to each visible shader, such as Record, ray origin and launch index.
    // Hit stages always receive an instance count and paths. Ordinary instancing supplies one
    // element, so AccelerationStructure and MultiLevelAccelerationStructure<2> share the same
    // signature. Inspecting shader bodies would make independently compiled signatures disagree.
    programDescriptor.hitRequirements = getStageContractRequirements(Stage::ClosestHit);
    programDescriptor.missRequirements = getStageContractRequirements(Stage::Miss);
    programDescriptor.callableRequirements = getStageContractRequirements(Stage::Callable);
    bool firstGeometry = true;
    for (auto& group : programDescriptor.schema.hitGroups)
    {
        // A single primitive family permits a Metal geometry assumption. Mixed families leave
        // geometry Unknown, so the emitted intersector does not assume triangles or another kind.
        auto geometry = getGeometry(group.primitiveType);
        programDescriptor.geometry =
            firstGeometry ? geometry
                          : (programDescriptor.geometry == geometry ? geometry
                                                                    : MetalRTGeometryKind::Unknown);
        firstGeometry = false;
        // Any-hit or intersection requires candidate callbacks during traversal. The Metal
        // emitter uses this flag to pass the intersection-function table and ray data to intersect.
        programDescriptor.hasCandidates |= group.anyHit.isPresent || group.intersection.isPresent;
        // All runtime hit records use one stride, large enough for every group's Record.
        // getRecordStride includes the header and alignment and rejects unsupported layouts.
        IRIntegerValue stride;
        if (!getRecordStride(target, group.recordType, useSite, sink, stride))
            return nullptr;
        programDescriptor.hitStride = Math::Max(programDescriptor.hitStride, stride);
    }
    // Miss and callable records use separate sections, so compute their maximum strides
    // independently. Each stride starts at the header size, including for void data.
    for (const auto& miss : programDescriptor.schema.missShaders)
    {
        IRIntegerValue stride;
        if (!getRecordStride(target, miss.recordType, useSite, sink, stride))
            return nullptr;
        programDescriptor.missStride = Math::Max(programDescriptor.missStride, stride);
    }
    for (const auto& callable : programDescriptor.schema.callableShaders)
    {
        IRIntegerValue stride;
        if (!getRecordStride(target, callable.recordType, useSite, sink, stride))
            return nullptr;
        programDescriptor.callableStride = Math::Max(programDescriptor.callableStride, stride);
    }
    IRBuilder builder(module);
    // The contribution lookup consumes Metal's native traversal input: a path for deeper
    // traversal, or one scalar instance index for ordinary instancing.
    programDescriptor.instancePathType = programDescriptor.maxLevels > 0
                                             ? builder.getType(kIROp_MetalRTInstancePath)
                                             : builder.getUIntType();
    return result;
}

/// Creates fixed stage signatures from the schema contract.
void createDescriptorResources(
    IRModule* module,
    TraceProgramDescriptor& programDescriptor,
    Dictionary<IRType*, IRFunc*>& instanceContributions,
    const MetalRTCandidateResources& candidateResources)
{
    IRBuilder builder(module);
    auto descriptorPointer = builder.getPtrType(builder.getUIntType(), AddressSpace::Uniform);
    auto instancePathType = programDescriptor.instancePathType;
    programDescriptor.resources = createMetalRTDescriptorResources(
        module,
        programDescriptor.sourceType,
        programDescriptor.tags,
        programDescriptor.maxLevels,
        getVisibleFunctionSignature(builder, programDescriptor.missRequirements, descriptorPointer),
        getVisibleFunctionSignature(builder, programDescriptor.hitRequirements, descriptorPointer),
        getCallableFunctionSignature(
            builder,
            programDescriptor.callableRequirements,
            descriptorPointer),
        candidateResources);
    if (programDescriptor.tags & UInt(MetalRTTag::Instancing))
    {
        if (!instanceContributions.containsKey(instancePathType))
            instanceContributions.add(
                instancePathType,
                createInstanceContribution(module, instancePathType));
        programDescriptor.instanceContribution = instanceContributions.getValue(instancePathType);
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
void lowerTrace(
    IRModule* module,
    IRMetalTraceRay* trace,
    TraceProgramDescriptor& programDescriptor,
    const MetalRTAccelerationStructureResources& accelerationStructureResources)
{
    auto payloadType = cast<IRPtrTypeBase>(trace->getPayload()->getDataType())->getValueType();
    auto& carrier = getCarrier(module, programDescriptor, payloadType);
    IRBuilder builder(module);
    builder.setInsertBefore(trace);
    auto load = [&](MetalRTDescriptorResource role)
    {
        return loadResource(
            builder,
            trace->getDescriptor(),
            programDescriptor.resources.getField(role));
    };
    auto records = load(MetalRTDescriptorResource::Records);
    // The native AS and its metadata are loaded from the same selected argument buffer.
    // For example, choosing between two AS values must also choose between their contribution
    // arrays, even when both traces share one program descriptor and its shader records.
    auto accelerationStructure = loadResource(
        builder,
        trace->getAccelerationStructure(),
        accelerationStructureResources.nativeASField);
    auto instanceContributionMetadata = loadResource(
        builder,
        trace->getAccelerationStructure(),
        accelerationStructureResources.metadataField);
    auto rayData = builder.emitVar(carrier.type);
    builder.addNameHintDecoration(rayData, toSlice("rayData"));
    builder.emitStore(rayData, builder.emitDefaultConstruct(carrier.type));
    auto prefix = builder.emitFieldAddress(rayData, carrier.prefixKey);
    IRInst* payload = nullptr;
    if (carrier.prefix.payloadKey)
    {
        payload = builder.emitFieldAddress(prefix, carrier.prefix.payloadKey);
        builder.emitStore(payload, builder.emitLoad(trace->getPayload()));
    }
    auto header = builder.emitFieldAddress(prefix, carrier.prefix.headerKey);
    const auto& headerType = *programDescriptor.header;
    builder.emitStore(builder.emitFieldAddress(header, headerType.recordsKey), records);
    builder.emitStore(
        builder.emitFieldAddress(header, headerType.instanceContributionMetadataKey),
        instanceContributionMetadata);
    builder.emitStore(
        builder.emitFieldAddress(header, headerType.sbtOffsetKey),
        trace->getSbtOffset());
    builder.emitStore(
        builder.emitFieldAddress(header, headerType.sbtStrideKey),
        trace->getSbtStride());
    builder.emitStore(
        builder.emitFieldAddress(header, headerType.hitRecordStrideKey),
        builder.getIntValue(builder.getUIntType(), programDescriptor.hitStride));
    auto candidateField =
        programDescriptor.resources.getField(MetalRTDescriptorResource::CandidateResources);
    auto candidateAddress = builder.emitFieldAddress(
        builder.getPtrType(candidateField->getFieldType(), AddressSpace::Uniform),
        trace->getDescriptor(),
        candidateField->getKey());
    builder.emitStore(
        builder.emitFieldAddress(header, headerType.candidateResourcesKey),
        candidateAddress);
    auto time = trace->getOperandCount() == 13 ? trace->getTime()
                                               : builder.getFloatValue(builder.getFloatType(), 0);
    builder.emitStore(
        builder.emitFieldAddress(header, headerType.minDistanceKey),
        trace->getRayTMin());
    builder.emitStore(
        builder.emitFieldAddress(header, headerType.rayFlagsKey),
        trace->getRayFlags());
    builder.emitStore(builder.emitFieldAddress(header, headerType.rayTimeKey), time);
    auto uint3 = builder.getVectorType(builder.getUIntType(), 3);
    auto index = builder.emitIntrinsicInst(uint3, kIROp_MetalRTGetDispatchRaysIndex, 0, nullptr);
    auto dimensions =
        builder.emitIntrinsicInst(uint3, kIROp_MetalRTGetDispatchRaysDimensions, 0, nullptr);
    builder.emitStore(builder.emitFieldAddress(header, headerType.dispatchRaysIndexKey), index);
    builder.emitStore(
        builder.emitFieldAddress(header, headerType.dispatchRaysDimensionsKey),
        dimensions);
    auto erasedPointer = builder.getPtrType(builder.getVoidType(), AddressSpace::ThreadLocal);
    auto resourcePointer = builder.getPtrType(builder.getUInt8Type(), AddressSpace::ThreadLocal);
    auto customHitKind = builder.emitFieldAddress(header, headerType.customHitKindKey);
    auto attributes = carrier.attributesKey
                          ? builder.emitBitCast(
                                erasedPointer,
                                builder.emitFieldAddress(rayData, carrier.attributesKey))
                          : builder.getNullPtrValue(erasedPointer);
    auto attributesOffset =
        builder.emitFieldAddress(header, headerType.committedAttributesOffsetKey);
    auto tables = load(MetalRTDescriptorResource::Tables);
    auto tableEntry = builder.emitGetOffsetPtr(tables, trace->getSbtOffset());
    auto tableHandle = builder.emitLoad(
        builder.emitFieldAddress(tableEntry, programDescriptor.resources.tableField->getKey()));
    auto table = builder.emitIntrinsicInst(
        cast<IRDescriptorHandleType>(tableHandle->getDataType())->getResourceType(),
        kIROp_CastDescriptorHandleToResource,
        1,
        &tableHandle);
    auto integer = [&](IRIntegerValue value)
    { return builder.getIntValue(builder.getIntType(), value); };
    IRInst* operands[] = {
        integer(programDescriptor.tags),
        integer(programDescriptor.maxLevels),
        integer(programDescriptor.missRequirements.mask),
        integer(programDescriptor.hitRequirements.mask),
        integer(IRIntegerValue(programDescriptor.geometry)),
        builder.getBoolValue(programDescriptor.hasCandidates),
        builder.getBoolValue(programDescriptor.schema.missShaders.getCount() != 0),
        builder.getBoolValue(programDescriptor.schema.hitGroups.getCount() != 0),
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
        accelerationStructure,
        table,
        load(MetalRTDescriptorResource::MissVisibleFunctionTable),
        load(MetalRTDescriptorResource::ClosestHitVisibleFunctionTable),
        trace->getDescriptor(),
        records,
        builder.getIntValue(builder.getUIntType(), programDescriptor.hitStride),
        builder.getIntValue(builder.getUIntType(), programDescriptor.missStride),
        rayData,
        builder.getBoolValue(true),
        builder.getBoolValue(true),
        builder
            .emitIntrinsicInst(resourcePointer, kIROp_MetalRTGetGlobalResourceContext, 0, nullptr),
        programDescriptor.instanceContribution
            ? static_cast<IRInst*>(programDescriptor.instanceContribution)
            : integer(0),
        instanceContributionMetadata,
        payload ? builder.emitBitCast(erasedPointer, payload)
                : builder.getNullPtrValue(erasedPointer),
        attributes,
        customHitKind,
        index,
        dimensions,
        attributesOffset,
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
void lowerCallable(
    IRModule* module,
    IRMetalRTCallShader* call,
    TraceProgramDescriptor& programDescriptor)
{
    IRBuilder builder(module);
    builder.setInsertBefore(call);
    auto descriptor = call->getDescriptor();
    auto records = loadResource(
        builder,
        descriptor,
        programDescriptor.resources.getField(MetalRTDescriptorResource::Records));
    auto uint3 = builder.getVectorType(builder.getUIntType(), 3);
    auto zero = builder.emitDefaultConstruct(uint3);
    auto& requirements = programDescriptor.callableRequirements;
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
        builder.getIntValue(builder.getUIntType(), programDescriptor.callableStride),
        builder.getBoolValue(requirements.has(MetalRTStageRequirement::Record)),
        descriptor->getDataType(),
        programDescriptor.resources
            .getField(MetalRTDescriptorResource::CallableVisibleFunctionTable)
            ->getKey(),
        builder.getBoolValue(true),
        builder.emitIntrinsicInst(
            builder.getPtrType(builder.getUInt8Type(), AddressSpace::ThreadLocal),
            kIROp_MetalRTGetGlobalResourceContext,
            0,
            nullptr)};
    builder.emitIntrinsicInst(
        builder.getVoidType(),
        kIROp_MetalRTCallDispatch,
        SLANG_COUNT_OF(operands),
        operands);
    call->removeAndDeallocate();
}

/// Collects TraceRay and CallShader instructions after specialization. Each instruction retains
/// its descriptor type, including when a shader uses CallShader without TraceRay.
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
        indexParameter.legalize(index);
    for (auto dimension : dimensions)
        dimensionsParameter.legalize(dimension);
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
    // 1. Build wrappers for the selected stages while their source input types still describe
    // the portable contract. The AS argument-buffer conversion below replaces those types.
    // Do this before collecting dispatches because candidate wrappers can clone shader bodies.
    IRBuilder builder(module);
    auto candidateResources =
        createMetalRTCandidateResources(module, getAnyHitFunctionSignature(builder));
    auto header = createRayDataHeader(module, candidateResources);
    auto adapters = createMetalStageAdapters(
        module,
        targetProgram,
        entryPoints,
        header,
        candidateResources,
        requiresResultTypeLowering,
        sink);
    if (sink->getErrorCount())
        return;

    // Metal launches ray generation as a compute kernel, even when it does not trace any rays.
    List<IRMetalTraceRay*> traces;
    List<IRMetalRTCallShader*> calls;
    // Collect TraceRay and CallShader instructions.
    collectDispatches(module->getModuleInst(), traces, calls);

    for (auto entry : entryPoints)
    {
        auto decoration = entry->findDecoration<IREntryPointDecoration>();
        if (!decoration || decoration->getProfile().getStage() != Stage::RayGeneration)
            continue;
        decoration->setOperand(
            0,
            builder.getIntValue(builder.getIntType(), Profile(Stage::Compute).raw));
    }

    // 2. Read each schema and prepare its Metal types.
    // Read acceleration-structure and motion types, plus the lists' concrete types, including
    // Payload, Record, Primitive and Attributes. Compute record strides and stage signatures,
    // then choose Metal acceleration-structure types and create descriptor table types.
    Dictionary<IRTraceProgramDescriptorType*, RefPtr<TraceProgramDescriptor>> programDescriptors;
    Dictionary<IRType*, IRFunc*> instanceContributions;
    auto discover = [&](IRTraceProgramDescriptorType* descriptorType, IRInst* useSite)
    {
        if (!programDescriptors.containsKey(descriptorType))
            programDescriptors.add(
                descriptorType,
                createTraceProgramDescriptor(
                    module,
                    targetProgram->getTargetReq(),
                    descriptorType,
                    header,
                    useSite,
                    sink));
    };
    // A separately compiled provider retains the complete component resource layout, even if
    // it never traces a ray. For example, a Miss shader that only reads `background` still has
    // the same `TraceProgramDescriptor<Schema> descriptor` global as ray generation. Physicalize
    // that declaration from its schema contract too; discovering only dispatch operands would
    // leave an opaque source descriptor inside the provider's shared resource context.
    // Concrete hoistable descriptor types live directly in the module after specialization.
    // Snapshot them before creating the target-local types added by createTraceProgramDescriptor().
    List<IRTraceProgramDescriptorType*> descriptorTypes;
    for (auto inst : module->getGlobalInsts())
        if (auto descriptorType = as<IRTraceProgramDescriptorType>(inst))
            descriptorTypes.add(descriptorType);
    for (auto descriptorType : descriptorTypes)
        discover(descriptorType, descriptorType);
    for (auto trace : traces)
    {
        auto descriptorType =
            cast<IRTraceProgramDescriptorType>(trace->getDescriptor()->getDataType());
        discover(descriptorType, trace);
    }
    for (auto call : calls)
        discover(cast<IRTraceProgramDescriptorType>(call->getDescriptor()->getDataType()), call);
    for (auto& pair : programDescriptors)
        if (!pair.second)
            return;
    List<AccelerationStructureUse> accelerationStructures;
    for (auto trace : traces)
    {
        auto type = cast<IRTraceProgramDescriptorType>(trace->getDescriptor()->getDataType());
        accelerationStructures.add(
            {trace->getAccelerationStructure(), programDescriptors.getValue(type)->tags, trace});
    }
    List<MetalRTAccelerationStructureResources> accelerationStructureResources;
    if (!physicalizeAccelerationStructures(
            module,
            accelerationStructures,
            sink,
            accelerationStructureResources))
        return;
    Dictionary<IRType*, const MetalRTAccelerationStructureResources*> resourcesByAccelerationType;
    List<MetalRTArgumentBufferResources> accelerationStructureLayouts;
    for (const auto& resource : accelerationStructureResources)
    {
        resourcesByAccelerationType.add(resource.physicalParameterBlockType, &resource);
        accelerationStructureLayouts.add(resource);
    }
    for (auto& pair : programDescriptors)
        createDescriptorResources(module, *pair.second, instanceContributions, candidateResources);

    // 3. Replace TraceRay and CallShader with MetalRTTraceDispatch and MetalRTCallDispatch.
    // For TraceRay, prepare per-ray storage and copy Payload in and back out around the call.
    // The new instructions carry tables, record strides and data pointers. The Metal emitter
    // later expands them into traversal and function-table calls.
    for (auto trace : traces)
    {
        auto type = cast<IRTraceProgramDescriptorType>(trace->getDescriptor()->getDataType());
        auto resource =
            resourcesByAccelerationType.getValue(trace->getAccelerationStructure()->getDataType());
        lowerTrace(module, trace, *programDescriptors.tryGetValue(type)->Ptr(), *resource);
    }
    for (auto call : calls)
    {
        auto type = cast<IRTraceProgramDescriptorType>(call->getDescriptor()->getDataType());
        lowerCallable(module, call, *programDescriptors.tryGetValue(type)->Ptr());
    }

    // Native candidate callbacks belong to selected stage exports. A tracing entry only
    // selects the host's IFT; it does not generate schema-owned dispatch functions.
    List<MetalRTDescriptorResources> resources;
    for (auto& pair : programDescriptors)
        resources.add(pair.second->resources);

    // 5. Replace logical stage-input accesses with values supplied by the generated wrappers.
    // For example, input.payload and input.record use wrapper arguments. Obtain launch indices
    // and dimensions from compute-kernel parameters, then replace descriptor types and layouts
    // with their concrete Metal representations. The later introduceExplicitGlobalContext pass
    // supplies the shared global-resource pointers.
    lowerVisibleStageInputs(module, adapters);
    lowerLaunchQueries(module);
    physicalizeMetalRTDescriptorResources(module, resources, accelerationStructureLayouts);
}

} // namespace Slang
