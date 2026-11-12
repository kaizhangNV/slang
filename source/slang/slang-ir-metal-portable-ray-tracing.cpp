#include "slang-ir-metal-portable-ray-tracing.h"

#include "slang-ir-layout.h"
#include "slang-ir-metal-portable-ray-tracing-acceleration.h"
#include "slang-ir-metal-portable-ray-tracing-candidates.h"
#include "slang-ir-metal-portable-ray-tracing-instance-path.h"
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
    StageRequirements callableRequirements;
    StageRequirements allRequirements;
    RayDataHeader header;
    Dictionary<IRType*, RayData> carriers;
    IRIntegerValue hitStride = kMetalRTRecordHeaderSize;
    IRIntegerValue missStride = kMetalRTRecordHeaderSize;
    IRIntegerValue callableStride = kMetalRTRecordHeaderSize;
    IRFunc* instanceContribution = nullptr;
    IRType* instancePathType = nullptr;
    bool hasCandidates = false;
    bool isTraced = false;
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

/// Constructs invocation storage from the schema contract. The exact payload IR type is the cache
/// key: a trace and its compatible stage therefore use one carrier without reconstructing or
/// comparing source syntax. Empty source payloads share canonical void as their storage key. In
/// particular, the library's payload-less overload and an application's empty payload need the same
/// procedural attributes.
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

/// Prepares the Metal traversal settings, stage arguments, and record strides for one schema.
/// `type` is a TraceProgramDescriptor<Schema> whose associated types have already been resolved
/// by specialization. The returned Program holds the choices used by createProgramResources
/// to build descriptor resource types and by lowerTrace/lowerCallable to lower their operations.
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
/// resources, and traversal callbacks are built later from the returned Program. Shader invoke
/// bodies are compiled when their entries are selected separately. An unsupported traversal
/// setting or Record layout is diagnosed at useSite and returns nullptr.
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
    // Read the concrete acceleration-structure and motion types, each hit group's Payload,
    // Record, Primitive and Attributes, and the ordered miss/callable Record lists. The hit
    // groups also say whether closest-hit, any-hit and intersection shaders are present.
    readSchemaContract(type, program.schema);

    // Translate the declared acceleration-structure topology and motion mode into Metal options
    // such as instancing and primitive_motion, plus a native traversal-depth limit. Check that
    // the target supports these choices.
    TraceContextRequirements contextRequirements;
    if (!getTraceContextRequirements(program.schema, target, useSite, sink, contextRequirements))
        return nullptr;
    program.tags = contextRequirements.tags;
    program.maxLevels = contextRequirements.maxLevels;
    // Every later Payload storage type starts with this header: the records-buffer pointer
    // and the TraceRay call's SBT offset/stride, which traversal callbacks need to find a record.
    program.header = createHeader(module);
    auto accelerationStructure =
        cast<IRRaytracingAccelerationStructureType>(program.schema.accelerationStructure);
    // An explicit topology uses instance-path arguments, including an explicit one-level
    // topology. This describes the source input contract, not whether maxLevels is greater than 1.
    bool hasInstancePath = accelerationStructure->getOperandCount() != 0;
    // Fix the values passed to each visible shader, such as Record, ray origin and launch index.
    // Stage and topology determine these arguments; inspecting shader bodies would make the
    // calling convention depend on which implementations happen to be compiled together.
    program.hitRequirements = getStageContractRequirements(Stage::ClosestHit, hasInstancePath);
    program.missRequirements = getStageContractRequirements(Stage::Miss, hasInstancePath);
    program.callableRequirements = getStageContractRequirements(Stage::Callable, hasInstancePath);
    // getCarrier uses this union to decide which values travel with the ray's Payload.
    // CallShader passes its CallableData pointer directly and does not use that ray storage.
    program.allRequirements.combine(program.hitRequirements);
    program.allRequirements.combine(program.missRequirements);

    bool firstGeometry = true;
    for (auto& group : program.schema.hitGroups)
    {
        // A single primitive family permits a Metal geometry assumption. Mixed families leave
        // geometry Unknown, so the emitted intersector does not assume triangles or another kind.
        auto geometry = getGeometry(group.primitiveType);
        program.geometry =
            firstGeometry
                ? geometry
                : (program.geometry == geometry ? geometry : MetalRTGeometryKind::Unknown);
        firstGeometry = false;
        // Any-hit or intersection requires candidate callbacks during traversal. The Metal
        // emitter uses this flag to pass the intersection-function table and ray data to intersect.
        program.hasCandidates |= group.anyHit.isPresent || group.intersection.isPresent;
        // A triangle group can expose barycentrics; a curve group can expose curve parameters.
        // These tags request those traversal values even if no shader body reads them.
        if (geometry == MetalRTGeometryKind::Triangle)
            program.tags |= UInt(MetalRTTag::TriangleData);
        if (geometry == MetalRTGeometryKind::Curve)
            program.tags |= UInt(MetalRTTag::CurveData);
        // All runtime hit records use one stride, large enough for every group's Record.
        // getRecordStride includes the header and alignment and rejects unsupported layouts.
        IRIntegerValue stride;
        if (!getRecordStride(target, group.recordType, useSite, sink, stride))
            return nullptr;
        program.hitStride = Math::Max(program.hitStride, stride);
    }
    // Miss and callable records use separate sections, so compute their maximum strides
    // independently. Program initializes each stride to the header size, including for void data.
    for (const auto& miss : program.schema.missShaders)
    {
        IRIntegerValue stride;
        if (!getRecordStride(target, miss.recordType, useSite, sink, stride))
            return nullptr;
        program.missStride = Math::Max(program.missStride, stride);
    }
    for (const auto& callable : program.schema.callableShaders)
    {
        IRIntegerValue stride;
        if (!getRecordStride(target, callable.recordType, useSite, sink, stride))
            return nullptr;
        program.callableStride = Math::Max(program.callableStride, stride);
    }
    // Instanced hit shaders need the traversal result's transforms and object-space ray values.
    // Request those values even if the selected implementations do not read them.
    if ((program.tags & UInt(MetalRTTag::Instancing)) && program.schema.hitGroups.getCount())
        program.tags |= UInt(MetalRTTag::WorldSpaceData);

    IRBuilder builder(module);
    // Use a Metal path type for the instance-path contract and an integer index for ordinary
    // instancing. createProgramResources uses it for the SBT-contribution lookup when instancing
    // is enabled.
    program.instancePathType =
        hasInstancePath ? builder.getType(kIROp_MetalRTInstancePath) : builder.getUIntType();
    return result;
}

/// Creates fixed stage signatures from the schema contract.
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
        getVisibleFunctionSignature(builder, program.missRequirements, descriptorPointer),
        getVisibleFunctionSignature(builder, program.hitRequirements, descriptorPointer),
        getCallableFunctionSignature(builder, program.callableRequirements, descriptorPointer),
        getAnyHitFunctionSignature(builder),
        getIntersectionFunctionSignature(builder));
    if (program.tags & UInt(MetalRTTag::Instancing))
    {
        if (!instanceContributions.containsKey(instancePathType))
            instanceContributions.add(
                instancePathType,
                createInstanceContribution(module, instancePathType));
        program.instanceContribution = instanceContributions.getValue(instancePathType);
    }
}

/// Creates traversal glue from schema types and table indices. Source stage bodies are emitted
/// only when selected independently; the glue calls their visible-function table entries.
void createProgramCandidateDispatch(IRModule* module, Program& program)
{
    for (auto& group : program.schema.hitGroups)
        getCarrier(module, program, group.payloadType);
    StringBuilder candidatePrefix;
    candidatePrefix << "__slang_portable_rt_" << program.index << "_candidate";
    createCandidateAdapters(
        module,
        program.schema,
        program.resources,
        program.carriers,
        program.header,
        program.tags,
        program.maxLevels,
        program.hitStride,
        program.instanceContribution,
        program.instancePathType,
        candidatePrefix.getUnownedSlice());
}

/// Loads a named resource from the descriptor using the physical field's exact type. The
/// descriptor is still opaque at this point; the following physicalization step replaces that
/// type everywhere, including helper signatures and nested global resources.
IRInst* loadResource(IRBuilder& builder, IRInst* descriptor, IRStructField* field)
{
    auto addressType = builder.getPtrType(field->getFieldType(), AddressSpace::Uniform);
    return builder.emitLoad(builder.emitFieldAddress(addressType, descriptor, field->getKey()));
}

/// Selects committed custom attributes by the schema's group index. Consider two procedural
/// groups with different attribute types: their fields belong to the caller's carrier, while
/// the separately compiled closest-hit shader sees only a pointer to its own attributes.
/// Builtin primitives and empty attributes have no custom storage and return a null pointer.
IRFunc* createAttributesLookup(IRModule* module, const Program& program, const RayData& carrier)
{
    IRBuilder builder(module);
    builder.setInsertInto(module);
    auto result = builder.createFunc();
    auto pointer = builder.getPtrType(builder.getVoidType(), AddressSpace::ThreadLocal);
    IRType* parameterTypes[] = {
        builder.getPtrType(carrier.type, AddressSpace::ThreadLocal),
        builder.getUIntType()};
    result->setFullType(builder.getFuncType(2, parameterTypes, pointer));
    builder.addNameHintDecoration(result, toSlice("getCommittedAttributes"));
    builder.setInsertInto(result);
    builder.emitBlock();
    auto data = builder.emitParam(parameterTypes[0]);
    auto index = builder.emitParam(parameterTypes[1]);
    IRInst* value = builder.getNullPtrValue(pointer);
    for (Index i = 0; i < program.schema.hitGroups.getCount(); ++i)
    {
        const auto& group = program.schema.hitGroups[i];
        auto key = carrier.customAttributeKeys.tryGetValue(group.attributesType);
        if (!key || isEmptyType(group.attributesType))
            continue;
        auto address = builder.emitBitCast(pointer, builder.emitFieldAddress(data, *key));
        IRInst* operands[] = {
            builder.emitEql(index, builder.getIntValue(builder.getUIntType(), i)),
            address,
            value};
        value = builder.emitIntrinsicInst(pointer, kIROp_Select, 3, operands);
    }
    builder.emitReturn(value);
    return result;
}

/// Tests schema membership without requiring a no-op implementation for NoClosestHit.
IRFunc* createClosestHitPresenceLookup(IRModule* module, const Program& program)
{
    IRBuilder builder(module);
    builder.setInsertInto(module);
    auto result = builder.createFunc();
    IRType* parameterTypes[] = {builder.getUIntType()};
    result->setFullType(builder.getFuncType(1, parameterTypes, builder.getBoolType()));
    builder.addNameHintDecoration(result, toSlice("hasClosestHit"));
    builder.setInsertInto(result);
    builder.emitBlock();
    auto index = builder.emitParam(builder.getUIntType());
    IRInst* value = builder.getBoolValue(false);
    for (Index i = 0; i < program.schema.hitGroups.getCount(); ++i)
    {
        if (!program.schema.hitGroups[i].closestHit.isPresent)
            continue;
        IRInst* operands[] = {
            builder.emitEql(index, builder.getIntValue(builder.getUIntType(), i)),
            builder.getBoolValue(true),
            value};
        value = builder.emitIntrinsicInst(builder.getBoolType(), kIROp_Select, 3, operands);
    }
    builder.emitReturn(value);
    return result;
}

/// Expands the trace boundary, leaving traversal and the post-trace choice explicit for emission.
/// The payload is copied into ray_data before traversal and back afterward; stages all reference
/// that invocation's carrier rather than independent copies of the logical input object.
void lowerTrace(IRModule* module, IRMetalTraceRay* trace, Program& program)
{
    auto payloadType = cast<IRPtrTypeBase>(trace->getPayload()->getDataType())->getValueType();
    auto& carrier = getCarrier(module, program, payloadType);
    auto attributesLookup = createAttributesLookup(module, program, carrier);
    auto closestHitPresence = createClosestHitPresenceLookup(module, program);
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
    auto index = builder.emitIntrinsicInst(uint3, kIROp_MetalRTGetDispatchRaysIndex, 0, nullptr);
    auto dimensions =
        builder.emitIntrinsicInst(uint3, kIROp_MetalRTGetDispatchRaysDimensions, 0, nullptr);
    builder.emitStore(builder.emitFieldAddress(rayData, carrier.dispatchRaysIndexKey), index);
    builder.emitStore(
        builder.emitFieldAddress(rayData, carrier.dispatchRaysDimensionsKey),
        dimensions);
    auto erasedPointer = builder.getPtrType(builder.getVoidType(), AddressSpace::ThreadLocal);
    auto resourcePointer = builder.getPtrType(builder.getUInt8Type(), AddressSpace::ThreadLocal);
    auto customHitKind =
        carrier.customHitKindKey
            ? builder.emitFieldAddress(rayData, carrier.customHitKindKey)
            : builder.getNullPtrValue(
                  builder.getPtrType(builder.getUIntType(), AddressSpace::ThreadLocal));
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
        builder.getBoolValue(true),
        builder.getBoolValue(true),
        builder
            .emitIntrinsicInst(resourcePointer, kIROp_MetalRTGetGlobalResourceContext, 0, nullptr),
        program.instanceContribution ? static_cast<IRInst*>(program.instanceContribution)
                                     : integer(0),
        payload ? builder.emitBitCast(erasedPointer, payload)
                : builder.getNullPtrValue(erasedPointer),
        attributesLookup,
        customHitKind,
        index,
        dimensions,
        closestHitPresence,
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
/// its descriptor type, including when the program uses CallShader without TraceRay.
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
    // 1. Make each selected ray-generation entry a compute entry.
    // Metal launches ray generation as a compute kernel, even when it does not trace any rays.
    List<IRMetalTraceRay*> traces;
    List<IRMetalRTCallShader*> calls;
    // Collect TraceRay and CallShader instructions.
    collectDispatches(module->getModuleInst(), traces, calls);

    IRBuilder builder(module);
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
    Dictionary<IRTraceProgramDescriptorType*, RefPtr<Program>> programs;
    Dictionary<IRType*, IRFunc*> instanceContributions;
    auto discover = [&](IRTraceProgramDescriptorType* descriptorType, IRInst* useSite)
    {
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
    // A separately compiled provider retains the complete component resource layout, even if
    // it never traces a ray. For example, a Miss shader that only reads `background` still has
    // the same `TraceProgramDescriptor<Schema> descriptor` global as ray generation. Physicalize
    // that declaration from its schema contract too; discovering only dispatch operands would
    // leave an opaque source descriptor inside the provider's shared resource context.
    // Concrete hoistable descriptor types live directly in the module after specialization.
    // Snapshot them before creating the target-local types added by createProgram().
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
        if (auto program = programs.getValue(descriptorType))
            program->isTraced = true;
    }
    for (auto call : calls)
        discover(cast<IRTraceProgramDescriptorType>(call->getDescriptor()->getDataType()), call);
    for (auto& pair : programs)
        if (!pair.second)
            return;
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

    // 3. Replace TraceRay and CallShader with MetalRTTraceDispatch and MetalRTCallDispatch.
    // For TraceRay, prepare per-ray storage and copy Payload in and back out around the call.
    // The new instructions carry tables, record strides and data pointers. The Metal emitter
    // later expands them into traversal and function-table calls.
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

    // 4. Generate wrappers for the shader entries explicitly selected by the application.
    // For schemas used by TraceRay, generate any needed traversal callbacks that invoke any-hit and
    // intersection shaders through function tables. Listing a shader in a schema does not
    // select its invoke body for compilation.
    auto adapters = prepareMetalStandaloneStages(
        module,
        targetProgram,
        entryPoints,
        requiresResultTypeLowering,
        sink);
    List<MetalRTDescriptorResources> resources;
    for (auto& pair : programs)
    {
        auto& program = *pair.second;
        // Native intersection callbacks belong to traversal artifacts. A provider only needs
        // the descriptor's fixed resource type and must not export another set of callbacks.
        if (program.isTraced)
            createProgramCandidateDispatch(module, program);
        resources.add(program.resources);
    }

    // 5. Replace logical stage-input accesses with values supplied by the generated wrappers.
    // For example, input.payload and input.record use wrapper arguments. Obtain launch indices
    // and dimensions from compute-kernel parameters, then replace descriptor types and layouts
    // with their concrete Metal representations. The later introduceExplicitGlobalContext pass
    // supplies the shared global-resource pointers.
    lowerVisibleStageInputs(module, adapters);
    lowerLaunchQueries(module);
    physicalizeMetalRTDescriptorResources(module, resources);
}

} // namespace Slang
