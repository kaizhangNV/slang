#pragma once

#include "slang-ir-metal-portable-ray-tracing-target.h"
#include "slang-ir-portable-ray-tracing-schema.h"

namespace Slang
{
class DiagnosticSink;
class TargetProgram;
class TargetRequest;
struct MetalRTCandidateResources;

namespace PortableRT
{
namespace Metal
{

/// Selects the stage-specific fields in a fixed invocation context. Launch information and
/// resource pointers are always present, independently of the shader body.
struct StageRequirements
{
    UInt mask = 0;

    bool has(MetalRTStageRequirement requirement) const { return (mask & UInt(requirement)) != 0; }
    void add(MetalRTStageRequirement requirement) { mask |= UInt(requirement); }
};

/// Holds one canonical context type per VFT stage kind in this module. Caller and provider
/// compilations build the same fixed layouts, regardless of which properties their bodies use.
struct InvocationContexts
{
    IRStructType* closestHit = nullptr;
    IRStructType* miss = nullptr;
    IRStructType* callable = nullptr;

    IRStructType* get(Stage stage) const;
};

/// Creates the fixed context layouts once, before constructing adapters or descriptor tables.
InvocationContexts createInvocationContexts(IRModule* module);

/// Builds the VFT signature accepting one pointer to the caller's per-thread context.
IRFuncType* getInvocationFunctionSignature(IRBuilder& builder, IRStructType* contextType);

/// Defines the schema-independent prefix supplied to every native candidate callback. The
/// selected stage knows its Payload type; byte offsets describe the remaining traversal storage.
struct RayDataHeader
{
    IRStructType* type = nullptr;
    IRStructKey* recordsKey = nullptr;
    IRStructKey* instanceContributionMetadataKey = nullptr;
    IRStructKey* candidateResourcesKey = nullptr;
    IRStructKey* sbtOffsetKey = nullptr;
    IRStructKey* sbtStrideKey = nullptr;
    IRStructKey* hitRecordStrideKey = nullptr;
    IRStructKey* committedAttributesOffsetKey = nullptr;
    IRStructKey* minDistanceKey = nullptr;
    IRStructKey* rayTimeKey = nullptr;
    IRStructKey* rayFlagsKey = nullptr;
    IRStructKey* dispatchRaysIndexKey = nullptr;
    IRStructKey* dispatchRaysDimensionsKey = nullptr;
    IRStructKey* customHitKindKey = nullptr;
};

/// Describes the common header followed by one concrete Payload. Traversal may append committed
/// attribute storage, but separately compiled callbacks need only this prefix and its byte offset.
struct RayData
{
    IRStructType* type = nullptr;
    IRStructKey* payloadKey = nullptr;
    IRStructKey* headerKey = nullptr;
    const RayDataHeader* header = nullptr;
};

/// Builds the fixed header shared by traversal and independently compiled native callbacks.
RayDataHeader createRayDataHeader(
    IRModule* module,
    const MetalRTCandidateResources& candidateResources);

/// Builds the header and Payload prefix without depending on a schema or attribute-type order.
RayData createRayDataPrefix(IRModule* module, const RayDataHeader& header, IRType* payloadType);

/// Uses canonical void as the physical payload key for every genuinely empty source type.
/// Nominal source types remain unchanged; only their zero-byte transport representation is shared.
IRType* getRayPayloadStorageType(IRBuilder& builder, IRType* payloadType);

/// Supplies the source-typed payload reference for one reached accessor. Empty values have no
/// carrier field, so their reference points to a local token removed by empty-type legalization.
IRInst* getRayPayloadReference(
    IRBuilder& builder,
    const RayData& carrier,
    IRInst* rayData,
    IRType* payloadType);

/// Builds a pointer to dataType from the thread void* passed through a Metal function table.
/// Nonempty data keeps its existing address and storage. Void returns a typed null pointer;
/// an empty struct gets a typed local that later empty-type legalization removes.
IRInst* recoverStageDataPointer(IRBuilder& builder, IRInst* untypedDataPointer, IRType* dataType);

/// Records which wrapper value supplies an input-read instruction with a given opcode and type.
/// For example, {MetalRTGetCurrentDistance, float, distanceValue} tells lowerVisibleStageInputs
/// to pass the loaded distance field through source helpers and replace their matching
/// reads. The getter's result type can differ from the supplied value's type: an instance-index
/// getter returns uint, while its wrapper supplies a pointer to the array of instance indices.
struct StageInputReplacement
{
    IROp op;
    IRType* resultType;
    IRInst* value;
};

/// Returns the generated wrapper and the input replacement rules for lowerVisibleStageInputs.
/// That later step runs after TraceRay/CallShader lowering has created any additional input reads.
struct VisibleAdapter
{
    IRFunc* function = nullptr;
    List<StageInputReplacement> inputReplacements;
    IRInst* descriptorResources = nullptr;
    IRInst* records = nullptr;
};

/// Selects a stage ABI from its declared kind. Hit stages receive a path for every topology;
/// ordinary instancing supplies one element. Shader bodies cannot change this contract, so
/// caller and selected-stage compilations agree independently.
StageRequirements getStageContractRequirements(Stage stage);

/// Collects reachable invocation getters, excluding dispatch and candidate-control operations.
/// Adapters use these operations to identify the fixed invocation values that replace getters.
void collectStageInputOperations(IRFunc* invoke, List<IRInst*>& operations);

/// Identifies a logical invocation getter before a selected adapter supplies its value.
bool isStageInputOperation(IROp op);

/// Distinguishes zero-storage built-in attribute views using the getter's checked primitive.
bool isBuiltinHitAttributeView(IRInst* operation);

/// Creates one closest-hit, miss, or callable VFT function without changing its callable source
/// method. Create every adapter before lowerVisibleStageInputs so a shared method sees all its
/// callers.
VisibleAdapter createVisibleAdapter(
    IRModule* module,
    TargetRequest* target,
    const SchemaStage& stage,
    IRStructType* contextType,
    UnownedStringSlice symbolName);

/// Threads adapter-supplied values through ordinary helpers and replaces their logical getters.
/// Candidate-stage mutable/control state is deliberately outside this visible-stage operation.
void lowerVisibleStageInputs(IRModule* module, const List<VisibleAdapter>& adapters);

} // namespace Metal
} // namespace PortableRT

/// Creates native candidate callbacks and VFT adapters for selected logical stage entry points.
/// No schema discovery is needed: separately compiled callers use the same fixed signatures.
/// Removes those logical entries from entryPoints, leaving ordinary native entries in that list,
/// and returns the generated adapters separately.
/// Include these roots in the shared lowerVisibleStageInputs batch so ordinary source helpers
/// receive the invocation state of their actual caller.
List<PortableRT::Metal::VisibleAdapter> createMetalStageAdapters(
    IRModule* module,
    TargetProgram* targetProgram,
    List<IRFunc*>& entryPoints,
    const PortableRT::Metal::RayDataHeader& header,
    const MetalRTCandidateResources& candidateResources,
    const PortableRT::Metal::InvocationContexts& invocationContexts,
    bool& requiresResultTypeLowering,
    DiagnosticSink* sink);

} // namespace Slang
