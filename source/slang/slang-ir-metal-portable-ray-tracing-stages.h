#pragma once

#include "slang-ir-metal-portable-ray-tracing-target.h"
#include "slang-ir-portable-ray-tracing-schema.h"

namespace Slang
{
namespace PortableRT
{
namespace Metal
{

/// Describes the fixed invocation contract shared by independently compiled stages. The mask
/// selects VFT parameters; the booleans select state carried from the originating trace.
struct StageRequirements
{
    UInt mask = 0;
    bool minDistance = false;
    bool rayTime = false;
    bool rayFlags = false;
    bool dispatchRaysIndex = false;
    bool dispatchRaysDimensions = false;

    bool has(MetalRTStageRequirement requirement) const { return (mask & UInt(requirement)) != 0; }
    void add(MetalRTStageRequirement requirement) { mask |= UInt(requirement); }

    /// Combines the contracts whose state a shared traversal carrier must transport.
    void combine(const StageRequirements& other);
};

/// Names the runtime record selectors shared by every payload carrier in one schema.
struct RayDataHeader
{
    IRStructType* type = nullptr;
    IRStructKey* recordsKey = nullptr;
    IRStructKey* sbtOffsetKey = nullptr;
    IRStructKey* sbtStrideKey = nullptr;
};

/// Refers to the one canonical carrier created by descriptor lowering. This is a transient view
/// of its fields, not an alternative IR identity; the owning lowering context retains its lifetime.
struct RayData
{
    IRStructType* type = nullptr;
    IRStructKey* payloadKey = nullptr;
    IRStructKey* headerKey = nullptr;
    RayDataHeader* header = nullptr;
    IRStructKey* minDistanceKey = nullptr;
    IRStructKey* rayTimeKey = nullptr;
    IRStructKey* rayFlagsKey = nullptr;
    IRStructKey* dispatchRaysIndexKey = nullptr;
    IRStructKey* dispatchRaysDimensionsKey = nullptr;
    IRStructKey* customHitKindKey = nullptr;
    Dictionary<IRType*, IRStructKey*> customAttributeKeys;
};

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

/// Restores a stage's nominal data reference from its erased ABI pointer. Void and empty source
/// types carry no bytes; their references remain nominal tokens until ordinary legalization.
IRInst* getStageDataReference(IRBuilder& builder, IRInst* erasedAddress, IRType* dataType);

/// Binds one logical accessor result to a concrete adapter value. A reference getter's logical
/// result type can differ from the value's physical pointer type, which retains its address space.
struct StageInputBinding
{
    IROp op;
    IRType* resultType;
    IRInst* value;
};

/// Keeps adapter-owned invocation values until all calls have been created and can be threaded.
struct VisibleAdapter
{
    IRFunc* function = nullptr;
    List<StageInputBinding> bindings;
    IRInst* descriptorResources = nullptr;
    IRInst* records = nullptr;
};

/// Selects a stage ABI from its declared kind and acceleration-structure topology. A shader body
/// cannot change this contract, so caller and selected-stage compilations agree independently.
StageRequirements getStageContractRequirements(Stage stage, bool multiLevel = false);

/// Collects reachable invocation getters, excluding dispatch and candidate-control operations.
/// Adapters use these operations only to bind source getters to the fixed invocation values.
void collectStageInputOperations(IRFunc* invoke, List<IRInst*>& operations);

/// Identifies a logical invocation getter before a selected adapter supplies its value.
bool isStageInputOperation(IROp op);

/// Distinguishes zero-storage built-in attribute views using the getter's checked primitive.
bool isBuiltinHitAttributeView(IRInst* operation);

/// Builds the table-wide physical signature used by both VFT resources and their adapters.
IRFuncType* getVisibleFunctionSignature(
    IRBuilder& builder,
    const StageRequirements& requirements,
    IRType* descriptorPointerType);

/// Builds the callable-table ABI: erased data, fixed launch/record inputs, and the descriptor
/// resources needed by nested dispatch. The runtime record still determines the concrete data type.
IRFuncType* getCallableFunctionSignature(
    IRBuilder& builder,
    const StageRequirements& requirements,
    IRType* descriptorPointerType);

/// Creates one closest-hit or miss VFT function without changing its callable source method.
/// Create every adapter before lowerVisibleStageInputs so a shared method sees all its callers.
VisibleAdapter createVisibleAdapter(
    IRModule* module,
    const SchemaStage& stage,
    const StageRequirements& tableRequirements,
    UnownedStringSlice symbolName,
    IRType* descriptorPointerType);

/// Creates a callable VFT function while retaining the ordinary static source implementation.
/// Include the returned root in the same immutable-input batch as visible and candidate stages.
VisibleAdapter createCallableAdapter(
    IRModule* module,
    const SchemaStage& stage,
    const StageRequirements& tableRequirements,
    UnownedStringSlice symbolName,
    IRType* descriptorPointerType);

/// Threads adapter-bound values through ordinary helpers and replaces their logical getters.
/// Candidate-stage mutable/control state is deliberately outside this visible-stage operation.
void lowerVisibleStageInputs(IRModule* module, const List<VisibleAdapter>& adapters);

} // namespace Metal
} // namespace PortableRT
} // namespace Slang
