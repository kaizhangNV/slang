#pragma once

#include "slang-ir-insts.h"
#include "slang-ir-metal-portable-ray-tracing-target.h"

namespace Slang
{

/// Describes a synthesized argument buffer containing Metal resource IDs, GPU pointers, and
/// inline resource structs. Descriptor and acceleration-structure resources share layout creation.
struct MetalRTArgumentBufferResources
{
    IRUniformParameterGroupType* physicalParameterBlockType = nullptr;
    IRStructType* resourcesStruct = nullptr;
};

/// Holds the inline descriptor fields needed by a native candidate callback. All descriptors
/// share this type so a callback can use its pointer without knowing the calling schema.
/// Fields use DescriptorHandle storage, matching Metal argument-buffer element lowering. This
/// also keeps raw pointers to the subobject intact when resource legalization splits logical types.
struct MetalRTCandidateResources
{
    IRStructType* type = nullptr;
    IRStructField* anyHitFunctions = nullptr;
};

/// Owns one schema's physical argument-buffer type. The source descriptor remains the canonical
/// schema identity until physicalization replaces it; fields are indexed by the shared ABI role.
struct MetalRTDescriptorResources : MetalRTArgumentBufferResources
{
    IRTraceProgramDescriptorType* sourceType = nullptr;
    MetalRTCandidateResources candidateResources;
    IRStructType* tableEntryType = nullptr;
    IRStructField* tableField = nullptr;
    IRStructField* fields[UInt(MetalRTDescriptorResource::Count)] = {};

    IRStructField* getField(MetalRTDescriptorResource role) const { return fields[UInt(role)]; }
};

/// Creates the common inline candidate-resource type containing the procedural AnyHit VFT.
MetalRTCandidateResources createMetalRTCandidateResources(
    IRModule* module,
    IRFuncType* anyHitSignature);

/// Creates the fixed descriptor's IFT-array pointer, four VFTs, and records pointer. The AnyHit
/// VFT is an inline candidate-resource subobject; the source descriptor type stays unchanged.
MetalRTDescriptorResources createMetalRTDescriptorResources(
    IRModule* module,
    IRTraceProgramDescriptorType* sourceType,
    UInt tagMask,
    IRIntegerValue maxLevels,
    IRFuncType* missSignature,
    IRFuncType* closestHitSignature,
    IRFuncType* callableSignature,
    const MetalRTCandidateResources& candidateResources);

/// Replaces the listed opaque descriptor types and their existing storage layouts in one walk.
/// Concrete type/layout pairs distinguish shared field keys in different generic specializations.
/// Existing bindings and array strides remain authoritative; only the opaque buffer's private
/// contents become visible to target legalization. Source descriptor pointers are invalid
/// afterward.
void physicalizeMetalRTDescriptorResources(
    IRModule* module,
    const List<MetalRTDescriptorResources>& resources,
    const List<MetalRTArgumentBufferResources>& additionalResources = {});

} // namespace Slang
