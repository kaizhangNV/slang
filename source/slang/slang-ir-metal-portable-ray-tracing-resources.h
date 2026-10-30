#pragma once

#include "slang-ir-insts.h"
#include "slang-ir-metal-portable-ray-tracing-target.h"

namespace Slang
{

/// Owns one schema's physical argument-buffer type. The source descriptor remains the canonical
/// schema identity until physicalization replaces it; fields are indexed by the shared ABI role.
struct MetalRTDescriptorResources
{
    IRTraceProgramDescriptorType* sourceType = nullptr;
    IRUniformParameterGroupType* physicalParameterBlockType = nullptr;
    IRStructType* resourcesStruct = nullptr;
    IRStructField* fields[UInt(MetalRTDescriptorResource::Count)] = {};

    IRStructField* getField(MetalRTDescriptorResource role) const { return fields[UInt(role)]; }
};

/// Creates the physical IFT, five VFTs, and records pointer without changing the source type.
/// Signatures are the already-computed table-wide native signatures, including unused tables.
MetalRTDescriptorResources createMetalRTDescriptorResources(
    IRModule* module,
    IRTraceProgramDescriptorType* sourceType,
    UInt tagMask,
    IRIntegerValue maxLevels,
    IRFuncType* missSignature,
    IRFuncType* closestHitSignature,
    IRFuncType* callableSignature,
    IRFuncType* anyHitSignature,
    IRFuncType* intersectionSignature);

/// Replaces the listed opaque descriptor types and their existing storage layouts in one walk.
/// Concrete type/layout pairs distinguish shared field keys in different generic specializations.
/// Existing bindings and array strides remain authoritative; only the opaque buffer's private
/// contents become visible to target legalization. Source descriptor pointers are invalid
/// afterward.
void physicalizeMetalRTDescriptorResources(
    IRModule* module,
    const List<MetalRTDescriptorResources>& resources);

} // namespace Slang
