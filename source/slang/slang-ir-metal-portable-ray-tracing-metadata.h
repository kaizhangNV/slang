#pragma once

#include "slang-ir.h"

namespace Slang
{
class ArtifactPostEmitMetadata;

/// Creates an artifact metadata record and links its owning descriptor resources to it.
void addMetalRTProgramMetadata(
    IRModule* module,
    IRStructType* resources,
    UInt tags,
    IRIntegerValue maxLevels,
    IRIntegerValue hitStride,
    IRIntegerValue missStride,
    IRIntegerValue callableStride,
    UInt hitGroupCount,
    UInt missShaderCount,
    UInt callableShaderCount,
    UInt candidateFunctionTableSize);

/// Associates an exported adapter with its owning descriptor and native table index.
void addMetalRTFunctionMetadata(
    IRModule* module,
    IRFunc* function,
    IRStructType* resources,
    UInt functionIndex,
    UInt requirements);

/// Copies schema-defined table capacities and this artifact's actual exports into metadata.
void collectMetalRTMetadata(const IRModule* module, ArtifactPostEmitMetadata& metadata);
} // namespace Slang
