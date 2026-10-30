#pragma once

#include "slang-ir-metal-portable-ray-tracing-stages.h"

namespace Slang
{
class DiagnosticSink;
class TargetProgram;

/// Makes explicitly selected post-traversal stages ordinary exported Metal helpers. The caller
/// supplies only the invocation values they use; a schema is required to form an actual VFT ABI.
/// Include the returned roots in the same lowerVisibleStageInputs batch as schema adapters so a
/// shared static implementation receives the correct caller's state in both compilation modes.
List<PortableRT::Metal::VisibleAdapter> prepareMetalStandaloneStages(
    IRModule* module,
    TargetProgram* targetProgram,
    List<IRFunc*>& entryPoints,
    DiagnosticSink* sink);
} // namespace Slang
