#pragma once

#include "slang-ir-metal-portable-ray-tracing-stages.h"

namespace Slang
{
class DiagnosticSink;
class TargetProgram;

/// Creates VFT adapters for selected, specialized logical stage entry points.
/// No schema discovery is needed: separately compiled callers use the same fixed signatures.
/// Removes those logical entries from entryPoints, leaving ordinary native entries in that list,
/// and returns the generated adapters separately.
/// Include these roots in the shared lowerVisibleStageInputs batch so ordinary source helpers
/// receive the invocation state of their actual caller.
List<PortableRT::Metal::VisibleAdapter> prepareMetalStandaloneStages(
    IRModule* module,
    TargetProgram* targetProgram,
    List<IRFunc*>& entryPoints,
    bool& requiresResultTypeLowering,
    DiagnosticSink* sink);
} // namespace Slang
