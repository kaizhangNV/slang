#pragma once

#include "slang-ir-metal-portable-ray-tracing-stages.h"

namespace Slang
{
class DiagnosticSink;
class TargetProgram;

/// Creates VFT adapters for explicitly selected stages from their checked Context contracts.
/// No schema discovery is needed: separately compiled callers use the same fixed signatures.
/// Include these roots in the shared lowerVisibleStageInputs batch so ordinary source helpers
/// receive the invocation state of their actual caller.
List<PortableRT::Metal::VisibleAdapter> prepareMetalStandaloneStages(
    IRModule* module,
    TargetProgram* targetProgram,
    List<IRFunc*>& entryPoints,
    bool& requiresResultTypeLowering,
    DiagnosticSink* sink);
} // namespace Slang
