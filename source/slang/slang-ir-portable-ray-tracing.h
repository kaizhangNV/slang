#pragma once

#include "slang-ir.h"

namespace Slang
{
class TargetProgram;
class DiagnosticSink;

/// Lowers portable ray tracing for the requested target after generic and higher-order
/// specialization, before native ABI legalization. Builds target entry adapters and lowers
/// invocation-state accessors; Metal also lowers schema descriptors and trace/call dispatch.
/// The source static methods remain ordinary functions, so a call from another stage still uses
/// that caller's invocation. Marks requiresResultTypeLowering when generated Metal control flow
/// needs result-type lowering.
void lowerPortableRayTracingEntryPoints(
    IRModule* module,
    TargetProgram* targetProgram,
    List<IRFunc*>& entryPoints,
    DiagnosticSink* sink,
    bool& requiresResultTypeLowering);
} // namespace Slang
