#pragma once

#include "slang-ir.h"

namespace Slang
{
class DiagnosticSink;
class TargetProgram;

/// Materializes a specialized portable schema as Metal argument-buffer resources and callable
/// adapters, then replaces source trace operations with physical traversal and table dispatch.
/// Run after ordinary specialization resolves the schema witnesses and before resource lowering.
void lowerMetalPortableRayTracing(
    IRModule* module,
    TargetProgram* targetProgram,
    List<IRFunc*>& entryPoints,
    DiagnosticSink* sink,
    bool& requiresResultTypeLowering);

/// Diagnoses executable logical operations left without an invocation ABI. Run after DCE, so
/// ordinary library helpers remain valid until selected roots and reachability are known.
void diagnoseUnboundMetalPortableRayTracing(IRModule* module, DiagnosticSink* sink);

} // namespace Slang
