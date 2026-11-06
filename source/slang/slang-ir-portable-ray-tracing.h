#pragma once

#include "slang-ir.h"

namespace Slang
{
class TargetProgram;
class DiagnosticSink;

/// Separates selected portable stages from their ordinary static implementations before any
/// pass interprets entry-point parameters as native shader inputs. The initial adapter has no
/// native parameters; its checked input contract is specialized along with the call it owns.
void preparePortableRayTracingEntryPoints(IRModule* module, List<IRFunc*>& entryPoints);

/// Builds native entry-point adapters for checked portable ray-tracing stages and lowers their
/// invocation-state accessors. Run after generic and higher-order specialization, before native
/// payload ABI and varying-parameter legalization. The source static methods remain ordinary
/// functions, so an invoke forwarded from another stage still uses that caller's invocation.
void lowerPortableRayTracingEntryPoints(
    IRModule* module,
    TargetProgram* targetProgram,
    List<IRFunc*>& entryPoints,
    DiagnosticSink* sink);
} // namespace Slang
