#pragma once

#include "slang-ir.h"
#include "slang-structural-ray-tracing.h"

// This file declares the normalization boundary that turns trusted structural ray-tracing source
// interfaces into distinct IR interface types. Consumers can therefore recognize stage contracts
// from IR identity without retaining AST declarations or consulting source names.

namespace Slang
{

class DiagnosticSink;
struct IRModule;

/// Return the compiler-owned interface-type opcode for `kind`, or `kIROp_Invalid` for an invalid
/// kind. Distinct opcodes preserve structural stage identity after source declarations are linked
/// or serialized and their AST pointers are no longer available.
IROp getStructuralRayTracingStageInterfaceOp(StructuralRayTracingStageKind kind);

/// Diagnose a trace-program descriptor type that target lowering did not consume. Supported
/// backends must erase or materialize every descriptor before source/binary emission; keeping this
/// final check independent of target lowering turns an unsupported path into a user diagnostic.
void diagnoseUnloweredTraceProgramDescriptorTypes(IRModule* module, DiagnosticSink* sink);

/// Give every structural stage interface in `module` its compiler-owned IR opcode.
///
/// The registry supplies the trusted AST declarations, and their mangled symbols locate the
/// corresponding `IRInterfaceType` instructions. Returns false when any stage symbol cannot be
/// found; `outMissingStage`, when present, identifies that stage. This function is the only
/// boundary that converts an ordinary serialized interface type into a structural-stage IR type.
bool identifyStructuralRayTracingStageInterfaces(
    Module* module,
    const StructuralRayTracingDeclRegistry& registry,
    StructuralRayTracingStageKind* outMissingStage = nullptr);

} // namespace Slang
