// slang-ir-collect-global-uniforms.h
#pragma once

#include "slang-compiler.h"

namespace Slang
{
struct IRModule;
struct IRVarLayout;
struct IRGlobalParam;

/// Collect global-scope shader parameters that use uniform/ordinary
/// storage into a single `struct` (possibly wrapped in a constant buffer).
///
/// If requested, replace `outGlobalParams` with the resulting globals in component layout order.
/// The wrapper appears once, at the first ordinary uniform's position; unwrapped globals retain
/// their relative order. Layout fields that no longer represent global parameters are omitted.
void collectGlobalUniformParameters(
    IRModule* module,
    IRVarLayout* globalScopeVarLayout,
    CodeGenTarget target = CodeGenTarget::Unknown,
    List<IRGlobalParam*>* outGlobalParams = nullptr);

} // namespace Slang
