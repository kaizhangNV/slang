#pragma once

#include "core/slang-list.h"

namespace Slang
{
struct IRModule;
struct IRInst;
struct IRGlobalParam;
class DiagnosticSink;

/// Returns whether a resource can be supplied directly by the Metal intersection driver.
bool isMetalPortableRayTracingBindableResource(IRInst* resource);

/// Keeps global resource declarations and their layout order for separately compiled RT shaders.
/// Takes the resulting globals returned by collectGlobalUniformParameters, before optimization.
void preserveMetalRTGlobalResources(IRModule* module, const List<IRGlobalParam*>& globalParams);

/// Validates traversal callback bindings after explicit global-context lowering.
void finalizeMetalPortableRayTracingGlobalContext(IRModule* module, DiagnosticSink* sink);
} // namespace Slang
