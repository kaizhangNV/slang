#pragma once

namespace Slang
{
struct IRModule;
struct IRInst;
struct IRVarLayout;
class DiagnosticSink;

/// Returns whether a resource can be supplied directly by the Metal intersection driver.
bool isMetalPortableRayTracingBindableResource(IRInst* resource);

/// Keeps component resources in a consistent order for separately compiled RT shaders.
/// Runs immediately after global uniform collection, before entry uniforms and optimization.
void preserveMetalRTGlobalResources(IRModule* module, IRVarLayout* globalScopeVarLayout);

/// Validates traversal callback bindings after explicit global-context lowering.
void finalizeMetalPortableRayTracingGlobalContext(IRModule* module, DiagnosticSink* sink);
} // namespace Slang
