#pragma once

namespace Slang
{
struct IRModule;
struct IRInst;
struct IRVarLayout;
class DiagnosticSink;

/// Returns whether a resource can be supplied directly by the Metal intersection driver.
bool isMetalPortableRayTracingBindableResource(IRInst* resource);

/// Preserves the linked component's declared global resources before selected-shader DCE.
void preserveMetalPortableRayTracingGlobalParameters(
    IRModule* module,
    IRVarLayout* globalScopeVarLayout);

/// Validates traversal callback bindings after explicit global-context lowering.
void finalizeMetalPortableRayTracingGlobalContext(IRModule* module, DiagnosticSink* sink);
} // namespace Slang
