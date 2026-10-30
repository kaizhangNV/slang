#pragma once

namespace Slang
{
struct IRModule;
class DiagnosticSink;

/// Finishes the physical Metal table ABI after ordinary explicit-global-context lowering.
/// Visible functions share their caller's context; traversal callbacks receive bindable buffers.
void finalizeMetalPortableRayTracingGlobalContext(IRModule* module, DiagnosticSink* sink);
} // namespace Slang
