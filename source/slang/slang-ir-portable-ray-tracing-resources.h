#pragma once

namespace Slang
{
struct IRModule;

/// Removes descriptor storage on native SBT targets after schema consumers have used its type.
/// The native pipeline supplies the SBT; ordinary void-type cleanup removes erased fields and
/// helper arguments without giving the descriptor a dummy resource binding.
void eraseNativeRayTracingDescriptors(IRModule* module);
} // namespace Slang
