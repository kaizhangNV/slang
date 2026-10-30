#pragma once

#include "slang-ir-metal-portable-ray-tracing-stages.h"

namespace Slang
{
class DiagnosticSink;

namespace PortableRT
{
namespace Metal
{

/// Returns the fixed native IFT functions and their ordinary group callbacks. The callbacks own
/// invocation values; combine them with visible adapters before threading immutable inputs.
struct CandidateAdapters
{
    Dictionary<MetalRTCandidateKind, IRFunc*> intersectionFunctions;
    List<VisibleAdapter> invocationRoots;
    bool producesResultTypes = false;
};

/// Creates candidate dispatch for a descriptor. Runtime records choose a schema group; that
/// group's callback runs its AnyHit/Intersection methods without changing their source identities.
/// The caller must create every group's carrier and include rayFlags for procedural traversal.
CandidateAdapters createCandidateAdapters(
    IRModule* module,
    const Schema& schema,
    const Dictionary<IRType*, RayData>& carriers,
    const RayDataHeader& header,
    const StageRequirements& requirements,
    UInt tags,
    IRIntegerValue maxLevels,
    IRIntegerValue hitStride,
    IRFunc* instanceContribution,
    IRType* instancePathParamType,
    UnownedStringSlice symbolPrefix,
    DiagnosticSink* sink);

} // namespace Metal
} // namespace PortableRT
} // namespace Slang
