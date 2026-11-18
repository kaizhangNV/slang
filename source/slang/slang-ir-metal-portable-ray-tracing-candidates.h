#pragma once

#include "slang-ir-metal-portable-ray-tracing-resources.h"
#include "slang-ir-metal-portable-ray-tracing-stages.h"

namespace Slang
{
class DiagnosticSink;
class TargetRequest;

namespace PortableRT
{
namespace Metal
{

/// Builds the fixed signature used by procedural AnyHit providers and their VFT.
IRFuncType* getAnyHitFunctionSignature(IRBuilder& builder);

/// Compiles the selected candidate stage using its checked primitive and trace context.
/// Triangle/curve AnyHit and procedural Intersection become native intersection callbacks;
/// procedural AnyHit remains a visible provider called by ReportHit.
VisibleAdapter createSelectedCandidateAdapter(
    IRModule* module,
    TargetRequest* target,
    const SchemaStage& stage,
    const RayDataHeader& header,
    const MetalRTCandidateResources& candidateResources,
    UnownedStringSlice symbolName,
    bool& requiresResultTypeLowering,
    DiagnosticSink* sink);

} // namespace Metal
} // namespace PortableRT
} // namespace Slang
