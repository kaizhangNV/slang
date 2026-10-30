#pragma once

#include "slang-ir-metal-portable-ray-tracing-resources.h"
#include "slang-ir-metal-portable-ray-tracing-stages.h"

namespace Slang
{
class DiagnosticSink;

namespace PortableRT
{
namespace Metal
{

/// Returns schema-owned native IFT glue. It never references a source stage implementation.
struct CandidateAdapters
{
    Dictionary<MetalRTCandidateKind, IRFunc*> intersectionFunctions;
};

/// Builds the fixed signatures shared by independently compiled candidate stages and their tables.
IRFuncType* getAnyHitFunctionSignature(IRBuilder& builder);
IRFuncType* getIntersectionFunctionSignature(IRBuilder& builder);

/// Compiles only the selected candidate stage. An intersection adapter receives its any-hit table
/// and index from schema glue, so the same compiled shader can participate in different groups.
VisibleAdapter createSelectedCandidateAdapter(
    IRModule* module,
    const SchemaStage& stage,
    UnownedStringSlice symbolName,
    bool& requiresResultTypeLowering,
    DiagnosticSink* sink);

/// Creates body-free candidate dispatch for a descriptor. Runtime records choose a schema group;
/// its checked types determine storage, while independently supplied VFT handles select code.
CandidateAdapters createCandidateAdapters(
    IRModule* module,
    const Schema& schema,
    const MetalRTDescriptorResources& resources,
    const Dictionary<IRType*, RayData>& carriers,
    const RayDataHeader& header,
    UInt tags,
    IRIntegerValue maxLevels,
    IRIntegerValue hitStride,
    IRFunc* instanceContribution,
    IRType* instancePathParamType,
    UnownedStringSlice symbolPrefix);

} // namespace Metal
} // namespace PortableRT
} // namespace Slang
