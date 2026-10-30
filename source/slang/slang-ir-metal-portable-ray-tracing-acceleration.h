#pragma once

#include "slang-ir-metal-portable-ray-tracing-target.h"
#include "slang-ir-portable-ray-tracing-schema.h"

namespace Slang
{
class TargetRequest;
class DiagnosticSink;

namespace PortableRT
{
namespace Metal
{

/// Describes the normalized trace-wide topology and motion signature. Optional stage-data tags
/// are added by stage requirement collection; maxLevels is zero for ordinary instancing/N=1.
struct TraceContextRequirements
{
    UInt tags = UInt(MetalRTTag::Instancing);
    IRIntegerValue maxLevels = 0;
};

/// Validates the selected schema's target-dependent topology and motion contract once. Both
/// descriptor construction and resource physicalization must consume this same result.
bool getTraceContextRequirements(
    const Schema& schema,
    TargetRequest* target,
    IRInst* useSite,
    DiagnosticSink* sink,
    TraceContextRequirements& result);

/// Associates one trace use with its required resource signature. Separate values of the same
/// source AS type can have different motion signatures; one shared value cannot.
struct AccelerationStructureUse
{
    IRInst* value;
    UInt tags;
    IRInst* useSite;
};

/// Rewrites the complete SSA/storage component of each used acceleration structure, including
/// local storage, selected fields, array containers, function parameters/results, and joins.
bool physicalizeAccelerationStructures(
    IRModule* module,
    const List<AccelerationStructureUse>& uses,
    DiagnosticSink* sink);

} // namespace Metal
} // namespace PortableRT
} // namespace Slang
