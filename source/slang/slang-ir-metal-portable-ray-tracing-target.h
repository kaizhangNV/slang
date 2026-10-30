#pragma once

#include "core/slang-common.h"
#include "slang.h"

namespace Slang
{

/// Encodes the shared native tags inferred for one Metal descriptor. These internal bit values
/// are not Apple's MTLIntersectionFunctionSignature values; reflection must translate explicitly.
using MetalRTTag = slang::MetalRayTracingTag;

/// Identifies the inputs in each stage's fixed visible-function signature. Every provider accepts
/// the contract's inputs, including values its implementation does not read.
using MetalRTStageRequirement = slang::MetalRayTracingStageRequirement;

/// Selects the native geometry annotation or an intersector geometry assumption. Unknown means
/// the schema contains multiple primitive kinds and the intersector must not assume one kind.
using MetalRTGeometryKind = slang::MetalRayTracingPrimitive;

/// Defines the stable argument-buffer IDs of a physical TraceProgramDescriptor. Unused tables
/// retain their field IDs so adding a stage does not move the record buffer or other resources.
enum class MetalRTDescriptorResource : UInt
{
    IntersectionFunctionTable = 0,
    MissVisibleFunctionTable = 1,
    ClosestHitVisibleFunctionTable = 2,
    CallableVisibleFunctionTable = 3,
    Records = 4,
    AnyHitVisibleFunctionTable = 5,
    IntersectionVisibleFunctionTable = 6,
    Count,
};

/// Defines the header words containing byte offsets into the host-built records buffer.
/// Ordinary instancing uses a flat contribution array; multilevel traversal uses the same
/// section as an instance-path trie. Neither representation reuses user instance IDs.
enum class MetalRTRecordSection : UInt
{
    InstanceHitGroupContribution = 0,
    Hit = 1,
    Miss = 2,
    Callable = 3,
};

/// Selects fixed IFT slots by primitive kind before the generated dispatcher reads an SBT record.
enum class MetalRTCandidateKind : UInt
{
    Triangle = 0,
    BoundingBox = 1,
    Curve = 2,
};

/// Every record begins with its reflected function index and reserved words, followed by the
/// application record. Keep these constants shared by reflection, lowering, and Metal emission.
static constexpr UInt kMetalRTRecordHeaderSize = 16;
static constexpr UInt kMetalRTRecordAlignment = 16;

/// Includes the compiler-owned header and rounds the record stride to the shared ABI alignment.
inline UInt64 getMetalRTRecordStride(UInt64 dataSize)
{
    return (kMetalRTRecordHeaderSize + dataSize + kMetalRTRecordAlignment - 1) &
           ~UInt64(kMetalRTRecordAlignment - 1);
}

} // namespace Slang
