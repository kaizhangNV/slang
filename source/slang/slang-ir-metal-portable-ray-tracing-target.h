#pragma once

#include "core/slang-common.h"

namespace Slang
{

/// Encodes the shared native tags inferred for one Metal descriptor. These internal bit values
/// select the native tag names written by the Metal emitter.
enum class MetalRTTag : UInt
{
    Instancing = 1 << 0,
    TriangleData = 1 << 1,
    CurveData = 1 << 2,
    WorldSpaceData = 1 << 3,
    PrimitiveMotion = 1 << 4,
    InstanceMotion = 1 << 5,
    ExtendedLimits = 1 << 6,
};

/// Identifies the inputs in each stage's fixed visible-function signature. Every provider accepts
/// the contract's inputs, including values its implementation does not read.
enum class MetalRTStageRequirement : UInt
{
    None = 0,
    HitAttributes = 1 << 0,
    TriangleBarycentricCoord = 1 << 1,
    TriangleFrontFacing = 1 << 2,
    CurveParameter = 1 << 3,
    Distance = 1 << 4,
    HitKind = 1 << 5,
    WorldSpaceOrigin = 1 << 6,
    WorldSpaceDirection = 1 << 7,
    Record = 1 << 8,
    CallableDispatch = 1 << 9,
    PrimitiveIndex = 1 << 10,
    GeometryIndex = 1 << 11,
    InstanceIndex = 1 << 12,
    InstanceID = 1 << 13,
    ObjectSpaceRay = 1 << 14,
    ObjectToWorld = 1 << 15,
    WorldToObject = 1 << 16,
    InstanceCount = 1 << 17,
    InstanceIndexAtLevel = 1 << 18,
    InstanceIDAtLevel = 1 << 19,
};

/// Selects the native geometry annotation or an intersector geometry assumption. Unknown means
/// the schema contains multiple primitive kinds and the intersector must not assume one kind.
enum class MetalRTGeometryKind : UInt
{
    Unknown,
    Triangle,
    Curve,
    BoundingBox,
};

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

/// Every record begins with its function index and reserved words, followed by the application
/// record. Keep these constants shared by lowering and Metal emission.
static constexpr UInt kMetalRTRecordHeaderSize = 16;
static constexpr UInt kMetalRTRecordAlignment = 16;

/// Includes the compiler-owned header and rounds the record stride to the shared ABI alignment.
inline UInt64 getMetalRTRecordStride(UInt64 dataSize)
{
    return (kMetalRTRecordHeaderSize + dataSize + kMetalRTRecordAlignment - 1) &
           ~UInt64(kMetalRTRecordAlignment - 1);
}

} // namespace Slang
