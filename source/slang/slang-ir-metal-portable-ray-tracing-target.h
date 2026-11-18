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

/// Defines the stable argument-buffer IDs of a physical TraceProgramDescriptor. Each field
/// occupies eight bytes; CandidateResources is an inline struct containing one AnyHit VFT.
/// Unused fields stay present, keeping the descriptor size at 48 bytes for every schema.
enum class MetalRTDescriptorResource : UInt
{
    Tables = 0,
    MissVisibleFunctionTable = 1,
    ClosestHitVisibleFunctionTable = 2,
    CallableVisibleFunctionTable = 3,
    Records = 4,
    CandidateResources = 5,
    Count,
};

/// Defines the header words containing byte offsets into the host-built records buffer.
/// Instance contributions belong to the acceleration structure, not this program's records.
enum class MetalRTRecordSection : UInt
{
    Hit = 0,
    Miss = 1,
    Callable = 2,
};

/// Hit records start with closest-hit and procedural-any-hit VFT indices, then two reserved
/// words. Miss/callable records start with their VFT index and three reserved words. Application
/// record bytes follow this header; absent hit providers use 0xffffffff.
static constexpr UInt kMetalRTRecordHeaderSize = 16;
static constexpr UInt kMetalRTRecordAlignment = 16;

/// Includes the compiler-owned header and rounds the record stride to the shared ABI alignment.
inline UInt64 getMetalRTRecordStride(UInt64 dataSize)
{
    return (kMetalRTRecordHeaderSize + dataSize + kMetalRTRecordAlignment - 1) &
           ~UInt64(kMetalRTRecordAlignment - 1);
}

} // namespace Slang
