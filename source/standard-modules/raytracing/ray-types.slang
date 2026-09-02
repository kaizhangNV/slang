implementing raytracing;

// This file defines the trace-wide choices that are independent of any particular shader group:
// the ray and traversal arguments, acceleration-structure topology, motion mode, and supported
// primitive kinds. Primitive data is exposed through storage-free properties so a stage requests
// only the native values it actually reads.

namespace rt
{

/// Identifies the motion model used by a trace program.
///
/// Select one of the motion types declared by this module as `ITraceContext.Motion`. This interface
/// is sealed because each supported motion mode has defined traversal semantics.
/// @category raytracing
[sealed]
public interface IRayMotion {}

/// Identifies a motion mode that provides a time value to eligible stage inputs.
///
/// This is a sealed classification interface. Applications select one of the concrete motion
/// types instead of implementing it directly.
/// @category raytracing
[sealed]
public interface IEnabledRayMotion : IRayMotion {}

/// Selects traversal without motion.
/// @category raytracing
public struct NoMotion : IRayMotion {}

/// Selects traversal with motion applied to primitive geometry.
/// @category raytracing
[require(metallib_2_4)]
public struct PrimitiveMotion : IEnabledRayMotion {}

/// Selects traversal with motion applied to acceleration-structure instances.
/// @category raytracing
[require(metallib_2_4)]
public struct InstanceMotion : IEnabledRayMotion {}

/// Selects traversal with motion applied to both primitive geometry and instances.
/// @category raytracing
[require(metallib_2_4)]
public struct PrimitiveAndInstanceMotion : IEnabledRayMotion {}

// Alias the core module's existing RayDesc value type so the portable trace path can call TraceRay
// without converting the ray. Lowercase properties provide the structural API spelling without
// adding storage.
/// Describes the origin, direction, and active distance interval of a ray.
///
/// This alias uses Slang's existing core-module value type. The structural API adds the lowercase
/// properties below as equivalent spellings of its fields.
/// @category raytracing
public typealias RayDesc = RayDesc;

public extension RayDesc
{
    /// The ray origin in world space.
    public property float3 origin
    {
        get { return Origin; }
        set { Origin = newValue; }
    }

    /// The minimum distance from `origin` at which traversal may accept an intersection.
    public property float tMin
    {
        get { return TMin; }
        set { TMin = newValue; }
    }

    /// The ray direction in world space.
    public property float3 direction
    {
        get { return Direction; }
        set { Direction = newValue; }
    }

    /// The maximum distance from `origin` at which traversal may accept an intersection.
    public property float tMax
    {
        get { return TMax; }
        set { TMax = newValue; }
    }
}

/// Collects the per-call arguments that control ray traversal and shader binding table (SBT)
/// record selection.
///
/// Initialize every field before passing this value to `RayTracer.trace`. The `sbtOffset`,
/// `sbtStride`, and `missIndex` values use the logical slots declared by the corresponding
/// `ITraceProgramLayout`.
/// @category raytracing
public struct RayTraversalDesc
{
    /// The ray and distance interval to trace.
    public RayDesc ray;

    /// The motion time used when the trace context selects an enabled motion mode.
    public float time;

    /// Flags that control traversal, culling, opacity, and hit-stage execution.
    public RAY_FLAG rayFlags;

    /// The visibility mask used to include or exclude acceleration-structure instances.
    public uint instanceMask;

    /// The per-trace contribution to hit-group slot selection.
    public uint sbtOffset;

    /// The geometry-index multiplier used for hit-group slot selection.
    public uint sbtStride;

    /// The logical miss-group slot selected when traversal finds no accepted hit.
    public uint missIndex;
}

/// Classifies acceleration-structure types accepted by a structural trace context.
///
/// The `RayTracer.trace` overloads below define the supported traversal operations for the
/// concrete acceleration-structure types declared by this module.
/// @category raytracing
public interface IAccelerationStructure {}

// Portable TLAS-to-BLAS / IAS-to-primitive-AS traversal.
/// Makes Slang's native ray-tracing acceleration structure available to structural tracing.
public extension RaytracingAccelerationStructure : IAccelerationStructure {}

/// An acceleration structure with the portable two-level instanced-scene topology.
///
/// Use this type when a top-level instance acceleration structure refers to bottom-level primitive
/// acceleration structures.
/// @category raytracing
public typealias AccelerationStructure = RaytracingAccelerationStructure;

// Metal traversal with an explicit maximum acceleration-structure depth.
/// An acceleration structure that permits traversal through at most `maxLevelCount` hierarchy
/// levels.
///
/// This type represents Metal's multi-level acceleration-structure topology. Select it as
/// `ITraceContext.AccelerationStructure` when the scene is not described by the portable two-level
/// topology.
/// @category raytracing
[require(metal)]
__intrinsic_type(RaytracingAccelerationStructure)
public struct MultiLevelAccelerationStructure<let maxLevelCount : int>
    : IAccelerationStructure
{}

// Built-in primitive data is exposed as a zero-storage property view. Each property maps
// independently to the corresponding target state. A default target-switch arm is a placeholder
// for targets whose structural lowering supplies the property and must not survive emission.
/// Provides built-in intersection data for a triangle candidate or committed triangle hit.
///
/// This is a storage-free view supplied through the `triangle` property of triangle hit-stage
/// inputs. Reading an individual property requests the corresponding built-in value.
/// @category raytracing
public struct TriangleData
{
    /// The barycentric weights for the triangle's second and third vertices.
    ///
    /// The weight of the first vertex is `1 - barycentricCoord.x - barycentricCoord.y`.
    [require(raytracing_anyhit_closesthit)]
    public property float2 barycentricCoord
    {
        get;
    }

    /// Whether the ray hit the triangle's front-facing side.
    [require(raytracing_anyhit_closesthit)]
    public property bool frontFacing
    {
        get
        {
            __target_switch
            {
            case hlsl:
            case glsl:
            case spirv:
            case cuda:
                return HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE;
            default:
                return false;
            }
        }
    }
}

/// Provides built-in intersection data for a curve candidate or committed curve hit.
///
/// This is a storage-free view supplied through the `curve` property of curve hit-stage inputs.
/// @category raytracing
public struct CurveData
{
    /// The parametric position of the intersection along the curve segment.
    public property float parameter
    {
        get;
    }
}

// These sealed markers keep the public primitive set closed. Applications select one of the
// public primitive types below instead of introducing an unsupported native geometry kind.
/// Identifies a primitive kind and the attributes visible after it reports an intersection.
///
/// The built-in triangle and curve kinds use storage-free attribute views. A procedural
/// bounding-box primitive uses an application-defined value transported from its intersection
/// stage to its any-hit and closest-hit stages. This interface is sealed; select a concrete
/// primitive type declared below.
/// @category raytracing
[sealed]
public interface IIntersectionPrimitive
{
    /// The intersection data visible to stages that process this primitive.
    associatedtype Attributes;

}

/// Identifies a procedural primitive whose application-defined intersection stage reports hits.
///
/// This is a sealed classification interface. Use `BoundingBoxPrimitive<CustomAttributes>` rather
/// than implementing it directly.
/// @category raytracing
[sealed]
public interface ICustomIntersectionPrimitive : IIntersectionPrimitive {}

/// Selects built-in triangle geometry for a hit context.
///
/// Triangle intersections are provided by traversal, and hit stages access their built-in data
/// through `TriangleData`.
/// @category raytracing
public struct TrianglePrimitive : IIntersectionPrimitive
{
    /// The built-in data view provided for triangle hits.
    public typealias Attributes = TriangleData;
}

/// Selects built-in curve geometry for a hit context.
///
/// Curve intersections are provided by Metal traversal, and hit stages access their built-in data
/// through `CurveData`.
/// @category raytracing
[require(metallib_3_1)]
public struct CurvePrimitive : IIntersectionPrimitive
{
    /// The built-in data view provided for curve hits.
    public typealias Attributes = CurveData;
}

/// Selects procedural axis-aligned bounding-box geometry with custom intersection attributes.
///
/// A corresponding `IIntersectionShader` reports candidate distances and `CustomAttributes`
/// values. Accepted attributes are then visible to the hit group's any-hit and closest-hit stages.
/// @category raytracing
public struct BoundingBoxPrimitive<CustomAttributes> : ICustomIntersectionPrimitive
{
    /// The application-defined data reported for each candidate intersection.
    public typealias Attributes = CustomAttributes;
}

} // namespace rt
