implementing raytracing;

// Stage-input structs present ray-tracing system values as typed, storage-free views. A context
// determines the payload, record, topology, motion, and primitive-specific properties that are
// legal for a stage. Properties in the base structs are generally available for that stage;
// constrained extensions add values that exist only for a selected motion, topology, or primitive.

namespace rt
{

// Keep each native system-value lookup in one helper so every stage-input property uses the same
// target mapping. Force-inlining preserves the original property behavior: the native intrinsic
// remains at the property-use site, and its capability requirement is propagated to that caller.

/// Returns the minimum valid distance for the current ray.
[ForceInline]
internal float getStageInputMinDistance()
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case spirv:
    case cuda:
        return RayTMin();
    default:
        return 0.0;
    }
}

/// Returns the current hit distance or traversal upper bound, depending on the stage.
[ForceInline]
internal float getStageInputCurrentDistance()
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case spirv:
    case cuda:
        return RayTCurrent();
    default:
        return 0.0;
    }
}

/// Returns the classification of the current candidate or committed hit.
[ForceInline]
internal uint getStageInputHitKind()
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case spirv:
    case cuda:
        return HitKind();
    default:
        return 0;
    }
}

/// Returns the flags supplied to the current trace operation.
[ForceInline]
internal uint getStageInputRayFlags()
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case spirv:
    case cuda:
        return RayFlags();
    default:
        return 0;
    }
}

/// Returns the index of the current primitive within its geometry.
[ForceInline]
internal uint getStageInputPrimitiveIndex()
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case spirv:
    case cuda:
        return PrimitiveIndex();
    default:
        return 0;
    }
}

/// Returns the index of the current geometry within its primitive acceleration structure.
[ForceInline]
internal uint getStageInputGeometryIndex()
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case spirv:
        return GeometryIndex();
    default:
        return 0;
    }
}

/// Returns the origin of the current ray in world space.
[ForceInline]
internal float3 getStageInputWorldSpaceOrigin()
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case spirv:
    case cuda:
        return WorldRayOrigin();
    default:
        return 0.0;
    }
}

/// Returns the direction of the current ray in world space.
[ForceInline]
internal float3 getStageInputWorldSpaceDirection()
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case spirv:
    case cuda:
        return WorldRayDirection();
    default:
        return 0.0;
    }
}

/// Returns the three-dimensional launch index of the current ray dispatch.
[ForceInline]
internal uint3 getStageInputDispatchRaysIndex()
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case spirv:
    case cuda:
        return DispatchRaysIndex();
    default:
        return 0;
    }
}

/// Returns the dimensions of the current ray dispatch.
[ForceInline]
internal uint3 getStageInputDispatchRaysDimensions()
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case spirv:
    case cuda:
        return DispatchRaysDimensions();
    default:
        return 0;
    }
}

/// Returns the motion-interpolation time of the current ray.
[ForceInline]
internal float getStageInputRayTime()
{
    __target_switch
    {
    case glsl:
    case spirv:
        return RayCurrentTime();
    default:
        return 0.0;
    }
}

/// Returns the index of the current instance in its acceleration structure.
[ForceInline]
internal uint getStageInputInstanceIndex()
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case spirv:
    case cuda:
        return InstanceIndex();
    default:
        return 0;
    }
}

/// Returns the application-provided identifier of the current instance.
[ForceInline]
internal uint getStageInputInstanceID()
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case spirv:
    case cuda:
        return InstanceID();
    default:
        return 0;
    }
}

/// Returns the current ray transformed into object space.
[ForceInline]
internal RayDesc getStageInputObjectSpaceRay()
{
    RayDesc result = {};
    __target_switch
    {
    case hlsl:
    case glsl:
    case spirv:
    case cuda:
        result.Origin = ObjectRayOrigin();
        result.TMin = RayTMin();
        result.Direction = ObjectRayDirection();
        result.TMax = RayTCurrent();
    default:
        break;
    }
    return result;
}

/// Returns the transform from the current instance's object space to world space.
[ForceInline]
internal float4x3 getStageInputObjectToWorld()
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case spirv:
    case cuda:
        return ObjectToWorld4x3();
    default:
        return 0.0;
    }
}

/// Returns the transform from world space to the current instance's object space.
[ForceInline]
internal float4x3 getStageInputWorldToObject()
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case spirv:
    case cuda:
        return WorldToObject4x3();
    default:
        return 0.0;
    }
}

// These types must remain storage-free. The compiler maps an accessed property to the corresponding
// entry-point parameter or system value instead of constructing an input object in memory. A
// default target-switch arm is only a well-formed source placeholder for a property that target
// lowering must replace before emission; it is not the property's runtime value.
/// Provides the values available while processing the closest accepted hit for `Context`.
///
/// The compiler supplies this type as the value parameter of `IClosestHitShader.invoke`. It may be
/// passed by value to helpers that are valid for the closest-hit stage, but it cannot be
/// constructed, stored, returned, or passed by reference.
/// @category raytracing
[require(structural_raytracing_closesthit)]
public struct ClosestHitInput<Context>
    where Context : IHitContext
{
    /// A mutable reference to the ray payload shared with the trace caller.
    public property Context.TraceContext.Payload payload
    {
        ref;
    }

    /// The read-only local shader-record data for the selected hit-group slot.
    public property Context.Record record
    {
        get;
    }

    /// The minimum valid intersection distance for the current ray.
    public property float minDistance
    {
        get { return getStageInputMinDistance(); }
    }

    /// The distance along the ray to the committed closest hit.
    public property float distance
    {
        get { return getStageInputCurrentDistance(); }
    }

    /// The hit classification reported by traversal or by the procedural intersection stage.
    public property uint hitKind
    {
        get { return getStageInputHitKind(); }
    }

    /// The `RAY_FLAG` values used for the current trace.
    public property uint rayFlags
    {
        get { return getStageInputRayFlags(); }
    }

    /// The zero-based index of the intersected primitive within its geometry.
    public property uint primitiveIndex
    {
        get { return getStageInputPrimitiveIndex(); }
    }

    /// The zero-based index of the intersected geometry within its primitive acceleration
    /// structure.
    public property uint geometryIndex
    {
        get { return getStageInputGeometryIndex(); }
    }

    /// The origin of the current ray in world space.
    public property float3 worldSpaceOrigin
    {
        get { return getStageInputWorldSpaceOrigin(); }
    }

    /// The direction of the current ray in world space.
    public property float3 worldSpaceDirection
    {
        get { return getStageInputWorldSpaceDirection(); }
    }

    /// The three-dimensional launch index of the ray-generation invocation.
    public property uint3 dispatchRaysIndex
    {
        get { return getStageInputDispatchRaysIndex(); }
    }

    /// The dimensions of the ray dispatch that launched the current trace.
    public property uint3 dispatchRaysDimensions
    {
        get { return getStageInputDispatchRaysDimensions(); }
    }
}

public extension<Context> ClosestHitInput<Context>
    where Context : IHitContext
    where Context.TraceContext.Motion : IEnabledRayMotion
{
    /// The motion-interpolation time of the current ray.
    public property float time
    {
        get { return getStageInputRayTime(); }
    }
}

public extension<Context> ClosestHitInput<Context>
    where Context : IHitContext
    where Context.TraceContext.AccelerationStructure == AccelerationStructure
{
    /// The zero-based index of the current instance in its acceleration structure.
    public property uint instanceIndex
    {
        get { return getStageInputInstanceIndex(); }
    }

    /// The application-provided identifier of the current instance.
    public property uint instanceID
    {
        get { return getStageInputInstanceID(); }
    }

    /// The current ray transformed into the object space of the hit instance.
    ///
    /// Its maximum distance is the committed hit distance.
    public property RayDesc objectSpaceRay
    {
        get { return getStageInputObjectSpaceRay(); }
    }

    /// The transform from the current instance's object space to world space.
    public property float4x3 objectToWorld
    {
        get { return getStageInputObjectToWorld(); }
    }

    /// The transform from world space to the current instance's object space.
    public property float4x3 worldToObject
    {
        get { return getStageInputWorldToObject(); }
    }
}

public extension<Context> ClosestHitInput<Context>
    where Context : IHitContext
    where Context.Primitive : ICustomIntersectionPrimitive
{
    /// The custom attributes reported for the committed procedural intersection.
    public property Context.Primitive.Attributes attributes
    {
        get;
    }
}

public extension<Context> ClosestHitInput<Context>
    where Context : IHitContext
    where Context.Primitive == TrianglePrimitive
{
    /// The built-in data for the committed triangle intersection.
    public property TriangleData triangle
    {
        get { return {}; }
    }
}

public extension<Context> ClosestHitInput<Context>
    where Context : IHitContext
    where Context.Primitive == CurvePrimitive
{
    /// The built-in data for the committed curve intersection.
    public property CurveData curve
    {
        get { return {}; }
    }
}

/// Provides the values and traversal controls available while filtering a candidate hit for
/// `Context`.
///
/// The compiler supplies this type as the value parameter of `IAnyHitShader.invoke`. It may be
/// passed by value to helpers that are valid for the any-hit stage, but it cannot be constructed,
/// stored, returned, or passed by reference. Calling `ignoreHit` rejects the candidate, while
/// `acceptHitAndEndSearch` accepts it and terminates traversal. Returning normally accepts the
/// candidate and permits traversal to continue.
/// @category raytracing
[require(structural_raytracing_anyhit)]
public struct AnyHitInput<Context>
    where Context : IHitContext
{
    /// A mutable reference to the ray payload shared with the trace caller.
    public property Context.TraceContext.Payload payload
    {
        ref;
    }

    /// The read-only local shader-record data for the selected hit-group slot.
    public property Context.Record record
    {
        get;
    }

    /// The minimum valid intersection distance for the current ray.
    public property float minDistance
    {
        get { return getStageInputMinDistance(); }
    }

    /// The distance along the ray to the candidate intersection.
    public property float distance
    {
        get { return getStageInputCurrentDistance(); }
    }

    /// The hit classification reported by traversal or by the procedural intersection stage.
    public property uint hitKind
    {
        get { return getStageInputHitKind(); }
    }

    /// The `RAY_FLAG` values used for the current trace.
    public property uint rayFlags
    {
        get { return getStageInputRayFlags(); }
    }

    /// The zero-based index of the candidate primitive within its geometry.
    public property uint primitiveIndex
    {
        get { return getStageInputPrimitiveIndex(); }
    }

    /// The zero-based index of the candidate geometry within its primitive acceleration structure.
    public property uint geometryIndex
    {
        get { return getStageInputGeometryIndex(); }
    }

    /// The origin of the current ray in world space.
    public property float3 worldSpaceOrigin
    {
        get { return getStageInputWorldSpaceOrigin(); }
    }

    /// The direction of the current ray in world space.
    public property float3 worldSpaceDirection
    {
        get { return getStageInputWorldSpaceDirection(); }
    }

    /// The three-dimensional launch index of the ray-generation invocation.
    public property uint3 dispatchRaysIndex
    {
        get { return getStageInputDispatchRaysIndex(); }
    }

    /// The dimensions of the ray dispatch that launched the current trace.
    public property uint3 dispatchRaysDimensions
    {
        get { return getStageInputDispatchRaysDimensions(); }
    }

    /// Rejects the current candidate intersection and continues traversal.
    public void ignoreHit()
    {
        __target_switch
        {
        case hlsl:
        case glsl:
        case spirv:
        case cuda:
            IgnoreHit();
        default:
            break;
        }
    }

    /// Accepts the current candidate intersection and terminates traversal immediately.
    public void acceptHitAndEndSearch()
    {
        __target_switch
        {
        case hlsl:
        case glsl:
        case spirv:
        case cuda:
            AcceptHitAndEndSearch();
        default:
            break;
        }
    }
}

public extension<Context> AnyHitInput<Context>
    where Context : IHitContext
    where Context.TraceContext.Motion : IEnabledRayMotion
{
    /// The motion-interpolation time of the current ray.
    public property float time
    {
        get { return getStageInputRayTime(); }
    }
}

public extension<Context> AnyHitInput<Context>
    where Context : IHitContext
    where Context.TraceContext.AccelerationStructure == AccelerationStructure
{
    /// The zero-based index of the current instance in its acceleration structure.
    public property uint instanceIndex
    {
        get { return getStageInputInstanceIndex(); }
    }

    /// The application-provided identifier of the current instance.
    public property uint instanceID
    {
        get { return getStageInputInstanceID(); }
    }

    /// The current ray transformed into the object space of the candidate instance.
    ///
    /// Its maximum distance is the candidate intersection distance.
    public property RayDesc objectSpaceRay
    {
        get { return getStageInputObjectSpaceRay(); }
    }

    /// The transform from the current instance's object space to world space.
    public property float4x3 objectToWorld
    {
        get { return getStageInputObjectToWorld(); }
    }

    /// The transform from world space to the current instance's object space.
    public property float4x3 worldToObject
    {
        get { return getStageInputWorldToObject(); }
    }
}

public extension<Context> AnyHitInput<Context>
    where Context : IHitContext
    where Context.Primitive : ICustomIntersectionPrimitive
{
    /// The custom attributes reported for the candidate procedural intersection.
    public property Context.Primitive.Attributes attributes
    {
        get;
    }
}

public extension<Context> AnyHitInput<Context>
    where Context : IHitContext
    where Context.Primitive == TrianglePrimitive
{
    /// The built-in data for the candidate triangle intersection.
    public property TriangleData triangle
    {
        get { return {}; }
    }
}

public extension<Context> AnyHitInput<Context>
    where Context : IHitContext
    where Context.Primitive == CurvePrimitive
{
    /// The built-in data for the candidate curve intersection.
    public property CurveData curve
    {
        get { return {}; }
    }
}

/// Provides the values and reporting operation available while testing a procedural primitive for
/// `Context`.
///
/// The compiler supplies this type as the value parameter of `IIntersectionShader.invoke`. It may
/// be passed by value to helpers that are valid for the intersection stage, but it cannot be
/// constructed, stored, returned, or passed by reference. The shader may test and report multiple
/// candidate intersections; traversal determines which accepted candidate eventually becomes the
/// closest hit.
/// @category raytracing
[require(structural_raytracing_intersection)]
public struct IntersectionInput<Context>
    where Context : IHitContext
{
    /// The read-only local shader-record data for the selected hit-group slot.
    public property Context.Record record
    {
        get;
    }

    /// The minimum valid intersection distance for the current ray.
    public property float minDistance
    {
        get { return getStageInputMinDistance(); }
    }

    /// The `RAY_FLAG` values used for the current trace.
    public property uint rayFlags
    {
        get { return getStageInputRayFlags(); }
    }

    /// The current ray transformed into the object space of the tested instance.
    ///
    /// Its maximum distance is the current traversal upper bound. A reported candidate outside the
    /// ray's active interval is not accepted.
    public property RayDesc objectSpaceRay
    {
        get { return getStageInputObjectSpaceRay(); }
    }

    /// The zero-based index of the tested primitive within its geometry.
    public property uint primitiveIndex
    {
        get { return getStageInputPrimitiveIndex(); }
    }

    /// The zero-based index of the tested geometry within its primitive acceleration structure.
    public property uint geometryIndex
    {
        get { return getStageInputGeometryIndex(); }
    }

    /// The origin of the current ray in world space.
    public property float3 worldSpaceOrigin
    {
        get { return getStageInputWorldSpaceOrigin(); }
    }

    /// The direction of the current ray in world space.
    public property float3 worldSpaceDirection
    {
        get { return getStageInputWorldSpaceDirection(); }
    }

    /// The three-dimensional launch index of the ray-generation invocation.
    public property uint3 dispatchRaysIndex
    {
        get { return getStageInputDispatchRaysIndex(); }
    }

    /// The dimensions of the ray dispatch that launched the current trace.
    public property uint3 dispatchRaysDimensions
    {
        get { return getStageInputDispatchRaysDimensions(); }
    }

    /// Reports a candidate intersection with hit kind zero.
    /// @param distance The candidate distance along the current ray.
    /// @param attributes The primitive's custom data to provide to any-hit and closest-hit stages.
    /// @return `true` if traversal accepted the candidate, or `false` if it was outside the active
    /// ray interval or rejected by any-hit processing.
    public bool reportHit(float distance, Context.Primitive.Attributes attributes)
    {
        __target_switch
        {
        case hlsl:
        case glsl:
        case spirv:
            return ReportHit(distance, 0, attributes);
        default:
            return false;
        }
    }

    /// Reports a candidate intersection with an application-defined hit classification.
    /// @param distance The candidate distance along the current ray.
    /// @param hitKind The classification exposed to subsequent any-hit and closest-hit stages.
    /// @param attributes The primitive's custom data to provide to any-hit and closest-hit stages.
    /// @return `true` if traversal accepted the candidate, or `false` if it was outside the active
    /// ray interval or rejected by any-hit processing.
    public bool reportHit(
        float distance,
        uint hitKind,
        Context.Primitive.Attributes attributes)
    {
        __target_switch
        {
        case hlsl:
        case glsl:
        case spirv:
            return ReportHit(distance, hitKind, attributes);
        default:
            return false;
        }
    }
}

public extension<Context> IntersectionInput<Context>
    where Context : IHitContext
    where Context.TraceContext.Motion : IEnabledRayMotion
{
    /// The motion-interpolation time of the current ray.
    public property float time
    {
        get { return getStageInputRayTime(); }
    }
}

public extension<Context> IntersectionInput<Context>
    where Context : IHitContext
    where Context.TraceContext.AccelerationStructure == AccelerationStructure
{
    /// The zero-based index of the current instance in its acceleration structure.
    public property uint instanceIndex
    {
        get { return getStageInputInstanceIndex(); }
    }

    /// The application-provided identifier of the current instance.
    public property uint instanceID
    {
        get { return getStageInputInstanceID(); }
    }

    /// The transform from the current instance's object space to world space.
    public property float4x3 objectToWorld
    {
        get { return getStageInputObjectToWorld(); }
    }

    /// The transform from world space to the current instance's object space.
    public property float4x3 worldToObject
    {
        get { return getStageInputWorldToObject(); }
    }
}

/// Provides the values available when a ray completes without an accepted hit for `Context`.
///
/// The compiler supplies this type as the value parameter of `IMissShader.invoke`. It may be passed
/// by value to helpers that are valid for the miss stage, but it cannot be constructed, stored,
/// returned, or passed by reference.
/// @category raytracing
[require(structural_raytracing_miss)]
public struct MissInput<Context>
    where Context : IMissGroupContext
{
    /// A mutable reference to the ray payload shared with the trace caller.
    public property Context.TraceContext.Payload payload
    {
        ref;
    }

    /// The read-only local shader-record data for the selected miss-group slot.
    public property Context.Record record
    {
        get;
    }

    /// The minimum valid intersection distance for the current ray.
    public property float minDistance
    {
        get { return getStageInputMinDistance(); }
    }

    /// The maximum distance of the ray that completed without a hit.
    public property float distance
    {
        get { return getStageInputCurrentDistance(); }
    }

    /// The `RAY_FLAG` values used for the current trace.
    public property uint rayFlags
    {
        get { return getStageInputRayFlags(); }
    }

    /// The origin of the current ray in world space.
    public property float3 worldSpaceOrigin
    {
        get { return getStageInputWorldSpaceOrigin(); }
    }

    /// The direction of the current ray in world space.
    public property float3 worldSpaceDirection
    {
        get { return getStageInputWorldSpaceDirection(); }
    }

    /// The three-dimensional launch index of the ray-generation invocation.
    public property uint3 dispatchRaysIndex
    {
        get { return getStageInputDispatchRaysIndex(); }
    }

    /// The dimensions of the ray dispatch that launched the current trace.
    public property uint3 dispatchRaysDimensions
    {
        get { return getStageInputDispatchRaysDimensions(); }
    }
}

public extension<Context> MissInput<Context>
    where Context : IMissGroupContext
    where Context.TraceContext.Motion : IEnabledRayMotion
{
    /// The motion-interpolation time of the current ray.
    public property float time
    {
        get { return getStageInputRayTime(); }
    }
}

/// Provides the data and dispatch coordinates available to a callable shader for `Context`.
///
/// The compiler supplies this type as the value parameter of `ICallableShader.invoke`. It may be
/// passed by value to helpers that are valid for the callable stage, but it cannot be constructed,
/// stored, returned, or passed by reference.
/// @category raytracing
[require(structural_raytracing_callable)]
public struct CallableInput<Context>
    where Context : ICallableGroupContext
{
    /// A mutable reference to the application-defined value exchanged with the caller.
    public property Context.CallableData data
    {
        ref;
    }

    /// The read-only local shader-record data for the selected callable-group slot.
    public property Context.Record record
    {
        get;
    }

    /// The three-dimensional launch index of the ray-generation invocation.
    public property uint3 dispatchRaysIndex
    {
        get { return getStageInputDispatchRaysIndex(); }
    }

    /// The dimensions of the ray dispatch that launched the current trace.
    public property uint3 dispatchRaysDimensions
    {
        get { return getStageInputDispatchRaysDimensions(); }
    }
}

} // namespace rt
