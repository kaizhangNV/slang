implementing portable;

namespace rt
{

// Stage inputs are compiler-provided views of the current invocation, with no source fields.
// For example, `let local = input` copies the view and both names access the same invocation state.
// Copying a view does not duplicate its payload, callable-data, or shader-record storage. An `in`
// helper parameter receives the same view; reading `record` still returns application data by value.
// Native targets expose much of that state through existing core-module intrinsics. The helpers
// remain force-inlined so capability and stage checks apply at the shader's property access.
//
// Metal obtains the same values from generated intersection or post-traversal dispatch adapters.
// These internal operations describe reachable requirements without prescribing adapter parameters.
// A source helper can remain an ordinary, non-inlined function: lowering passes the required values
// from its adapter through its callers. Candidate control operations are separate because they
// affect the whole source stage, rather than merely returning from the helper that uses them.

/// Reads the originating ray's lower interval bound from the Metal invocation.
[require(metal)]
__intrinsic_op(metalRTGetMinDistance)
internal float __metalRTGetMinDistance();

/// Reads the Metal candidate or committed distance, or the miss-stage upper interval bound.
[require(metal)]
__intrinsic_op(metalRTGetCurrentDistance)
internal float __metalRTGetCurrentDistance();

/// Reads the candidate or committed hit classification from the Metal invocation.
[require(metal)]
__intrinsic_op(metalRTGetHitKind)
internal uint __metalRTGetHitKind();

/// Reads the flags of the trace call which created the Metal invocation.
[require(metal)]
__intrinsic_op(metalRTGetRayFlags)
internal uint __metalRTGetRayFlags();

/// Reads the geometry-local primitive index from the Metal invocation.
[require(metal)]
__intrinsic_op(metalRTGetPrimitiveIndex)
internal uint __metalRTGetPrimitiveIndex();

/// Reads the primitive acceleration structure's geometry index from the Metal invocation.
[require(metal)]
__intrinsic_op(metalRTGetGeometryIndex)
internal uint __metalRTGetGeometryIndex();

/// Reads the originating ray's world-space origin from the Metal invocation.
[require(metal)]
__intrinsic_op(metalRTGetWorldSpaceOrigin)
internal float3 __metalRTGetWorldSpaceOrigin();

/// Reads the originating ray's world-space direction from the Metal invocation.
[require(metal)]
__intrinsic_op(metalRTGetWorldSpaceDirection)
internal float3 __metalRTGetWorldSpaceDirection();

/// Reads the originating launch coordinates carried by the Metal invocation.
[require(metal)]
__intrinsic_op(metalRTGetDispatchRaysIndex)
internal uint3 __metalRTGetDispatchRaysIndex();

/// Reads the originating launch dimensions carried by the Metal invocation.
[require(metal)]
__intrinsic_op(metalRTGetDispatchRaysDimensions)
internal uint3 __metalRTGetDispatchRaysDimensions();

/// Reads the motion-sampling time carried by the Metal invocation.
[require(metal)]
__intrinsic_op(metalRTGetRayTime)
internal float __metalRTGetRayTime();

/// Reads the Metal invocation's object-space ray with its current interval.
[require(metal)]
__intrinsic_op(metalRTGetObjectSpaceRay)
internal RayDesc __metalRTGetObjectSpaceRay();

/// Reads the complete object-to-world transform, or identity without instancing.
[require(metal)]
__intrinsic_op(metalRTGetObjectToWorld)
internal float4x3 __metalRTGetObjectToWorld();

/// Reads the complete world-to-object transform, or identity without instancing.
[require(metal)]
__intrinsic_op(metalRTGetWorldToObject)
internal float4x3 __metalRTGetWorldToObject();

/// Rejects the current candidate and exits the source any-hit invocation, including its helpers.
[require(metal, portable_raytracing_anyhit)]
__intrinsic_op(metalRTIgnoreHit)
internal void __metalRTIgnoreHit();

/// Accepts the candidate, exits the source any-hit invocation, and ends the traversal search.
[require(metal, portable_raytracing_anyhit)]
__intrinsic_op(metalRTAcceptHitAndEndSearch)
internal void __metalRTAcceptHitAndEndSearch();

/// Reports one procedural candidate to the selected hit group's Metal any-hit filtering logic.
///
/// The attributes retain their concrete type until the hit-group adapter supplies the candidate
/// protocol. A successful report returns true; it does not itself end the intersection invocation.
[require(metal, portable_raytracing_intersection)]
__intrinsic_op(metalRTReportHit)
internal bool __metalRTReportHit<Attributes>(float distance, uint hitKind, Attributes attributes);

/// Returns the minimum valid distance for the current ray.
///
/// @return The lower ray-parameter bound supplied to traversal.
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
    case metal:
        return __metalRTGetMinDistance();
    }
}

/// Returns the current hit distance or traversal upper bound, depending on the stage.
///
/// @return The current hit parameter, or the miss-stage traversal upper bound.
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
    case metal:
        return __metalRTGetCurrentDistance();
    }
}

/// Returns the classification of the current candidate or committed hit.
///
/// @return The native candidate or committed-hit classification.
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
    case metal:
        return __metalRTGetHitKind();
    }
}

/// Returns the flags supplied to the current trace operation.
///
/// @return The originating trace call's `RAY_FLAG` bit mask.
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
    case metal:
        return __metalRTGetRayFlags();
    }
}

/// Returns the index of the current primitive within its geometry.
///
/// @return The zero-based primitive index within the current geometry.
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
    case metal:
        return __metalRTGetPrimitiveIndex();
    }
}

/// Returns the index of the current geometry within its primitive acceleration structure.
///
/// @return The zero-based geometry index within the primitive acceleration structure.
[ForceInline]
internal uint getStageInputGeometryIndex()
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case spirv:
    case cuda:
        return GeometryIndex();
    case metal:
        return __metalRTGetGeometryIndex();
    }
}

/// Returns the origin of the current ray in world space.
///
/// @return The originating ray origin in world coordinates.
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
    case metal:
        return __metalRTGetWorldSpaceOrigin();
    }
}

/// Returns the direction of the current ray in world space.
///
/// @return The originating ray direction in world coordinates.
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
    case metal:
        return __metalRTGetWorldSpaceDirection();
    }
}

/// Returns the three-dimensional launch index of the current ray dispatch.
///
/// @return The current ray's launch coordinates in the originating dispatch.
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
    case metal:
        return __metalRTGetDispatchRaysIndex();
    }
}

/// Returns the dimensions of the current ray dispatch.
///
/// @return The originating dispatch dimensions.
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
    case metal:
        return __metalRTGetDispatchRaysDimensions();
    }
}

/// Returns the motion-interpolation time of the current ray.
///
/// @return The originating motion-enabled trace call's time.
[ForceInline]
internal float getStageInputRayTime()
{
    __target_switch
    {
    case glsl:
    case spirv:
        return RayCurrentTime();
    case metal:
        return __metalRTGetRayTime();
    case cuda:
        // TODO: A following PR must expose the current OptiX ray time through the core module.
        static_assert(false, "CUDA structural ray-tracing time is not implemented");
        return 0.0;
    }
}

/// Returns the index of the current instance in its acceleration structure.
///
/// @return The single instance index for portable two-level traversal.
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
    case metal:
        return getStageInputInstanceIndexAtLevel(0);
    }
}

/// Returns the application-provided identifier of the current instance.
///
/// @return The application-defined ID for portable two-level traversal.
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
    case metal:
        return getStageInputInstanceIDAtLevel(0);
    }
}

/// Returns the actual number of instances on the current multilevel primitive path.
///
/// @return The actual multilevel instance-path length; zero for direct primitive-AS traversal.
[require(metal)]
__intrinsic_op(metalRTGetInstanceCount)
internal uint getStageInputInstanceCount();

/// Returns the instance index at a valid level of the current multilevel primitive path.
///
/// @param[in] level Zero-based path level, from outermost to innermost. Must be smaller than
/// the actual instance count.
/// @return The instance index within the containing acceleration structure at `level`.
[require(metal)]
__intrinsic_op(metalRTGetInstanceIndex)
internal uint getStageInputInstanceIndexAtLevel(uint level);

/// Returns the application-defined ID at a valid level of the current multilevel primitive path.
///
/// @param[in] level Zero-based path level, from outermost to innermost. Must be smaller than
/// the actual instance count.
/// @return The application-defined instance ID at `level`.
[require(metal)]
__intrinsic_op(metalRTGetInstanceID)
internal uint getStageInputInstanceIDAtLevel(uint level);

/// Returns the current candidate ray transformed into object space.
///
/// Any-hit and intersection stages have direct object-ray intrinsics on every portable target.
///
/// @return The object-space candidate ray with its current traversal interval.
[ForceInline]
internal RayDesc getStageInputCandidateObjectSpaceRay()
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
    case metal:
        result = __metalRTGetObjectSpaceRay();
    }
    return result;
}

/// Returns the committed-hit ray transformed into object space.
///
/// OptiX does not expose object-ray intrinsics in a closest-hit stage, so that target reconstructs
/// the same view from closest-hit-legal world-ray and transform-stack values.
///
/// @return The object-space ray with the committed hit as its upper interval bound.
[ForceInline]
internal RayDesc getStageInputClosestHitObjectSpaceRay()
{
    RayDesc result = {};
    __target_switch
    {
    case hlsl:
    case glsl:
    case spirv:
        result.Origin = ObjectRayOrigin();
        result.TMin = RayTMin();
        result.Direction = ObjectRayDirection();
        result.TMax = RayTCurrent();
    case cuda:
        let worldToObject = WorldToObject3x4();
        result.Origin = mul(worldToObject, float4(WorldRayOrigin(), 1.0));
        result.TMin = RayTMin();
        result.Direction = mul(worldToObject, float4(WorldRayDirection(), 0.0));
        result.TMax = RayTCurrent();
    case metal:
        result = __metalRTGetObjectSpaceRay();
    }
    return result;
}

/// Returns the transform from the hit geometry's object space to the traversal's world space.
///
/// @return The complete object-to-world transform, or identity without instancing.
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
    case metal:
        return __metalRTGetObjectToWorld();
    }
}

/// Returns the transform from the traversal's world space to the hit geometry's object space.
///
/// @return The inverse complete transform, or identity without instancing.
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
    case metal:
        return __metalRTGetWorldToObject();
    }
}

/// Identifies the stage views that share ray and geometry state.
///
/// This internal contract shares implementations without introducing an erased public stage-input
/// type. Concrete input types retain their stage restrictions and context-dependent properties.
///
/// The shared property bodies live in the public extension below because Slang currently rejects
/// default property accessor bodies in interfaces and public members of an internal interface.
/// For example, putting the public `rayFlags` getter here would violate both rules. The extension
/// exposes `input.rayFlags` on each concrete input while this implementation-sharing marker remains
/// internal.
///
/// @par Constraints
/// Only the module's `ClosestHitInput`, `AnyHitInput`, and `IntersectionInput` views use this
/// internal sharing contract. It does not expose a public runtime input type.
internal interface ICommonHitInput
{}

/// Provides the properties available to closest-hit, any-hit, and intersection stages.
///
/// An intersection stage has not reported a hit yet, so `distance` and `hitKind` stay on the
/// closest-hit and any-hit inputs. Motion and scalar instance identifiers remain on their
/// constrained concrete extensions.
///
/// @tparam Input A concrete hit-input view conforming to `ICommonHitInput`.
/// @par Specialization
/// Extends `Input` only when `Input : ICommonHitInput`. The three conforming view families are
/// `ClosestHitInput<Context>`, `AnyHitInput<Context>`, and `IntersectionInput<Context>`; miss and
/// callable inputs do not acquire these members.
public extension<Input : ICommonHitInput> Input
{
    /// Reads the lower parameter bound of the originating ray.
    ///
    /// @par Constraints
    /// Read-only. Uses the same ray parameterization as `RayTraversalDesc.ray.TMin`; this getter
    /// does not normalize the ray direction.
    public property float minDistance
    {
        get { return getStageInputMinDistance(); }
    }

    /// Reads the traversal flags supplied by the originating trace call.
    ///
    /// @par Constraints
    /// Read-only `RAY_FLAG` bit mask, represented as `uint`. The selected target determines which
    /// flag combinations are supported.
    public property uint rayFlags
    {
        get { return getStageInputRayFlags(); }
    }

    /// Reads the index of the current primitive within its geometry.
    ///
    /// @par Constraints
    /// Read-only, zero-based geometry-local index. Requires a hit-stage view with a current
    /// candidate or committed primitive.
    public property uint primitiveIndex
    {
        get { return getStageInputPrimitiveIndex(); }
    }

    /// Reads the index of the current geometry within its primitive acceleration structure.
    ///
    /// @par Constraints
    /// Read-only, zero-based geometry index. On OptiX this follows the core `GeometryIndex()`
    /// contract: the SBT GAS index equals the geometry index when each build input has one SBT record.
    public property uint geometryIndex
    {
        get { return getStageInputGeometryIndex(); }
    }

    /// Reads the originating ray origin in world coordinates.
    ///
    /// @par Constraints
    /// Read-only `float3`; it uses the traversal's world space, before instance transforms.
    public property float3 worldSpaceOrigin
    {
        get { return getStageInputWorldSpaceOrigin(); }
    }

    /// Reads the originating ray direction in world coordinates.
    ///
    /// @par Constraints
    /// Read-only `float3`; it preserves the traversal direction and is not normalized by this view.
    public property float3 worldSpaceDirection
    {
        get { return getStageInputWorldSpaceDirection(); }
    }

    /// Transforms the hit geometry's object-space coordinates into the traversal's world space.
    ///
    /// This includes every instance transform along the hit path. For example, an inner instance
    /// translating by two units and an outer instance translating by ten units on the same axis
    /// produce a combined translation of twelve units. Without instancing, this is the identity.
    ///
    /// @par Constraints
    /// Read-only `float4x3` with the same meaning for portable and multilevel hit contexts.
    /// Direct primitive-AS traversal requires an identity matrix from the later Metal adapter.
    public property float4x3 objectToWorld
    {
        get { return getStageInputObjectToWorld(); }
    }

    /// Transforms world-space coordinates into the hit geometry's object space.
    ///
    /// This is the inverse of the complete `objectToWorld` transform, including all instance
    /// levels. Without instancing, this is the identity.
    ///
    /// @par Constraints
    /// Read-only `float4x3`, inverse to `objectToWorld`. Available to all three hit-input families;
    /// direct primitive-AS traversal requires an identity matrix from the later Metal adapter.
    public property float4x3 worldToObject
    {
        get { return getStageInputWorldToObject(); }
    }

    /// Reads the originating ray-dispatch coordinates.
    ///
    /// @par Constraints
    /// Read-only, zero-based coordinates within `dispatchRaysDimensions`. Nested hit or callable
    /// dispatch retains the originating launch coordinates.
    public property uint3 dispatchRaysIndex
    {
        get { return getStageInputDispatchRaysIndex(); }
    }

    /// Reads the dimensions of the originating ray dispatch.
    ///
    /// @par Constraints
    /// Read-only `uint3` describing the launch dimensions used by `dispatchRaysIndex`.
    public property uint3 dispatchRaysDimensions
    {
        get { return getStageInputDispatchRaysDimensions(); }
    }
}

/// Provides the payload, record, hit data, and ray state available to a closest-hit stage.
///
/// This is a compiler-provided zero-storage view. Adapter synthesis must map its properties to
/// native stage parameters or built-in state without adding source fields to the entry point.
///
/// @tparam Context The stage context; must conform to `IHitContext`.
/// @par Constraints
/// The view requires `Context : IHitContext` and its declared structural stage capability.
/// An initialized local copy shares the same invocation state and can be passed to an `in` helper
/// parameter. Copying the view does not duplicate its payload or record storage.
/// Applications cannot construct a fresh input, declare an uninitialized input local, use an input
/// as a global shader parameter or field, or return one from a function.
[require(portable_raytracing_closesthit)]
[KnownBuiltin("RayTracingClosestHitInput")]
public struct ClosestHitInput<Context> : ICommonHitInput
    where Context : IHitContext
{
    /// Gives mutable access to the payload supplied by the originating `RayTracer.trace` call.
    /// The compiler maps this accessor to the invocation's native payload reference.
    ///
    /// @par Constraints
    /// The referenced type is exactly `Context.Payload`, inherited through `IPayloadContext`.
    /// The `[nonmutating] ref` accessor leaves the view unchanged and permits mutation of the
    /// originating payload through an `in` parameter or immutable local copy. Empty structs are
    /// valid payload types.
    public property Context.Payload payload
    {
        __intrinsic_op(portableRTGetPayload)
        [nonmutating]
        ref;
    }

    /// Reads the host-provided data in the selected shader-binding-table record.
    /// The compiler maps this accessor to the selected invocation's record-data binding.
    ///
    /// @par Constraints
    /// Read-only value of exactly `Context.Record`. The getter returns application data by value;
    /// copied views read the same bound record. The host-provided record must match that type's
    /// target layout; `void` selects a context with no application record data.
    public property Context.Record record
    {
        __intrinsic_op(portableRTGetRecord)
        get;
    }

    /// Reads the committed hit's attributes using the type selected by `Context.Primitive`.
    ///
    /// A triangle context exposes `attributes.barycentricCoord` through `TriangleData`; a
    /// `BoundingBoxPrimitive<MyAttributes>` context exposes the `MyAttributes` value reported by
    /// its intersection shader. A curve context exposes `CurveData`. Generic callers need ordinary
    /// type constraints before accessing members of the associated attribute type.
    /// Adapter synthesis returns the built-in view or reads the procedural hit attributes.
    ///
    /// @par Constraints
    /// Read-only value of exactly `Context.Primitive.Attributes`. `TrianglePrimitive` selects
    /// `TriangleData`, `CurvePrimitive` selects `CurveData`, and `BoundingBoxPrimitive<T>`
    /// selects `T`. Access to members of an abstract attribute type needs ordinary type
    /// constraints.
    public property Context.Primitive.Attributes attributes
    {
        __intrinsic_op(portableRTGetHitAttributes)
        get;
    }

    /// Reads the committed hit parameter.
    ///
    /// @par Constraints
    /// Read-only `float` in the originating ray's parameterization. Available on `ClosestHitInput`;
    /// intersection inputs do not expose a hit distance before reporting a candidate.
    public property float distance
    {
        get { return getStageInputCurrentDistance(); }
    }

    /// Reads the classification of the committed hit.
    ///
    /// @par Constraints
    /// Read-only `uint` classification supplied by built-in intersection or procedural `reportHit`.
    /// The meaning of built-in classifications follows the selected target and primitive.
    public property uint hitKind
    {
        get { return getStageInputHitKind(); }
    }
}

/// Exposes ray time on motion-enabled `ClosestHitInput<Context>` views.
///
/// @tparam Context A stage context conforming to `IHitContext`.
/// @par Specialization
/// Extends `ClosestHitInput<Context>` when `Context : IHitContext` and
/// `Context.TraceContext.Motion : IEnabledRayMotion`. The motion conformance admits
/// `PrimitiveMotion`, `InstanceMotion`, and `PrimitiveAndInstanceMotion`; `NoMotion`
/// does not satisfy it.
public extension<Context> ClosestHitInput<Context>
    where Context : IHitContext
    where Context.TraceContext.Motion : IEnabledRayMotion
{
    /// Reads the time supplied by the originating motion-enabled trace call.
    ///
    /// @par Constraints
    /// Read-only. Available only when `Context.TraceContext.Motion : IEnabledRayMotion`; a
    /// `NoMotion` context does not expose this property.
    public property float time
    {
        get { return getStageInputRayTime(); }
    }
}

/// Exposes the portable two-level instance state on `ClosestHitInput<Context>`.
///
/// @tparam Context A hit context conforming to `IHitContext`.
/// @par Specialization
/// Extends `ClosestHitInput<Context>` when `Context : IHitContext` and
/// `Context.TraceContext.AccelerationStructure == AccelerationStructure`. This equality
/// selects the portable two-level handle, so a hit path has one instance entry; it does not
/// admit `MultiLevelAccelerationStructure<N>`. The motion type has no additional constraint.
public extension<Context> ClosestHitInput<Context>
    where Context : IHitContext
    where Context.TraceContext.AccelerationStructure == AccelerationStructure
{
    /// Reads the current instance's index within the top-level acceleration structure.
    ///
    /// @par Constraints
    /// Read-only, zero-based instance index. Available only for
    /// `Context.TraceContext.AccelerationStructure == AccelerationStructure`, which selects
    /// portable two-level traversal with one instance entry per hit path.
    public property uint instanceIndex
    {
        get { return getStageInputInstanceIndex(); }
    }

    /// Reads the application-defined ID of the current instance.
    ///
    /// @par Constraints
    /// Read-only. Available only for
    /// `Context.TraceContext.AccelerationStructure == AccelerationStructure`. The value comes
    /// from the instance descriptor and need not equal `instanceIndex`.
    public property uint instanceID
    {
        get { return getStageInputInstanceID(); }
    }

    /// Reads the committed-hit ray in the current primitive's object space.
    ///
    /// @par Constraints
    /// Read-only `RayDesc`; its interval uses the current traversal bounds.
    /// Available only for `Context.TraceContext.AccelerationStructure == AccelerationStructure`.
    public property RayDesc objectSpaceRay
    {
        get { return getStageInputClosestHitObjectSpaceRay(); }
    }
}

/// Exposes the instance path for a multilevel committed hit.
///
/// @tparam Context A hit context conforming to `IHitContext`.
/// @par Specialization
/// Extends `ClosestHitInput<Context>` when `Context : IHitContext` and
/// `Context.TraceContext.AccelerationStructure : IMultiLevelAccelerationStructure`. Only
/// `MultiLevelAccelerationStructure<N>` supplies that internal conformance; this selects
/// every supported depth, including direct primitive-AS traversal at `N == 1`.
/// @par Availability
/// Requires the `metal` capability. Runtime path reads still require adapter synthesis.
[require(metal)]
public extension<Context> ClosestHitInput<Context>
    where Context : IHitContext
    where Context.TraceContext.AccelerationStructure : IMultiLevelAccelerationStructure
{
    /// Gives the maximum number of instances on a path as a specialization-time constant.
    ///
    /// One acceleration-structure level belongs to the primitive AS, so a level count of three
    /// allows at most two instances. The actual `instanceCount` can be smaller than this limit.
    ///
    /// @par Constraints
    /// Equals `Context.TraceContext.AccelerationStructure.levelCount - 1` for the selected
    /// multilevel handle. It is a specialization-time `uint`, not the number of instances in a hit.
    public static const uint maxInstanceCount =
        Context.TraceContext.AccelerationStructure.levelCount - 1;

    /// Counts the instances on the current primitive's path.
    ///
    /// The count excludes the primitive acceleration structure and is at most `maxLevelCount - 1`.
    /// It is the actual path length, which can be shorter than that maximum. Direct primitive-AS
    /// traversal with `MultiLevelAccelerationStructure<1>` has no instances, so its count is zero.
    ///
    /// @par Constraints
    /// Read-only runtime value satisfying `instanceCount <= maxInstanceCount`. Available only
    /// for a `MultiLevelAccelerationStructure<N>` context through its
    /// `IMultiLevelAccelerationStructure` conformance; the static maximum is not the actual count.
    public property uint instanceCount
    {
        get { return getStageInputInstanceCount(); }
    }

    /// Returns the instance index within the containing acceleration structure at the given level.
    ///
    /// Level zero is the outermost instance; increasing levels move toward the primitive.
    /// For a path [3, 7], level zero selects instance 3 and level one selects instance 7 inside it.
    /// The caller must provide `level < instanceCount`.
    ///
    /// @param[in] level Zero-based instance-path level, ordered from outermost to innermost.
    /// @pre `level < instanceCount`; indexing the configured maximum is insufficient for a shorter
    /// hit.
    /// @return The selected instance's index within that level's containing acceleration structure.
    public uint getInstanceIndex(uint level)
    {
        return getStageInputInstanceIndexAtLevel(level);
    }

    /// Returns the application-defined instance ID at the given level of the current path.
    ///
    /// Levels use the same outermost-to-innermost order as `getInstanceIndex`.
    /// The caller must provide `level < instanceCount`.
    ///
    /// @param[in] level Zero-based instance-path level, ordered from outermost to innermost.
    /// @pre `level < instanceCount`; indexing the configured maximum is insufficient for a shorter
    /// hit.
    /// @return The selected instance's application-defined ID.
    public uint getInstanceID(uint level)
    {
        return getStageInputInstanceIDAtLevel(level);
    }
}

/// Provides the candidate payload, record, hit data, and ray state available to an any-hit stage.
///
/// Control methods on this zero-storage view use native any-hit operations where available.
/// On Metal, the generated candidate dispatcher implements the same control decisions.
///
/// @tparam Context The stage context; must conform to `IHitContext`.
/// @par Constraints
/// The view requires `Context : IHitContext` and its declared structural stage capability.
/// An initialized local copy shares the same invocation state and can be passed to an `in` helper
/// parameter. Copying the view does not duplicate its payload or record storage.
/// Applications cannot construct a fresh input, declare an uninitialized input local, use an input
/// as a global shader parameter or field, or return one from a function.
[require(portable_raytracing_anyhit)]
[KnownBuiltin("RayTracingAnyHitInput")]
public struct AnyHitInput<Context> : ICommonHitInput
    where Context : IHitContext
{
    /// Gives mutable access to the payload supplied by the originating `RayTracer.trace` call.
    /// The compiler maps this accessor to the invocation's native payload reference.
    ///
    /// @par Constraints
    /// The referenced type is exactly `Context.Payload`, inherited through `IPayloadContext`.
    /// The `[nonmutating] ref` accessor leaves the view unchanged and permits mutation of the
    /// originating payload through an `in` parameter or immutable local copy. Empty structs are
    /// valid payload types.
    public property Context.Payload payload
    {
        __intrinsic_op(portableRTGetPayload)
        [nonmutating]
        ref;
    }

    /// Reads the host-provided data in the selected shader-binding-table record.
    /// The compiler maps this accessor to the selected invocation's record-data binding.
    ///
    /// @par Constraints
    /// Read-only value of exactly `Context.Record`. The getter returns application data by value;
    /// copied views read the same bound record. The host-provided record must match that type's
    /// target layout; `void` selects a context with no application record data.
    public property Context.Record record
    {
        __intrinsic_op(portableRTGetRecord)
        get;
    }

    /// Reads the candidate hit's attributes using the type selected by `Context.Primitive`.
    ///
    /// Built-in primitives expose their `TriangleData` or `CurveData` view; procedural primitives
    /// expose the application-defined value supplied by the intersection shader's `reportHit` call.
    /// Adapter synthesis returns the built-in view or reads the procedural hit attributes.
    ///
    /// @par Constraints
    /// Read-only value of exactly `Context.Primitive.Attributes`. `TrianglePrimitive` selects
    /// `TriangleData`, `CurvePrimitive` selects `CurveData`, and `BoundingBoxPrimitive<T>`
    /// selects `T`. Access to members of an abstract attribute type needs ordinary type
    /// constraints.
    public property Context.Primitive.Attributes attributes
    {
        __intrinsic_op(portableRTGetHitAttributes)
        get;
    }

    /// Reads the current candidate hit parameter.
    ///
    /// @par Constraints
    /// Read-only `float` in the originating ray's parameterization. Available on `AnyHitInput`;
    /// intersection inputs do not expose a hit distance before reporting a candidate.
    public property float distance
    {
        get { return getStageInputCurrentDistance(); }
    }

    /// Reads the classification of the candidate hit.
    ///
    /// @par Constraints
    /// Read-only `uint` classification supplied by built-in intersection or procedural `reportHit`.
    /// The meaning of built-in classifications follows the selected target and primitive.
    public property uint hitKind
    {
        get { return getStageInputHitKind(); }
    }

    /// Rejects the current any-hit candidate and allows traversal to search for another hit.
    ///
    /// @pre Called through an `AnyHitInput<Context>` in an any-hit stage.
    public void ignoreHit()
    {
        __target_switch
        {
        case hlsl:
        case glsl:
        case spirv:
        case cuda:
            IgnoreHit();
        case metal:
            __metalRTIgnoreHit();
            break;
        }
    }

    /// Accepts the current any-hit candidate and terminates the traversal search.
    ///
    /// @pre Called through an `AnyHitInput<Context>` in an any-hit stage.
    public void acceptHitAndEndSearch()
    {
        __target_switch
        {
        case hlsl:
        case glsl:
        case spirv:
        case cuda:
            AcceptHitAndEndSearch();
        case metal:
            __metalRTAcceptHitAndEndSearch();
            break;
        }
    }
}

/// Exposes ray time on motion-enabled `AnyHitInput<Context>` views.
///
/// @tparam Context A stage context conforming to `IHitContext`.
/// @par Specialization
/// Extends `AnyHitInput<Context>` when `Context : IHitContext` and
/// `Context.TraceContext.Motion : IEnabledRayMotion`. The motion conformance admits
/// `PrimitiveMotion`, `InstanceMotion`, and `PrimitiveAndInstanceMotion`; `NoMotion`
/// does not satisfy it.
public extension<Context> AnyHitInput<Context>
    where Context : IHitContext
    where Context.TraceContext.Motion : IEnabledRayMotion
{
    /// Reads the time supplied by the originating motion-enabled trace call.
    ///
    /// @par Constraints
    /// Read-only. Available only when `Context.TraceContext.Motion : IEnabledRayMotion`; a
    /// `NoMotion` context does not expose this property.
    public property float time
    {
        get { return getStageInputRayTime(); }
    }
}

/// Exposes the portable two-level instance state on `AnyHitInput<Context>`.
///
/// @tparam Context A hit context conforming to `IHitContext`.
/// @par Specialization
/// Extends `AnyHitInput<Context>` when `Context : IHitContext` and
/// `Context.TraceContext.AccelerationStructure == AccelerationStructure`. This equality
/// selects the portable two-level handle, so a hit path has one instance entry; it does not
/// admit `MultiLevelAccelerationStructure<N>`. The motion type has no additional constraint.
public extension<Context> AnyHitInput<Context>
    where Context : IHitContext
    where Context.TraceContext.AccelerationStructure == AccelerationStructure
{
    /// Reads the current instance's index within the top-level acceleration structure.
    ///
    /// @par Constraints
    /// Read-only, zero-based instance index. Available only for
    /// `Context.TraceContext.AccelerationStructure == AccelerationStructure`, which selects
    /// portable two-level traversal with one instance entry per hit path.
    public property uint instanceIndex
    {
        get { return getStageInputInstanceIndex(); }
    }

    /// Reads the application-defined ID of the current instance.
    ///
    /// @par Constraints
    /// Read-only. Available only for
    /// `Context.TraceContext.AccelerationStructure == AccelerationStructure`. The value comes
    /// from the instance descriptor and need not equal `instanceIndex`.
    public property uint instanceID
    {
        get { return getStageInputInstanceID(); }
    }

    /// Reads the candidate ray in the current primitive's object space.
    ///
    /// @par Constraints
    /// Read-only `RayDesc`; its interval uses the current traversal bounds.
    /// Available only for `Context.TraceContext.AccelerationStructure == AccelerationStructure`.
    public property RayDesc objectSpaceRay
    {
        get { return getStageInputCandidateObjectSpaceRay(); }
    }
}

/// Exposes the instance path for a multilevel candidate primitive.
///
/// @tparam Context A hit context conforming to `IHitContext`.
/// @par Specialization
/// Extends `AnyHitInput<Context>` when `Context : IHitContext` and
/// `Context.TraceContext.AccelerationStructure : IMultiLevelAccelerationStructure`. Only
/// `MultiLevelAccelerationStructure<N>` supplies that internal conformance; this selects
/// every supported depth, including direct primitive-AS traversal at `N == 1`.
/// @par Availability
/// Requires the `metal` capability. Runtime path reads still require adapter synthesis.
[require(metal)]
public extension<Context> AnyHitInput<Context>
    where Context : IHitContext
    where Context.TraceContext.AccelerationStructure : IMultiLevelAccelerationStructure
{
    /// Gives the maximum number of instances on a path as a specialization-time constant.
    ///
    /// One acceleration-structure level belongs to the primitive AS, so a level count of three
    /// allows at most two instances. The actual `instanceCount` can be smaller than this limit.
    ///
    /// @par Constraints
    /// Equals `Context.TraceContext.AccelerationStructure.levelCount - 1` for the selected
    /// multilevel handle. It is a specialization-time `uint`, not the number of instances in a hit.
    public static const uint maxInstanceCount =
        Context.TraceContext.AccelerationStructure.levelCount - 1;

    /// Counts the instances on the current primitive's path.
    ///
    /// The count excludes the primitive acceleration structure and is at most `maxLevelCount - 1`.
    /// It is the actual path length, which can be shorter than that maximum. Direct primitive-AS
    /// traversal with `MultiLevelAccelerationStructure<1>` has no instances, so its count is zero.
    ///
    /// @par Constraints
    /// Read-only runtime value satisfying `instanceCount <= maxInstanceCount`. Available only
    /// for a `MultiLevelAccelerationStructure<N>` context through its
    /// `IMultiLevelAccelerationStructure` conformance; the static maximum is not the actual count.
    public property uint instanceCount
    {
        get { return getStageInputInstanceCount(); }
    }

    /// Returns the instance index within the containing acceleration structure at the given level.
    ///
    /// Level zero is the outermost instance; increasing levels move toward the primitive.
    /// For a path [3, 7], level zero selects instance 3 and level one selects instance 7 inside it.
    /// The caller must provide `level < instanceCount`.
    ///
    /// @param[in] level Zero-based instance-path level, ordered from outermost to innermost.
    /// @pre `level < instanceCount`; indexing the configured maximum is insufficient for a shorter
    /// hit.
    /// @return The selected instance's index within that level's containing acceleration structure.
    public uint getInstanceIndex(uint level)
    {
        return getStageInputInstanceIndexAtLevel(level);
    }

    /// Returns the application-defined instance ID at the given level of the current path.
    ///
    /// Levels use the same outermost-to-innermost order as `getInstanceIndex`.
    /// The caller must provide `level < instanceCount`.
    ///
    /// @param[in] level Zero-based instance-path level, ordered from outermost to innermost.
    /// @pre `level < instanceCount`; indexing the configured maximum is insufficient for a shorter
    /// hit.
    /// @return The selected instance's application-defined ID.
    public uint getInstanceID(uint level)
    {
        return getStageInputInstanceIDAtLevel(level);
    }
}

/// Provides the record and ray state used by a procedural intersection stage.
///
/// Payload access is additionally available on Metal and CUDA/OptiX.
/// `reportHit` submits each candidate and its `Context.Primitive.Attributes` value. On D3D,
/// Vulkan, and OptiX it maps to the native report-intersection operation.
/// On Metal, the generated intersection function filters candidates through the group's any-hit
/// logic and retains the accepted candidate's attributes for closest-hit dispatch.
///
/// @tparam Context The stage context; must conform to `IHitContext`.
/// @par Constraints
/// The view requires `Context : IHitContext` and its declared structural stage capability.
/// An initialized local copy shares the same invocation state and can be passed to an `in` helper
/// parameter. Copying the view does not duplicate its payload or record storage.
/// Applications cannot construct a fresh input, declare an uninitialized input local, use an input
/// as a global shader parameter or field, or return one from a function.
/// An `IIntersectionShader` implementation additionally requires
/// `Context.Primitive : ICustomIntersectionPrimitive`.
[require(portable_raytracing_intersection)]
[KnownBuiltin("RayTracingIntersectionInput")]
public struct IntersectionInput<Context> : ICommonHitInput
    where Context : IHitContext
{
    /// Gives mutable access to the payload supplied by the originating `RayTracer.trace` call.
    /// The compiler maps this view to Metal's `ray_data` payload or OptiX payload state.
    ///
    /// @par Constraints
    /// Requires `portable_raytracing_intersection_payload` (Metal or CUDA/OptiX).
    /// The `[nonmutating] ref` accessor leaves the view unchanged and exposes `Context.Payload`
    /// for mutation through an `in` parameter or immutable local copy.
    [require(portable_raytracing_intersection_payload)]
    public property Context.Payload payload
    {
        __intrinsic_op(portableRTGetPayload)
        [nonmutating]
        ref;
    }

    /// Reads the host-provided data in the selected shader-binding-table record.
    /// The compiler maps this accessor to the selected invocation's record-data binding.
    ///
    /// @par Constraints
    /// Read-only value of exactly `Context.Record`. The getter returns application data by value;
    /// copied views read the same bound record. The host-provided record must match that type's
    /// target layout; `void` selects a context with no application record data.
    public property Context.Record record
    {
        __intrinsic_op(portableRTGetRecord)
        get;
    }

    /// Reads the candidate ray in the current primitive's object space.
    ///
    /// @par Constraints
    /// Read-only `RayDesc`; its interval uses the current traversal bounds.
    /// Available for every `IntersectionInput<Context>` with `Context : IHitContext`.
    public property RayDesc objectSpaceRay
    {
        get { return getStageInputCandidateObjectSpaceRay(); }
    }

    /// Reports a procedural hit candidate with a hit kind of zero.
    ///
    /// @param[in] distance Candidate ray parameter in the current traversal interval.
    /// @param[in] attributes Attributes of exactly `Context.Primitive.Attributes`, passed to the
    /// any-hit stage while evaluating the candidate, and to closest-hit if this candidate becomes
    /// the committed hit.
    /// @return Whether traversal accepts the candidate after any-hit processing.
    /// @pre The invoking intersection shader uses `Context.Primitive :
    /// ICustomIntersectionPrimitive`.
    public bool reportHit(float distance, Context.Primitive.Attributes attributes)
    {
        __target_switch
        {
        case hlsl:
        case glsl:
        case spirv:
        case cuda:
            return ReportHit(distance, 0, attributes);
        case metal:
            return __metalRTReportHit(distance, 0, attributes);
        }
    }

    /// Reports a procedural hit candidate with an application-supplied classification.
    ///
    /// @param[in] distance Candidate ray parameter in the current traversal interval.
    /// @param[in] hitKind Application-defined classification satisfying the target's procedural
    /// hit-kind restrictions.
    /// @param[in] attributes Attributes of exactly `Context.Primitive.Attributes`, passed to the
    /// any-hit stage while evaluating the candidate, and to closest-hit if this candidate becomes
    /// the committed hit.
    /// @return Whether traversal accepts the candidate after any-hit processing.
    /// @pre The invoking intersection shader uses `Context.Primitive :
    /// ICustomIntersectionPrimitive`.
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
        case cuda:
            return ReportHit(distance, hitKind, attributes);
        case metal:
            return __metalRTReportHit(distance, hitKind, attributes);
        }
    }
}

/// Exposes ray time on motion-enabled `IntersectionInput<Context>` views.
///
/// @tparam Context A stage context conforming to `IHitContext`.
/// @par Specialization
/// Extends `IntersectionInput<Context>` when `Context : IHitContext` and
/// `Context.TraceContext.Motion : IEnabledRayMotion`. The motion conformance admits
/// `PrimitiveMotion`, `InstanceMotion`, and `PrimitiveAndInstanceMotion`; `NoMotion`
/// does not satisfy it.
public extension<Context> IntersectionInput<Context>
    where Context : IHitContext
    where Context.TraceContext.Motion : IEnabledRayMotion
{
    /// Reads the time supplied by the originating motion-enabled trace call.
    ///
    /// @par Constraints
    /// Read-only. Available only when `Context.TraceContext.Motion : IEnabledRayMotion`; a
    /// `NoMotion` context does not expose this property.
    public property float time
    {
        get { return getStageInputRayTime(); }
    }
}

/// Exposes the portable two-level instance state on `IntersectionInput<Context>`.
///
/// @tparam Context A hit context conforming to `IHitContext`.
/// @par Specialization
/// Extends `IntersectionInput<Context>` when `Context : IHitContext` and
/// `Context.TraceContext.AccelerationStructure == AccelerationStructure`. This equality
/// selects the portable two-level handle, so a hit path has one instance entry; it does not
/// admit `MultiLevelAccelerationStructure<N>`. The motion type has no additional constraint.
public extension<Context> IntersectionInput<Context>
    where Context : IHitContext
    where Context.TraceContext.AccelerationStructure == AccelerationStructure
{
    /// Reads the current instance's index within the top-level acceleration structure.
    ///
    /// @par Constraints
    /// Read-only, zero-based instance index. Available only for
    /// `Context.TraceContext.AccelerationStructure == AccelerationStructure`, which selects
    /// portable two-level traversal with one instance entry per hit path.
    public property uint instanceIndex
    {
        get { return getStageInputInstanceIndex(); }
    }

    /// Reads the application-defined ID of the current instance.
    ///
    /// @par Constraints
    /// Read-only. Available only for
    /// `Context.TraceContext.AccelerationStructure == AccelerationStructure`. The value comes
    /// from the instance descriptor and need not equal `instanceIndex`.
    public property uint instanceID
    {
        get { return getStageInputInstanceID(); }
    }
}

/// Exposes the instance path for a multilevel candidate primitive.
///
/// @tparam Context A hit context conforming to `IHitContext`.
/// @par Specialization
/// Extends `IntersectionInput<Context>` when `Context : IHitContext` and
/// `Context.TraceContext.AccelerationStructure : IMultiLevelAccelerationStructure`. Only
/// `MultiLevelAccelerationStructure<N>` supplies that internal conformance; this selects
/// every supported depth, including direct primitive-AS traversal at `N == 1`.
/// @par Availability
/// Requires the `metal` capability. Runtime path reads still require adapter synthesis.
[require(metal)]
public extension<Context> IntersectionInput<Context>
    where Context : IHitContext
    where Context.TraceContext.AccelerationStructure : IMultiLevelAccelerationStructure
{
    /// Gives the maximum number of instances on a path as a specialization-time constant.
    ///
    /// One acceleration-structure level belongs to the primitive AS, so a level count of three
    /// allows at most two instances. The actual `instanceCount` can be smaller than this limit.
    ///
    /// @par Constraints
    /// Equals `Context.TraceContext.AccelerationStructure.levelCount - 1` for the selected
    /// multilevel handle. It is a specialization-time `uint`, not the number of instances in a hit.
    public static const uint maxInstanceCount =
        Context.TraceContext.AccelerationStructure.levelCount - 1;

    /// Counts the instances on the current primitive's path.
    ///
    /// The count excludes the primitive acceleration structure and is at most `maxLevelCount - 1`.
    /// It is the actual path length, which can be shorter than that maximum. Direct primitive-AS
    /// traversal with `MultiLevelAccelerationStructure<1>` has no instances, so its count is zero.
    ///
    /// @par Constraints
    /// Read-only runtime value satisfying `instanceCount <= maxInstanceCount`. Available only
    /// for a `MultiLevelAccelerationStructure<N>` context through its
    /// `IMultiLevelAccelerationStructure` conformance; the static maximum is not the actual count.
    public property uint instanceCount
    {
        get { return getStageInputInstanceCount(); }
    }

    /// Returns the instance index within the containing acceleration structure at the given level.
    ///
    /// Level zero is the outermost instance; increasing levels move toward the primitive.
    /// For a path [3, 7], level zero selects instance 3 and level one selects instance 7 inside it.
    /// The caller must provide `level < instanceCount`.
    ///
    /// @param[in] level Zero-based instance-path level, ordered from outermost to innermost.
    /// @pre `level < instanceCount`; indexing the configured maximum is insufficient for a shorter
    /// hit.
    /// @return The selected instance's index within that level's containing acceleration structure.
    public uint getInstanceIndex(uint level)
    {
        return getStageInputInstanceIndexAtLevel(level);
    }

    /// Returns the application-defined instance ID at the given level of the current path.
    ///
    /// Levels use the same outermost-to-innermost order as `getInstanceIndex`.
    /// The caller must provide `level < instanceCount`.
    ///
    /// @param[in] level Zero-based instance-path level, ordered from outermost to innermost.
    /// @pre `level < instanceCount`; indexing the configured maximum is insufficient for a shorter
    /// hit.
    /// @return The selected instance's application-defined ID.
    public uint getInstanceID(uint level)
    {
        return getStageInputInstanceIDAtLevel(level);
    }
}

/// Provides the payload, record, and ray state available to a miss stage.
///
/// Adapter synthesis must map this zero-storage view to native miss parameters where available
/// and to Metal's generated post-traversal dispatch.
///
/// @tparam Context The stage context; must conform to `IPayloadContext`.
/// @par Constraints
/// The view requires `Context : IPayloadContext` and its declared structural stage capability.
/// An initialized local copy shares the same invocation state and can be passed to an `in` helper
/// parameter. Copying the view does not duplicate its payload or record storage.
/// Applications cannot construct a fresh input, declare an uninitialized input local, use an input
/// as a global shader parameter or field, or return one from a function.
[require(portable_raytracing_miss)]
[KnownBuiltin("RayTracingMissInput")]
public struct MissInput<Context>
    where Context : IPayloadContext
{
    /// Gives mutable access to the payload supplied by the originating `RayTracer.trace` call.
    /// The compiler maps this accessor to the invocation's native payload reference.
    ///
    /// @par Constraints
    /// The referenced type is exactly `Context.Payload`, inherited through `IPayloadContext`.
    /// The `[nonmutating] ref` accessor leaves the view unchanged and permits mutation of the
    /// originating payload through an `in` parameter or immutable local copy. Empty structs are
    /// valid payload types.
    public property Context.Payload payload
    {
        __intrinsic_op(portableRTGetPayload)
        [nonmutating]
        ref;
    }

    /// Reads the host-provided data in the selected shader-binding-table record.
    /// The compiler maps this accessor to the selected invocation's record-data binding.
    ///
    /// @par Constraints
    /// Read-only value of exactly `Context.Record`. The getter returns application data by value;
    /// copied views read the same bound record. The host-provided record must match that type's
    /// target layout; `void` selects a context with no application record data.
    public property Context.Record record
    {
        __intrinsic_op(portableRTGetRecord)
        get;
    }

    /// Reads the lower parameter bound of the originating ray.
    ///
    /// @par Constraints
    /// Read-only. Uses the same ray parameterization as `RayTraversalDesc.ray.TMin`; this getter
    /// does not normalize the ray direction.
    public property float minDistance
    {
        get { return getStageInputMinDistance(); }
    }

    /// Reads the traversal upper bound when no hit was accepted.
    ///
    /// @par Constraints
    /// Read-only `float` in the originating ray's parameterization. Available on `MissInput`;
    /// intersection inputs do not expose a hit distance before reporting a candidate.
    public property float distance
    {
        get { return getStageInputCurrentDistance(); }
    }

    /// Reads the traversal flags supplied by the originating trace call.
    ///
    /// @par Constraints
    /// Read-only `RAY_FLAG` bit mask, represented as `uint`. The selected target determines which
    /// flag combinations are supported.
    public property uint rayFlags
    {
        get { return getStageInputRayFlags(); }
    }

    /// Reads the originating ray origin in world coordinates.
    ///
    /// @par Constraints
    /// Read-only `float3`; it uses the traversal's world space, before instance transforms.
    public property float3 worldSpaceOrigin
    {
        get { return getStageInputWorldSpaceOrigin(); }
    }

    /// Reads the originating ray direction in world coordinates.
    ///
    /// @par Constraints
    /// Read-only `float3`; it preserves the traversal direction and is not normalized by this view.
    public property float3 worldSpaceDirection
    {
        get { return getStageInputWorldSpaceDirection(); }
    }

    /// Reads the originating ray-dispatch coordinates.
    ///
    /// @par Constraints
    /// Read-only, zero-based coordinates within `dispatchRaysDimensions`. Nested hit or callable
    /// dispatch retains the originating launch coordinates.
    public property uint3 dispatchRaysIndex
    {
        get { return getStageInputDispatchRaysIndex(); }
    }

    /// Reads the dimensions of the originating ray dispatch.
    ///
    /// @par Constraints
    /// Read-only `uint3` describing the launch dimensions used by `dispatchRaysIndex`.
    public property uint3 dispatchRaysDimensions
    {
        get { return getStageInputDispatchRaysDimensions(); }
    }
}

/// Exposes ray time on motion-enabled `MissInput<Context>` views.
///
/// @tparam Context A stage context conforming to `IPayloadContext`.
/// @par Specialization
/// Extends `MissInput<Context>` when `Context : IPayloadContext` and
/// `Context.TraceContext.Motion : IEnabledRayMotion`. The motion conformance admits
/// `PrimitiveMotion`, `InstanceMotion`, and `PrimitiveAndInstanceMotion`; `NoMotion`
/// does not satisfy it.
public extension<Context> MissInput<Context>
    where Context : IPayloadContext
    where Context.TraceContext.Motion : IEnabledRayMotion
{
    /// Reads the time supplied by the originating motion-enabled trace call.
    ///
    /// @par Constraints
    /// Read-only. Available only when `Context.TraceContext.Motion : IEnabledRayMotion`; a
    /// `NoMotion` context does not expose this property.
    public property float time
    {
        get { return getStageInputRayTime(); }
    }
}

/// Provides callable data, record data, and dispatch coordinates to a callable stage.
///
/// The data property is the native inout callable-data parameter; record data identifies the
/// runtime callable SBT record selected by `callShader`.
///
/// @tparam Context The stage context; must conform to `ICallableContext`.
/// @par Constraints
/// The view requires `Context : ICallableContext` and its declared structural stage capability.
/// An initialized local copy shares the same invocation state and can be passed to an `in` helper
/// parameter. Copying the view does not duplicate its callable-data or record storage.
/// Applications cannot construct a fresh input, declare an uninitialized input local, use an input
/// as a global shader parameter or field, or return one from a function.
[require(portable_raytracing_callable)]
[KnownBuiltin("RayTracingCallableInput")]
public struct CallableInput<Context>
    where Context : ICallableContext
{
    /// Gives mutable access to the data supplied by the originating `RayTracer.callShader` call.
    /// The compiler maps this accessor to the invocation's native callable-data reference.
    ///
    /// @par Constraints
    /// The referenced type is exactly `Context.CallableData`, required by `ICallableContext`.
    /// The `[nonmutating] ref` accessor leaves the view unchanged and permits mutation of the
    /// originating callable data through an `in` parameter or immutable local copy.
    public property Context.CallableData data
    {
        __intrinsic_op(portableRTGetCallableData)
        [nonmutating]
        ref;
    }

    /// Reads the host-provided data in the selected shader-binding-table record.
    /// The compiler maps this accessor to the selected invocation's record-data binding.
    ///
    /// @par Constraints
    /// Read-only value of exactly `Context.Record`. The getter returns application data by value;
    /// copied views read the same bound record. The host-provided record must match that type's
    /// target layout; `void` selects a context with no application record data.
    public property Context.Record record
    {
        __intrinsic_op(portableRTGetRecord)
        get;
    }

    /// Reads the originating ray-dispatch coordinates.
    ///
    /// @par Constraints
    /// Read-only, zero-based coordinates within `dispatchRaysDimensions`. Nested hit or callable
    /// dispatch retains the originating launch coordinates.
    public property uint3 dispatchRaysIndex
    {
        get { return getStageInputDispatchRaysIndex(); }
    }

    /// Reads the dimensions of the originating ray dispatch.
    ///
    /// @par Constraints
    /// Read-only `uint3` describing the launch dimensions used by `dispatchRaysIndex`.
    public property uint3 dispatchRaysDimensions
    {
        get { return getStageInputDispatchRaysDimensions(); }
    }
}

} // namespace rt
