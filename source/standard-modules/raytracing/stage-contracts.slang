implementing raytracing;

// Structural stages are declared as ordinary structs that implement one of the interfaces below.
// The interface fixes the stage kind and input context, allowing a program layout to name the
// implementation as a type. Traversal invokes `invoke`; application code does not call stage
// implementations directly.

namespace rt
{

/// Defines closest-hit behavior for the hit group described by `Context`.
///
/// Implement this interface on a struct and select that struct as `IHitGroup.ClosestHit`. The
/// struct name is also the entry-point name when the stage is compiled independently.
/// @category raytracing
[require(structural_raytracing_closesthit)]
public interface IClosestHitShader<Context>
    where Context : IHitContext
{
    /// Processes the closest accepted intersection after traversal finishes.
    /// @param input A storage-free view of the payload, hit data, and shader record.
    void invoke(ClosestHitInput<Context> input);
}

/// Defines any-hit filtering behavior for the hit group described by `Context`.
///
/// Implement this interface on a struct and select that struct as `IHitGroup.AnyHit`. The stage may
/// accept the current candidate, ignore it, or accept it and end traversal.
/// @category raytracing
[require(structural_raytracing_anyhit)]
public interface IAnyHitShader<Context>
    where Context : IHitContext
{
    /// Processes a candidate intersection before it can become the committed hit.
    /// @param input A storage-free view of the candidate, payload, and shader record.
    void invoke(AnyHitInput<Context> input);
}

// A hit group needs one associated-type constraint that accepts both a user intersection stage and
// the NoIntersection placeholder. Keeping this shared contract internal prevents applications from
// creating an unclassified third kind of intersection stage.
internal interface IIntersectionStage<Context>
    where Context : IHitContext
{
    // Both public alternatives have this callable shape, which lets group processing use the same
    // requirement witness without depending on which alternative was selected.
    void invoke(IntersectionInput<Context> input);
}

/// Defines procedural intersection behavior for the hit group described by `Context`.
///
/// This interface is available only when `Context.Primitive` is a custom intersection primitive.
/// Implement it on a struct, select that struct as `IHitGroup.Intersection`, and report every
/// candidate intersection through `IntersectionInput.reportHit`.
/// @category raytracing
[require(structural_raytracing_intersection)]
public interface IIntersectionShader<Context> : IIntersectionStage<Context>
    where Context : IHitContext
{
    __constraint Context.Primitive : ICustomIntersectionPrimitive;
}

/// Defines miss behavior for the miss group described by `Context`.
///
/// Implement this interface on a struct and select that struct as `IMissGroup.Miss`. The struct
/// name is also the entry-point name when the stage is compiled independently.
/// @category raytracing
[require(structural_raytracing_miss)]
public interface IMissShader<Context>
    where Context : IMissGroupContext
{
    /// Processes a ray for which traversal found no accepted intersection.
    /// @param input A storage-free view of the payload, ray data, and shader record.
    void invoke(MissInput<Context> input);
}

/// Defines callable-shader behavior for the callable group described by `Context`.
///
/// Implement this interface on a struct and select that struct as `ICallableGroup.Callable`. Calls
/// exchange a value of `Context.CallableData` with the implementation.
/// @category raytracing
[require(structural_raytracing_callable)]
public interface ICallableShader<Context>
    where Context : ICallableGroupContext
{
    /// Executes the callable shader selected by a `RayTracer.callShader` operation.
    /// @param input A storage-free view of the call data, dispatch coordinates, and shader record.
    void invoke(CallableInput<Context> input);
}

/// Selects a hit group with no closest-hit behavior.
///
/// Use this placeholder as `IHitGroup.ClosestHit`; it does not declare executable closest-hit
/// logic for the group.
/// @category raytracing
public struct NoClosestHit<Context> : IClosestHitShader<Context>
    where Context : IHitContext
{
    /// Provides the empty implementation required by the stage contract.
    public void invoke(ClosestHitInput<Context> input) {}
}

/// Selects a hit group with no any-hit filtering.
///
/// Use this placeholder as `IHitGroup.AnyHit`; it does not declare executable any-hit logic for the
/// group.
/// @category raytracing
public struct NoAnyHit<Context> : IAnyHitShader<Context>
    where Context : IHitContext
{
    /// Provides the empty implementation required by the stage contract.
    public void invoke(AnyHitInput<Context> input) {}
}

/// Selects a hit group whose intersections are supplied without a custom intersection stage.
///
/// Use this placeholder as `IHitGroup.Intersection` when traversal supplies the selected
/// primitive's intersections, as it does for built-in triangle and curve primitives.
/// @category raytracing
public struct NoIntersection<Context> : IIntersectionStage<Context>
    where Context : IHitContext
{
    /// Provides the empty implementation required by the internal stage contract.
    public void invoke(IntersectionInput<Context> input) {}
}

} // namespace rt
