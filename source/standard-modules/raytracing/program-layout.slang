implementing raytracing;

// A structural program layout is a compile-time description of the three logical shader binding
// table (SBT) sections.
// Individual group declarations name a slot, context, and stage implementation; group-list types
// collect those declarations; and ITraceProgramLayout joins the lists under one trace context.
// These declarations carry metadata only and are not shader runtime values.

namespace rt
{

/// Identifies a logical record index within one SBT section.
///
/// Slot indices are explicit and need not be contiguous. The containing hit, miss, or callable
/// group determines which SBT section the index addresses.
/// @category raytracing
public interface IShaderGroupSlot
{
    /// The zero-based logical record index.
    static const int index;
}

/// Defines a logical shader-group slot whose index is `indexValue`.
///
/// Prefer the section-specific `HitGroupSlot`, `MissSlot`, or `CallableSlot` aliases in group
/// declarations so the purpose of the index is clear.
/// @category raytracing
public struct ShaderGroupSlot<let indexValue : int> : IShaderGroupSlot
{
    /// The zero-based logical record index supplied by `indexValue`.
    public static const int index = indexValue;
}

/// Identifies a logical hit-group record index.
/// @category raytracing
public typealias HitGroupSlot<let indexValue : int> = ShaderGroupSlot<indexValue>;

/// Identifies a logical miss-group record index.
/// @category raytracing
public typealias MissSlot<let indexValue : int> = ShaderGroupSlot<indexValue>;

/// Identifies a logical callable-group record index.
/// @category raytracing
public typealias CallableSlot<let indexValue : int> = ShaderGroupSlot<indexValue>;

/// Describes one logical hit-group record and the stages that process it.
///
/// Implement this interface with associated-type aliases. Use `NoClosestHit` or `NoAnyHit` when the
/// group omits that behavior. Use `NoIntersection` when the selected primitive's intersections are
/// supplied by traversal. These placeholders do not declare executable stage logic.
/// @category raytracing
public interface IHitGroup
{
    /// The logical index of this record within the hit-group SBT section.
    associatedtype Slot : IShaderGroupSlot;

    /// The primitive, trace context, and local record type shared by this group's stages.
    associatedtype Context : IHitContext;

    /// The closest-hit stage, or `NoClosestHit<Context>` when no closest-hit behavior is required.
    associatedtype ClosestHit : IClosestHitShader<Context>;

    /// The any-hit stage, or `NoAnyHit<Context>` when no any-hit filtering is required.
    associatedtype AnyHit : IAnyHitShader<Context>;

    /// The procedural intersection stage, or `NoIntersection<Context>` when traversal supplies the
    /// primitive intersection.
    associatedtype Intersection : IIntersectionStage<Context>;

}

/// Describes one logical miss-group record and its miss stage.
///
/// Implement this interface with associated-type aliases.
/// @category raytracing
public interface IMissGroup
{
    /// The logical index of this record within the miss SBT section.
    associatedtype Slot : IShaderGroupSlot;

    /// The trace context and local record type used by the miss stage.
    associatedtype Context : IMissGroupContext;

    /// The miss stage selected by this record.
    associatedtype Miss : IMissShader<Context>;
}

/// Describes one logical callable-group record and its callable stage.
///
/// Implement this interface with associated-type aliases.
/// @category raytracing
public interface ICallableGroup
{
    /// The logical index of this record within the callable SBT section.
    associatedtype Slot : IShaderGroupSlot;

    /// The call data, trace context, and local record type used by the callable stage.
    associatedtype Context : ICallableGroupContext;

    /// The callable stage selected by this record.
    associatedtype Callable : ICallableShader<Context>;
}

/// Classifies a finite list of hit groups that all use `TraceContext`.
///
/// This interface is sealed. Construct a list with `HitGroupList`, or use `NoHitGroups` when the
/// program has no hit groups.
/// @category raytracing
[sealed]
public interface IHitGroupList<TraceContext>
    where TraceContext : ITraceContext
{
    /// The number of group types in the list, independent of their explicit slot indices.
    static const int count;
}

/// Classifies a finite list of miss groups that all use `TraceContext`.
///
/// This interface is sealed. Construct a list with `MissGroupList`, or use `NoMissGroups` when the
/// program has no miss groups.
/// @category raytracing
[sealed]
public interface IMissGroupList<TraceContext>
    where TraceContext : ITraceContext
{
    /// The number of group types in the list, independent of their explicit slot indices.
    static const int count;
}

/// Classifies a finite list of callable groups that all use `TraceContext`.
///
/// This interface is sealed. Construct a list with `CallableGroupList`, or use
/// `NoCallableGroups` when the program has no callable groups.
/// @category raytracing
[sealed]
public interface ICallableGroupList<TraceContext>
    where TraceContext : ITraceContext
{
    /// The number of group types in the list, independent of their explicit slot indices.
    static const int count;
}

/// Collects the hit-group declarations that belong to one trace context.
///
/// `Group` is a variadic type pack. Every group's context is constrained to refer to the supplied
/// `TraceContext`; each group retains its own explicit hit-group slot.
/// @category raytracing
public struct HitGroupList<TraceContext, each Group> : IHitGroupList<TraceContext>
    where TraceContext : ITraceContext
    where Group : IHitGroup
    where expand each Group.Context.TraceContext == TraceContext
{
    /// The number of hit-group types in `Group`.
    public static const int count = countof(Group);
}

/// Collects the miss-group declarations that belong to one trace context.
///
/// `Group` is a variadic type pack. Every group's context is constrained to refer to the supplied
/// `TraceContext`; each group retains its own explicit miss-group slot.
/// @category raytracing
public struct MissGroupList<TraceContext, each Group> : IMissGroupList<TraceContext>
    where TraceContext : ITraceContext
    where Group : IMissGroup
    where expand each Group.Context.TraceContext == TraceContext
{
    /// The number of miss-group types in `Group`.
    public static const int count = countof(Group);
}

/// Collects the callable-group declarations that belong to one trace context.
///
/// `Group` is a variadic type pack. Every group's context is constrained to refer to the supplied
/// `TraceContext`; each group retains its own explicit callable-group slot.
/// @category raytracing
public struct CallableGroupList<TraceContext, each Group> : ICallableGroupList<TraceContext>
    where TraceContext : ITraceContext
    where Group : ICallableGroup
    where expand each Group.Context.TraceContext == TraceContext
{
    /// The number of callable-group types in `Group`.
    public static const int count = countof(Group);
}

/// Represents an empty hit-group section for `TraceContext`.
/// @category raytracing
public struct NoHitGroups<TraceContext> : IHitGroupList<TraceContext>
    where TraceContext : ITraceContext
{
    /// The number of hit groups in this list.
    public static const int count = 0;
}

/// Represents an empty miss-group section for `TraceContext`.
/// @category raytracing
public struct NoMissGroups<TraceContext> : IMissGroupList<TraceContext>
    where TraceContext : ITraceContext
{
    /// The number of miss groups in this list.
    public static const int count = 0;
}

/// Represents an empty callable-group section for `TraceContext`.
/// @category raytracing
public struct NoCallableGroups<TraceContext> : ICallableGroupList<TraceContext>
    where TraceContext : ITraceContext
{
    /// The number of callable groups in this list.
    public static const int count = 0;
}

/// Describes the complete logical SBT layout used by a structural ray-tracing program.
///
/// Implement this interface with associated-type aliases. The resulting type is the source of
/// truth shared by `RayTracer`, `TraceProgramDescriptor`, and host reflection: it fixes one trace
/// context and the hit, miss, and callable records reachable from a trace.
/// @category raytracing
public interface ITraceProgramLayout
{
    /// The payload, acceleration-structure topology, and motion mode shared by the program.
    associatedtype TraceContext : ITraceContext;

    /// The program's logical miss-group records.
    associatedtype MissGroups : IMissGroupList<TraceContext>;

    /// The program's logical hit-group records.
    associatedtype HitGroups : IHitGroupList<TraceContext>;

    /// The program's logical callable-group records.
    associatedtype CallableGroups : ICallableGroupList<TraceContext>;
}

} // namespace rt
