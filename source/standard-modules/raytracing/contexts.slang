implementing raytracing;

// Context declarations connect the shared contract for one trace program to the specialized
// contract for each shader group. They are compile-time metadata: applications implement these
// interfaces with associated types, and stage inputs use those types to expose only the data that
// is valid for the selected payload, primitive, and record.

namespace rt
{

/// Defines the payload and traversal model shared by one structural ray-tracing program.
///
/// Every hit, miss, and callable group in an `ITraceProgramLayout` refers to the same trace
/// context. Implement this interface with associated-type aliases; a trace context is layout
/// metadata and is not constructed as a runtime value.
/// @category raytracing
public interface ITraceContext
{
    /// The application-defined value passed to `RayTracer.trace` and shared with reachable hit
    /// and miss stages.
    associatedtype Payload;

    /// The acceleration-structure topology accepted by this trace program.
    associatedtype AccelerationStructure : IAccelerationStructure;

    /// The motion mode used by this trace program.
    associatedtype Motion : IRayMotion;
}

/// Defines the types shared by the closest-hit, any-hit, and intersection stages of one hit group.
///
/// The selected primitive controls which primitive-specific stage-input properties are available,
/// while the record type describes the local shader-record data visible to those stages. Implement
/// this interface with associated-type aliases; a hit context has no runtime representation.
/// @category raytracing
public interface IHitContext
{
    /// The trace-wide context to which this hit group belongs.
    associatedtype TraceContext : ITraceContext;

    /// The geometry kind processed by this hit group.
    associatedtype Primitive : IIntersectionPrimitive;

    /// The application-defined local data stored in this hit group's shader record. Use `void`
    /// when the group has no local record data.
    associatedtype Record;
}

/// Defines the trace-wide context and local shader-record type for one miss group.
///
/// Implement this interface with associated-type aliases; a miss-group context has no runtime
/// representation.
/// @category raytracing
public interface IMissGroupContext
{
    /// The trace-wide context to which this miss group belongs.
    associatedtype TraceContext : ITraceContext;

    /// The application-defined local data stored in this miss shader's record. Use `void` when the
    /// group has no local record data.
    associatedtype Record;
}

/// Defines the data types used by one callable-shader group.
///
/// `CallableData` is the value exchanged by a call, whereas `Record` is local data selected with
/// the callable shader. Implement this interface with associated-type aliases; a callable-group
/// context has no runtime representation.
/// @category raytracing
public interface ICallableGroupContext
{
    /// The trace-wide context to which this callable group belongs.
    associatedtype TraceContext : ITraceContext;

    /// The application-defined value passed to and returned from the callable shader.
    associatedtype CallableData;

    /// The application-defined local data stored in this callable shader's record. Use `void` when
    /// the group has no local record data.
    associatedtype Record;
}

} // namespace rt
