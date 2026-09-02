implementing raytracing;

// RayTracer is the value-level connection between a trace site and its compile-time program
// layout. Constrained extensions enumerate the valid acceleration-structure and motion
// combinations, so selecting a trace context also selects exactly one trace operation contract.

namespace rt
{

/// Issues ray-tracing and callable dispatches using the logical shader binding table (SBT)
/// described by `ProgramLayout`.
///
/// Construct this zero-state helper in shader code and invoke the overload enabled by the program
/// layout's trace context. The layout connects each dispatch to its reachable stage types and to a
/// matching `TraceProgramDescriptor`.
/// @category raytracing
public struct RayTracer<ProgramLayout>
    where ProgramLayout : ITraceProgramLayout
{}

// The portable two-level, no-motion case can directly express the trace with Slang's existing
// TraceRay operation on targets that already provide native SBT dispatch.
public extension<ProgramLayout> RayTracer<ProgramLayout>
    where ProgramLayout : ITraceProgramLayout
    where ProgramLayout.TraceContext.AccelerationStructure == AccelerationStructure
    where ProgramLayout.TraceContext.Motion == NoMotion
{
    /// Traces a ray through a portable two-level acceleration structure without motion.
    /// @param desc The ray, traversal flags, visibility mask, and logical SBT indices.
    /// @param accelerationStructure The scene selected by the program's trace context.
    /// @param descriptor The bound program resources for `ProgramLayout`.
    /// @param payload The value shared with the selected hit or miss stages and returned to the
    /// caller after they finish.
    [ForceInline]
    [require(structural_raytracing_trace)]
    public void trace(
        RayTraversalDesc desc,
        ProgramLayout.TraceContext.AccelerationStructure accelerationStructure,
        TraceProgramDescriptor<ProgramLayout> descriptor,
        inout ProgramLayout.TraceContext.Payload payload)
    {
        __target_switch
        {
        case metal:
            // The compiler consumes this call as a structural trace operation before emission.
            break;
        default:
            TraceRay(
                accelerationStructure,
                desc.rayFlags,
                desc.instanceMask,
                desc.sbtOffset,
                desc.sbtStride,
                desc.missIndex,
                desc.ray,
                payload);
        }
    }
}

// Every enabled motion mode has the same source trace operation. The concrete motion type remains
// available to target lowering, which uses it to distinguish primitive motion, instance motion,
// or both when the target ABI requires that distinction.
public extension<ProgramLayout> RayTracer<ProgramLayout>
    where ProgramLayout : ITraceProgramLayout
    where ProgramLayout.TraceContext.AccelerationStructure == AccelerationStructure
    where ProgramLayout.TraceContext.Motion : IEnabledRayMotion
{
    /// Traces a ray through a portable two-level acceleration structure with motion.
    ///
    /// `desc.time` selects the motion sample used by traversal.
    /// @param desc The ray, motion time, traversal flags, visibility mask, and logical SBT indices.
    /// @param accelerationStructure The scene selected by the program's trace context.
    /// @param descriptor The bound program resources for `ProgramLayout`.
    /// @param payload The value shared with the selected hit or miss stages and returned to the
    /// caller after they finish.
    [ForceInline]
    [require(structural_raytracing_trace, structural_raytracing_motion)]
    public void trace(
        RayTraversalDesc desc,
        ProgramLayout.TraceContext.AccelerationStructure accelerationStructure,
        TraceProgramDescriptor<ProgramLayout> descriptor,
        inout ProgramLayout.TraceContext.Payload payload)
    {
        __target_switch
        {
        case metal:
            // The compiler consumes this call as a structural trace operation before emission.
            break;
        case cuda:
        case glsl:
        case spirv:
            TraceMotionRay(
                accelerationStructure,
                desc.rayFlags,
                desc.instanceMask,
                desc.sbtOffset,
                desc.sbtStride,
                desc.missIndex,
                desc.ray,
                desc.time,
                payload);
        }
    }
}

// Multi-level overloads carry the hierarchy depth in both the context and the resource parameter.
// The equality constraint ensures a descriptor cannot be traced against a resource with a
// different topology.
public extension<ProgramLayout> RayTracer<ProgramLayout>
    where ProgramLayout : ITraceProgramLayout
    where ProgramLayout.TraceContext.Motion == NoMotion
{
    /// Traces a ray through a multi-level acceleration structure without motion.
    /// @param desc The ray, traversal flags, visibility mask, and logical SBT indices.
    /// @param accelerationStructure The hierarchy whose maximum depth is `maxLevelCount`.
    /// @param descriptor The bound program resources for `ProgramLayout`.
    /// @param payload The value shared with the selected hit or miss stages and returned to the
    /// caller after they finish.
    [ForceInline]
    [require(structural_raytracing_trace, metal)]
    public void trace<let maxLevelCount : int>(
        RayTraversalDesc desc,
        MultiLevelAccelerationStructure<maxLevelCount> accelerationStructure,
        TraceProgramDescriptor<ProgramLayout> descriptor,
        inout ProgramLayout.TraceContext.Payload payload)
        where ProgramLayout.TraceContext.AccelerationStructure ==
            MultiLevelAccelerationStructure<maxLevelCount>
    {
        // The compiler consumes this call as a structural trace operation before emission.
    }
}

public extension<ProgramLayout> RayTracer<ProgramLayout>
    where ProgramLayout : ITraceProgramLayout
    where ProgramLayout.TraceContext.Motion : IEnabledRayMotion
{
    /// Traces a ray through a multi-level acceleration structure with motion.
    ///
    /// `desc.time` selects the motion sample used by traversal.
    /// @param desc The ray, motion time, traversal flags, visibility mask, and logical SBT indices.
    /// @param accelerationStructure The hierarchy whose maximum depth is `maxLevelCount`.
    /// @param descriptor The bound program resources for `ProgramLayout`.
    /// @param payload The value shared with the selected hit or miss stages and returned to the
    /// caller after they finish.
    [ForceInline]
    [require(structural_raytracing_trace, structural_raytracing_motion)]
    public void trace<let maxLevelCount : int>(
        RayTraversalDesc desc,
        MultiLevelAccelerationStructure<maxLevelCount> accelerationStructure,
        TraceProgramDescriptor<ProgramLayout> descriptor,
        inout ProgramLayout.TraceContext.Payload payload)
        where ProgramLayout.TraceContext.AccelerationStructure ==
            MultiLevelAccelerationStructure<maxLevelCount>
    {
        // The compiler consumes this call as a structural trace operation before emission.
    }
}

// Callable dispatch shares the descriptor's callable section but not the ray traversal arguments,
// so it is available for every program layout independently of topology and motion mode.
public extension<ProgramLayout> RayTracer<ProgramLayout>
    where ProgramLayout : ITraceProgramLayout
{
    /// Invokes the callable shader at `callableIndex` using `CallableContext`.
    /// @param callableIndex The logical slot in the callable section of `ProgramLayout`.
    /// @param descriptor The bound program resources for `ProgramLayout`.
    /// @param data The value passed to the callable shader and returned to the caller after it
    /// finishes.
    [ForceInline]
    [require(structural_raytracing_call_shader)]
    public void callShader<CallableContext>(
        uint callableIndex,
        TraceProgramDescriptor<ProgramLayout> descriptor,
        inout CallableContext.CallableData data)
        where CallableContext : ICallableGroupContext
        where CallableContext.TraceContext == ProgramLayout.TraceContext
    {
        __target_switch
        {
        case metal:
            // The compiler consumes this call as structural callable dispatch before emission.
            break;
        default:
            CallShader(callableIndex, data);
        }
    }
}

} // namespace rt
