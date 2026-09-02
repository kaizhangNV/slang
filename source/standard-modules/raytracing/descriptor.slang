implementing raytracing;

// A program layout describes logical shader records at the type level; a trace program descriptor
// is the corresponding bindable value. Its private resource block gives parameter binding and
// reflection one stable container while later target lowering chooses the concrete contents.

namespace rt
{

// These placeholders reserve distinct resource roles until the concrete program layout and target
// are known. Keeping them in a ParameterBlock makes specialization compatible with ordinary
// parameter binding and reflection.
internal struct TraceProgramDescriptorResources<ProgramLayout>
    where ProgramLayout : ITraceProgramLayout
{
    // Candidate-intersection dispatch resources indexed by logical hit-group slot.
    internal ByteAddressBuffer intersectionFunctions;

    // Miss-stage dispatch resources indexed by logical miss-group slot.
    internal ByteAddressBuffer missFunctions;

    // Closest-hit dispatch resources indexed by logical hit-group slot.
    internal ByteAddressBuffer closestHitFunctions;

    // Callable-stage dispatch resources indexed by logical callable-group slot.
    internal ByteAddressBuffer callableFunctions;

    // Local shader-record data for the slots declared by ProgramLayout.
    internal ByteAddressBuffer records;
}

/// A bindable ray-tracing program whose logical shader binding table (SBT) schema is described by
/// `ProgramLayout`.
///
/// Declare this opaque resource as a shader parameter or parameter-block field, populate the
/// bindings reported by reflection, and pass it to `RayTracer.trace` or `RayTracer.callShader`.
/// Shader code cannot inspect its target-specific representation.
/// @category raytracing
public struct TraceProgramDescriptor<ProgramLayout>
    where ProgramLayout : ITraceProgramLayout
{
    // The descriptor remains a single opaque source-level value while this private block carries
    // the resource roles materialized by target specialization.
    internal ParameterBlock<TraceProgramDescriptorResources<ProgramLayout>> resources;
}

} // namespace rt
