implementing raytracing;

// A program layout describes logical shader records at the type level; a trace program descriptor
// is the corresponding bindable value.

namespace rt
{

/// A bindable ray-tracing program whose logical shader binding table (SBT) schema is described by
/// `ProgramLayout`.
///
/// Declare this opaque resource as a shader parameter or parameter-block field, populate the
/// bindings reported by reflection, and pass it to `RayTracer.trace` or `RayTracer.callShader`.
/// The type exposes no fields or public constructor; valid values are supplied through host
/// binding. On targets with a native SBT, this value has no physical shader binding. Metal lowering
/// materializes the function tables and record-data buffer required by the selected program layout.
/// @category raytracing
__magic_type(TraceProgramDescriptorType)
__intrinsic_type(TraceProgramDescriptor)
public struct TraceProgramDescriptor<ProgramLayout>
    where ProgramLayout : ITraceProgramLayout
{
    // An explicit constructor suppresses the public member-initializer constructor that an empty
    // struct would otherwise receive. Its required private argument also prevents containing
    // aggregates from trying to call it as a default initializer.
    private __init(int unavailable) {}
}

} // namespace rt
