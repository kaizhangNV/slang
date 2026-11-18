implementing portable;

namespace rt
{

/// Determines whether each geometry has its own row of hit records.
public enum HitRecordLayout
{
    /// Each geometry has rayTypeCount consecutive records, one per ray type.
    PerGeometry = 0,

    /// All geometries in an instance share its rayTypeCount records.
    SharedAcrossGeometries = 1,
}

/// Represents a host-bound ray-tracing program with a fixed number of ray types.
///
/// Declare this opaque resource as a shader parameter or parameter-block field and pass it to
/// `RayTracer.trace` or `RayTracer.callShader`. It exposes no fields or public constructor.
/// The host supplies its value and the records addressed by each trace.
///
/// With `TraceProgramDescriptor<SceneSchema, 2>`, each geometry has two hit records: for example,
/// radiance at `desc.rayType == 0` and shadow at `desc.rayType == 1`. The hit-record index is
/// `instanceContribution + geometryIndex * 2 + desc.rayType`. SharedAcrossGeometries makes the
/// geometry multiplier zero; there are still two ray types and two records per instance.
/// The caller supplies valid record indices. No runtime bounds check or wrapping is added.
///
/// D3D, Vulkan, and OptiX bind the native pipeline and SBT; this descriptor consumes no shader
/// resource binding. Metal binds one argument buffer containing function tables and records.
/// Record byte stride is determined separately from the schema's Record layouts.
/// D3D, Vulkan, and OptiX traces require at most 15 ray types for PerGeometry or 16 for shared
/// records because their native selectors use four bits. Classic Metal IFT tracing has no such
/// four-bit restriction. These target limits do not restrict a descriptor's source declaration.
///
/// @tparam Schema The trace context and shader lists describing the bound program.
/// @tparam rayTypeCount Positive number of ray types.
/// @tparam hitRecordLayout Whether each geometry has a distinct row or shares its instance's row.
/// @category raytracing
__magic_type(TraceProgramDescriptorType)
__intrinsic_type(TraceProgramDescriptor)
public struct TraceProgramDescriptor<
    Schema,
    let rayTypeCount : int,
    let hitRecordLayout : HitRecordLayout = HitRecordLayout.PerGeometry>
    where Schema : ITraceProgramSchema
{}

/// Declares a host-bound program whose geometry multiplier will be supplied at each trace.
///
/// This opaque type reserves the flexible descriptor API. Trace/callable overloads and target
/// code generation are not implemented yet; declaring the type does not enable Metal IFB support.
/// @tparam Schema The trace context and shader lists describing the bound program.
/// @category raytracing
__magic_type(FlexibleTraceProgramDescriptorType)
__intrinsic_type(FlexibleTraceProgramDescriptor)
public struct FlexibleTraceProgramDescriptor<Schema>
    where Schema : ITraceProgramSchema
{}

} // namespace rt
