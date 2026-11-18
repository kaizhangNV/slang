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
/// `Schema` supplies the trace context and the available hit-group, miss, and callable shader
/// types. The host chooses how many records to allocate and which shader each record uses;
/// several records can use the same shader. Schema-list lengths do not determine the table width.
///
/// Consider `TraceProgramDescriptor<SceneSchema, 2>` with two geometries and two ray types:
/// radiance (`desc.rayType == 0`) and shadow (`desc.rayType == 1`). The default PerGeometry layout
/// gives each geometry a row of two records, even if those ray types use different payload types:
///
/// @verbatim
///     Hit records, instanceContribution = 0, rayTypeCount = 2
///     +--------------+----------------------+----------------------+
///     | Geometry row | Radiance, rayType=0  | Shadow, rayType=1    |
///     +--------------+----------------------+----------------------+
///     | 0            | hit record 0         | hit record 1         |
///     | 1            | hit record 2         | hit record 3         |
///     +--------------+----------------------+----------------------+
/// @endverbatim
///
/// With `TraceProgramDescriptor<SceneSchema, 2, HitRecordLayout.SharedAcrossGeometries>`, both
/// geometries instead use records 0 and 1. The count is still two; only the geometry multiplier
/// changes to zero. Other instances can select their own records through instanceContribution.
///
/// @verbatim
///     geometryMultiplier = hitRecordLayout == HitRecordLayout.PerGeometry ? rayTypeCount : 0
///     recordIndex = instanceContribution + geometryIndex * geometryMultiplier + desc.rayType
///
///     Miss records: desc.missIndex selects the miss record independently.
///     Callable records: callShader's callableIndex selects the callable record independently.
/// @endverbatim
///
/// `rayTypeCount` counts columns of records. It is fixed when the descriptor type is specialized;
/// each trace supplies its column through `desc.rayType`. Record byte stride is separate and must
/// follow the target's Record layout requirements. The caller supplies valid record indices;
/// no runtime bounds check or wrapping is added.
///
/// The descriptor has the following target representations:
/// - D3D/HLSL: no shader resource or register. The host binds the native pipeline and SBT.
/// - Vulkan/GLSL/SPIR-V: no descriptor-set binding. The host supplies the native SBT regions.
/// - CUDA/OptiX: no shader resource. The host supplies the native OptiX SBT.
/// - Metal: a generated ParameterBlock-like argument buffer containing function tables and a
///   record-data buffer. These are lowering artifacts, not fields available to Slang source.
///
/// The Metal argument buffer has six eight-byte fields, with eight-byte alignment and a total
/// size of 48 bytes. These block-relative argument IDs and byte offsets stay fixed for every
/// Schema, rayTypeCount, and hitRecordLayout, including when a table is unused:
///
/// @verbatim
///     Field                              Resource kind                 ID   Byte offset
///     tables                             pointer to IFT entries         0    0
///     missFunctions                      miss VFT                       1    8
///     closestHitFunctions                closest-hit VFT                2   16
///     callableFunctions                  callable VFT                   3   24
///     records                            pointer to record directory     4   32
///     candidateResources.anyHitFunctions procedural-any-hit VFT         5   40
/// @endverbatim
///
/// `candidateResources` is an inline eight-byte subobject, not another buffer allocation. The
/// compiler can pass its address to a callback that needs the procedural AnyHit table.
/// `tables` points to rayTypeCount contiguous IFT entries, each containing one eight-byte Metal
/// IFT gpuResourceID. A trace selects tables[desc.rayType]; SharedAcrossGeometries also has
/// rayTypeCount tables. There is no runtime count field or intersection-shader VFT.
///
/// The host writes buffer GPU addresses into pointer fields and table gpuResourceID values into
/// VFT fields; these are never CPU object pointers. Bind the buffer at the descriptor's reflected
/// outer binding and keep all indirectly referenced resources resident.
/// Increasing rayTypeCount adds IFT entries and table objects, not top-level shader bindings.
///
/// The record buffer begins with three uint byte offsets to the hit, miss, and callable sections,
/// respectively. Section starts are 16-byte aligned. Each record has a 16-byte header followed by
/// its Context.Record data. A section's uniform byte stride is
/// `alignUp(16 + largest Record size in that section, 16)`, with a minimum of 16 bytes. Use Metal's
/// structured-buffer layout for Record sizes and offsets, including every Record type in Schema.
///
/// The host assigns VFT slots independently of schema-list order and SBT record positions. Several
/// records may reuse one slot. Hit records store separate closest-hit and procedural-any-hit slots;
/// 0xffffffff means the corresponding shader is absent. Triangle/curve records always use
/// 0xffffffff for anyHitIndex because native traversal invokes their AnyHit callback directly.
///
/// @verbatim
///     Hit header:      [closestHitIndex][anyHitIndex][reserved = 0][reserved = 0]
///     Miss header:     [missVFTIndex]               [three reserved uints = 0]
///     Callable header: [callableVFTIndex]           [three reserved uints = 0]
///
///     hitRecord = byteBase(records) + hitSectionByteOffset + recordIndex * hitStride
///     input.record = Context.Record bytes after hitRecord's 16-byte header
///
///     Candidate hit:
///       tables[desc.rayType][nativeInstanceOffset + nativeGeometryOffset]
///         -> Intersection for procedural geometry, or AnyHit for triangles/curves
///       Procedural reportHit -> candidateResources.anyHitFunctions[anyHitIndex], if present
///     Committed hit:
///       compute recordIndex -> read closestHitIndex -> call that closest-hit VFT slot, if present
/// @endverbatim
///
/// The host populates each IFT with the selected stages' native function handles. For triangle or
/// curve groups without AnyHit, it uses Metal's matching opaque entry. Procedural Intersection
/// remains required even without AnyHit. Sharing a stage between groups reuses its function handle.
/// The native IFT offsets must select code compatible with the computed record; their numerical
/// values need not equal recordIndex. Instance contributions come from the separately bound
/// acceleration structure's metadata, which the host keeps consistent with its IFT mapping.
/// Set the pipeline's maxCallStackDepth for the longest indirect call chain: procedural
/// Intersection calling AnyHit requires at least two levels; nested traces/callables need more.
///
/// Compile related traces and native callbacks with the same TraceContext and target capabilities.
/// Their shared tag profile includes triangle_data, curve_data when metallib_3_1 is enabled, and
/// world_space_data when instancing is enabled. TraceContext supplies depth and motion; explicit
/// metal_raytracing_extended_limits enables extended_limits. Schema/body-based tag reduction is
/// not implemented. Ordinary global resources must also share the same component layout.
///
/// D3D, Vulkan, and OptiX traces require at most 15 ray types for PerGeometry or 16 for shared
/// records because their native selectors use four bits. Classic Metal IFT tracing has no such
/// four-bit restriction. These target limits do not restrict a descriptor's source declaration.
///
/// @tparam Schema A type conforming to ITraceProgramSchema. Its TraceContext selects acceleration
/// structure and motion types; HitGroups, MissShaders, and CallableShaders name available shaders.
/// @tparam rayTypeCount Required positive integer giving the number of ray-type columns in the
/// hit-record table. It has no default and is independent of the number of shaders in Schema.
/// @tparam hitRecordLayout Defaults to PerGeometry, which uses rayTypeCount as the geometry
/// multiplier. SharedAcrossGeometries uses zero so every geometry in an instance shares its row.
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
