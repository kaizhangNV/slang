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
/// The current Metal lowering uses one table for each dispatch role, shared by all payload types
/// in the schema. The block-relative argument-buffer IDs are:
///
/// @verbatim
///     Field                   Resource kind                         Metal [[id]]
///     intersectionFunctions   intersection-function table           0
///     missFunctions           miss visible-function table           1
///     closestHitFunctions     closest-hit visible-function table    2
///     callableFunctions       callable visible-function table       3
///     records                 record-data buffer                    4
///     anyHitFunctions         any-hit visible-function table        5
///     intersectionShaders     intersection visible-function table   6
/// @endverbatim
///
/// The record buffer begins with three uint byte offsets to the hit, miss, and callable sections,
/// respectively. Each record has a 16-byte header: a uint functionIndex and 12 reserved bytes,
/// followed by its Context.Record data. A section's uniform byte stride is
/// `alignUp(16 + largest Record size in that section, 16)`.
///
/// Metal preserves the runtime SBT record index; its record header replaces the native shader
/// identifier with a function index, while its application data still supplies `input.record`:
///
/// @verbatim
///     hitRecord = byteBase(records) + hitSectionByteOffset + recordIndex * hitStride
///
///     D3D/Vulkan hit record               Metal hit record
///     shader identifier / group handle -> header.functionIndex = f
///     application record data          -> Context.Record bytes after the 16-byte header
///
///     Candidate hit:
///       intersectionFunctions[primitiveSlot] -> compute recordIndex -> read f
///       -> select hit group f -> intersectionShaders[f] / anyHitFunctions[f], when present
///     Committed hit:
///       compute recordIndex -> read f -> closestHitFunctions[f], when present
/// @endverbatim
///
/// In this current mapping, the IFT slot selects a generated primitive dispatcher. That dispatcher
/// selects the hit group's candidate logic using f, the index in Schema.HitGroups. Several records
/// can share f. Miss and callable records similarly select entries in their corresponding VFTs
/// through their headers. Instance contributions come from the acceleration structure's metadata.
/// The planned IFT-array and native-stage-callback migration will change this Metal binding layout;
/// it is not implemented by the fixed-count API change alone.
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
