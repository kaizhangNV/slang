# Structural Dispatch Proposal: Revision For Dynamic Shader Binding Tables And Typed Payloads

Status: revision under iteration. This document records only what changes relative to
[PROPOSAL.md](PROPOSAL.md). Sections of the proposal that are not named here are unchanged. The
design rationale for the shader-binding-table part of the change is in
[DYNAMIC_SBT_DESIGN.md](DYNAMIC_SBT_DESIGN.md). The payload model, including exactly what is and is
not checked, is spelled out in
[PAYLOAD_SAFE_TRACE_VARIANTS.md](PAYLOAD_SAFE_TRACE_VARIANTS.md).

Two documents are cited for "what the text says today": PROPOSAL.md for the design, and the draft
implementation (module source `program-layout.slang`, the Metal lowering, the Cornell demo, the
Metal test host) for what was actually built from it. Where they differ, both are named.

## Catalog

- [0. Summary Of Revisions](#0-summary-of-revisions)
- [1. What Is Being Fixed](#1-what-is-being-fixed)
- [2. Revised Design Principles](#2-revised-design-principles)
- [3. Revised Contracts](#3-revised-contracts)
- [4. Revisions To PROPOSAL Section 2.2.2: Metal Lowering Of The Descriptor](#4-revisions-to-proposal-section-222-metal-lowering-of-the-descriptor)
- [5. Revisions To PROPOSAL Section 2.3.1: Metal Candidate Functions](#5-revisions-to-proposal-section-231-metal-candidate-functions)
- [6. New: Open Sections](#6-new-open-sections)
- [7. Revisions To PROPOSAL Sections 1.3 And 2.1](#7-revisions-to-proposal-sections-13-and-21)
- [8. Revisions To PROPOSAL Section 3: Migration And Reflection](#8-revisions-to-proposal-section-3-migration-and-reflection)
- [9. Diagnostics](#9-diagnostics)
- [10. Unchanged Sections And Affected Companion Documents](#10-unchanged-sections-and-affected-companion-documents)
- [11. Revised Open Design Questions](#11-revised-open-design-questions)

## 0. Summary Of Revisions

| PROPOSAL section | Was | Now | Why |
| --- | --- | --- | --- |
| 2.1 step 2, 2.2.1, the examples in 2.2.3, 3.1, 3.2 | Every group declares a compile-time `Slot`; the layout is a table of record indices | Nothing has a slot; the schema is the *set* of entries (hit groups, miss shaders, callable shaders) a trace can reach; the compiler assigns each entry a **function index** within its section, per payload for hit and miss | Record indices are host data on every native API; the static table cannot express N records over M groups |
| 2.2.1, 2.2.3, payload model | One payload per trace context, so one payload per layout; a second payload type requires a second layout and a second descriptor | The payload moves to the hit and miss **contexts**; `trace` is generic on the payload it passes and the compiler checks that some hit group or miss shader of the schema serves it; `sbtOffset`, `sbtStride`, and `missIndex` stay runtime fields | One SBT serves several payloads on every native API; the layout was standing in for the payload. What the compiler can and cannot check is stated honestly in the payload model, Section 5 |
| 2.2.3, payload model | Every `trace` passes a payload value | `trace(desc, as, descriptor)`, without a payload argument, for entries whose payload type is **empty** (a struct with no data members); the module defines no such type, a schema may contain at most one, and no value of it can be written in user code (not as a `trace` argument, not as a stage's `input.payload`) | A value that exists only to be ignored is noise at the most common call site; the type still partitions the schema and selects the Metal tables |
| 2.2.1 | Section lists take the trace context as a generic argument and constrain every entry with `where expand each Group.Context.TraceContext == TraceContext` | Lists take only their entries; the compiler checks trace-context agreement when it canonicalizes a schema | The variadic constraint is silently unenforced today and cannot reach linked entries; one mechanism instead of two |
| 2.2.1, 3.3, reflection | `ITraceProgramLayout`, the generic parameter `ProgramLayout`, `SlangReflectionTraceProgramLayout` | `ITraceProgramSchema`, `Schema`, `SlangReflectionTraceProgramSchema`; "layout" is kept for the bytes the host writes (record layout, record stride) | Without slots the type declares a set of entry types, not positions. In Slang a layout is what the compiler computes from a declaration (`TypeLayoutReflection`, `slang::ProgramLayout`), and `slang::ProgramLayout` already names the reflection object every host obtains from `getLayout()` |
| 2.2.1, 2.2.3 | `IHitContext`, `IMissGroupContext`, and `ICallableGroupContext` each repeat the trace context and the record type | One hierarchy: `IStageContext { TraceContext; Record }`, `IPayloadContext : IStageContext { Payload }`, `IHitContext : IPayloadContext { Primitive }`, `ICallableContext : IStageContext { CallableData }`; a miss shader may name a hit context | The repeated members were one fact spelled three times, and a miss that shares the hit record needed a second declaration for nothing |
| 2.2.1 | `IMissGroup` and `ICallableGroup` wrap one shader with a context; every stage contract takes its context as a generic parameter | Stage contracts expose `Context` as an associated type; `IHitGroup` is the only group and requires its three stages to share its context; the miss and callable sections list shaders directly | A miss or callable record binds exactly one shader, so the wrapper only restated a context the shader's conformance had already fixed; "group" survives where it is native |
| 2.2.1 | `input.record` on Metal is loaded "from the descriptor's generated data buffer before dispatching the logical stage"; the draft implementation realizes this by reading the record inside each generated stage through the group's compile-time slot | The trace dispatch resolves the record and passes a pointer; `input.record` means "per SBT record" on all targets | Today Metal reads per group, D3D/Vulkan per record; the two disagree |
| 2.2.2.1 | One visible-function-table entry per record; a four-word buffer header of table offsets; records carry no identity of their own; host maps `metalIFTIndex` to `logicalHitSlot` 1:1 | Tables hold one function per hit group and per miss shader of each payload, and one per callable shader; the record buffer is a real SBT: a four-word header of section offsets, then fixed-stride records whose first word is the function index; no IFT mapping constraint | Tables become fixed per pipeline; scene changes are buffer writes |
| 2.2.2.2, 3.3 Pattern B | Future IFB text indexes the closest-hit table by `logicalHitSlot` and installs by `hit.slot` | Same future item, indexed by the record's function index; the function buffer holds one candidate handle per record | Follows from the above |
| 2.3.1 | At most one generated Metal candidate function per hit group, installed by the host at a chosen IFT index when non-null | One generated candidate **dispatcher per primitive kind per payload** at fixed IFT indices 0/1/2, selecting the group from the record; per-group bodies become arms | Metal's IFT index has no ray-type term, so a per-group entry cannot follow the record |
| new | Entry set is closed in the ray-generation module | A section may be **open**: linked entries (hit groups or shaders) conforming to a tag interface join it at Slang link time | Separately compiled material modules without a declared target ABI |
| 1.3, 2.2.1 | Explicit layout intrinsics keep the SBT schema "finite, reviewable, and directly reflectable" from source | True for closed sections; an open section is finite and reflectable once the program is linked | Consequence of open sections |
| 3.1 to 3.3 | Hosts populate each table and place each record at the group's reflected slot | Hosts build one table entry per hit group or shader at its function index and write as many records as the scene needs | Follows from the above |
| diagnostics | `InvalidStructuralRayTracingGroupSlot`, `DuplicateStructuralRayTracingGroupSlot` | Duplicate entry, trace-context mismatch, payload not served, open-tag check, plain-data checks | Slot checks lose their object; hit-stage context agreement is expressed by `IHitGroup` constraints |

Everything else in PROPOSAL.md stands: what each stage may do and read (a stage struct now names its
context with `typealias Context` and the `payload` input property changes type), primitives, the
descriptor abstraction, `callShader`, tag-inference sources, topology, motion, `RayTraversalDesc`, and the
`reportHit` accumulator semantics. Section 10 lists exactly which passages change.

## 1. What Is Being Fixed

### 1.1 The layout was a compile-time table

`ITraceProgramLayout` was a compile-time table. Each group's `Slot` type (`IShaderGroupSlot`,
spelled `HitGroupSlot<N>` / `MissSlot<N>` / `CallableSlot<N>` in the examples) was defined as "a
record index within the corresponding SBT section" (PROPOSAL 2.2.1). Real shader binding tables are
decided by the host at runtime. The common engine shape is one record per `(instance, geometry,
rayType)`, thousands of records naming a handful of compiled groups and carrying per-record data. A
static slot per record cannot express that. Two facts from the draft implementation pin down where
the static assumption actually lives:

- On D3D12 and Vulkan the slot is never used for code generation. `RayTracer.trace` lowers to the
  native `TraceRay` with the runtime `sbtOffset`, `sbtStride`, and `missIndex`; stage structs become
  native entry points; `input.record` lowers to native shader-record data. The slot only feeds the
  negative/duplicate diagnostics and reflection.
- On Metal the trace dispatch already computes the hit record index at runtime
  (`instanceOffset[instance_id] + geometry_id * sbtStride + sbtOffset`) and indexes the
  visible-function table with it. But inside every generated stage function, `input.record` is read
  through the group's compile-time slot constant. One generated function installed at two table
  indices with two different records would read the same record twice.

So `input.record` means "per SBT record" on D3D12/Vulkan and "per group" on Metal today, the Metal
visible-function table has one entry per record, and the contract tells hosts to build exactly one
record per group. A hit-group slot is moreover only the physical record index when every instance's
contribution to the hit-group index is zero and `sbtStride * geometryIndex` is zero, so it was never
exact for the hit section even in the static case. Miss and callable slots were exact.

### 1.2 The layout was also standing in for the payload

`ITraceContext` carries one `Payload`, and a layout has one trace context, so a layout has one
payload type. A program with a radiance ray and an occlusion ray, the ordinary case, therefore
needs two layouts and two descriptors even though on D3D12 and Vulkan there is exactly one SBT and
one pipeline, with the two rays' records interleaved by `sbtStride`. Declaring a new layout to
change a payload type is the symptom: the payload belongs to the groups that read it, not to the
layout that lists them.

### 1.3 Cases the revision must cover

| Case | Meaning |
| --- | --- |
| A | Runtime record count and record-to-entry mapping over a closed entry set |
| B | Several payloads sharing one physical table, interleaved by the engine's ray-type convention |
| C | Open section: hit groups or shaders compiled in other modules and linked later |
| D | Per-record data varies while the entry is fixed |
| E | Instance and geometry contributions to the hit record index |
| F | Cheap per-frame table patching |

## 2. Revised Design Principles

### 2.1 Schema in shader, table on the host

The single-source-of-truth goal is kept for what the shader can be the source of: the **schema**.
No second schema contract is introduced: static and dynamic are properties of the host's table, not
of the schema type.

The shader owns the schema of the trace program: which entries exist, and for each hit group its
context, payload, stage composition, primitive, and record type, for each miss shader its context,
payload, and record type, for each callable shader its context, callable data, and record type; the
acceleration-structure topology and motion mode; the callable-data ABI. Reflection exposes that
schema together with the derived numbers a host otherwise hard-codes: payload sizes, native
attribute size, record strides and layouts, and on Metal the table shapes and candidate dispatchers.

The host owns the **instance** of the schema: how many records exist, which entry each record
names, the bytes in each record, the per-instance contributions in the acceleration structure, and
the interleave convention that `sbtOffset` and `sbtStride` follow. This is exactly the split D3D12
and Vulkan use, and the revision makes Metal follow it.

### 2.2 Function index

The bridge between schema and instance is the **function index**. Each section of a schema is a
set of *entries*: hit groups in the hit section, miss shaders in the miss section, callable shaders
in the callable section. An entry is what one SBT record binds to. The compiler assigns each entry
an ordinal within its section, partitioned by payload for the hit and miss sections (each payload's
hit groups and miss shaders are numbered from zero) and program-wide for the payload-less callable
section, and reflects it next to the entry's type name. A section's table holds exactly one
function per entry. The index is the Metal analogue of a DXR shader identifier. On Metal it is the
visible-function-table index the host installs the entry's function at and the value the host
writes into every record that uses the entry. On D3D12 and Vulkan it is only an enumeration order;
the native shader identifier remains the key. Like a shader identifier, it is valid for one compiled
program, so hosts key scene data by entry name and resolve name to index after each compile.

### 2.3 Payloads partition the schema

Every hit and miss context names its `Payload`. The set of payloads a schema serves is derived:

```text
payloads(Schema) = { G.Context.Payload | G ∈ Schema.HitGroups } ∪ { M.Context.Payload | M ∈ Schema.MissShaders }
                   over the linked program
```

`trace(desc, as, descriptor, inout P payload)` is generic on `P`, inferred from the argument, and
the compiler checks `P ∈ payloads(Schema)`. Within a payload, entries are partitioned and given
function indices; on Metal each payload gets its own tables.

Which record a ray reaches is not part of the schema:

```text
record = instanceContribution[instance] + geometryIndex * desc.sbtStride + desc.sbtOffset
          ─────────┬────────────           ──────┬──────   ───────┬──────   ───────┬──────
              host (TLAS)                   host (BLAS)     shader, runtime   shader, runtime
invoked stage = sbt[record].entry                                              host writes sbt[]
```

So the compiler establishes three facts about any program that compiles: the trace argument has a
concrete payload type `P`; some hit group or miss shader of the schema is typed on `P`; and every
stage is typed on its own context's payload, a hit group's three stages on one. It does not and
cannot establish that the *table* places a `P` entry where a `P` trace looks, because the table is
not in the program. That residual is one
host rule, reflected to the host; the payload model's Section 5 states it precisely.

For a primitive-only trace context (`MultiLevelAccelerationStructure<1>`, no `instancing` tag) the
instance term is absent on every target and no instance table is written; under `max_levels<N>` the
innermost instance id is used, as the trace dispatch already does.

## 3. Revised Contracts

### 3.1 Declarations

Was: PROPOSAL 2.2.1 declares `IShaderGroupSlot { static const int index; }` and an
`associatedtype Slot : IShaderGroupSlot` on `IHitGroup`, `IMissGroup`, and `ICallableGroup`; its
list interfaces are shown unsealed with elided bodies; `ITraceContext` carries `Payload`. The
draft module (`program-layout.slang`) additionally has `ShaderGroupSlot<N>`, the `HitGroupSlot` /
`MissSlot` / `CallableSlot` aliases, `[sealed]` list interfaces with a `static const int count`
requirement, the trace context as a generic argument of every list, and
`where expand each Group.Context.TraceContext == TraceContext` on every list. These details were
also recorded in the former implementation plan.

Now: the complete public surface. `contexts.slang`, `stage-contracts.slang` (each stage contract
exposes its context as an associated type), `program-schema.slang`, `trace.slang`, and
`descriptor.slang` change, as does the type of the `payload` property on `ClosestHitInput`,
`AnyHitInput`, and `MissInput` in `stage-inputs.slang`. The descriptor abstraction is unchanged,
but its source adopts the `Schema : ITraceProgramSchema` constraint. `RayTraversalDesc` is
unchanged.

```slang
namespace rt
{
    // ── contexts (one hierarchy; the payload moves from the trace context to the contexts) ──
    public interface ITraceContext
    {
        associatedtype AccelerationStructure : IAccelerationStructure;
        associatedtype Motion : IRayMotion;
    }

    // Common to every stage: the trace context and the per-record data the host writes.
    public interface IStageContext
    {
        associatedtype TraceContext : ITraceContext;
        associatedtype Record;
    }

    // Stages that carry a payload: hit and miss. A miss shader's context is any IPayloadContext, so
    // a hit context can serve a miss shader when the two share the record type.
    public interface IPayloadContext : IStageContext
    {
        associatedtype Payload;                         // the struct the stages read and write
    }

    public interface IHitContext : IPayloadContext
    {
        associatedtype Primitive : IIntersectionPrimitive;
    }

    public interface ICallableContext : IStageContext  // callables carry CallableData, not a payload
    {
        associatedtype CallableData;
    }

    // Stages that carry no data name an EMPTY payload type, a user struct with no data members
    // (payload model, Section 8). The module defines none: emptiness is a property of the type, not
    // a name, so a user's own empty struct cannot fall outside the rule. A trace of such entries
    // takes no payload argument (the second `trace` below), their stage inputs have no `payload`,
    // and no value of an empty payload type can be written.

    // ── stage contracts (the context becomes an associated type) ─────────────────────────
    // A stage names its own context. This lets the miss and callable sections list shaders
    // directly and lets IHitGroup require its three stages to agree on one context.
    public interface IClosestHitShader    { associatedtype Context : IHitContext;      void invoke(ClosestHitInput<Context> input); }
    public interface IAnyHitShader        { associatedtype Context : IHitContext;      void invoke(AnyHitInput<Context> input); }
    // Compiler-owned marker admitting IIntersectionShader or NoIntersection, as today. It is public
    // because the public `invoke` below names its Context (a public method may not reference a
    // member of an internal interface: E30604) and sealed so no user type conforms to it directly.
    [sealed] public interface IIntersectionStage { associatedtype Context : IHitContext; }
    public interface IIntersectionShader : IIntersectionStage
    {
        __constraint Context.Primitive : ICustomIntersectionPrimitive;
        void invoke(IntersectionInput<Context> input);
    }
    public interface IMissShader          { associatedtype Context : IPayloadContext;  void invoke(MissInput<Context> input); }
    public interface ICallableShader      { associatedtype Context : ICallableContext; void invoke(CallableInput<Context> input); }

    // Placeholders keep their generic argument and satisfy the associated type with it.
    public struct NoClosestHit<C : IHitContext>   : IClosestHitShader  { typealias Context = C; public void invoke(ClosestHitInput<C> input) {} }
    public struct NoAnyHit<C : IHitContext>       : IAnyHitShader      { typealias Context = C; public void invoke(AnyHitInput<C> input) {} }
    public struct NoIntersection<C : IHitContext> : IIntersectionStage { typealias Context = C; }

    // ── stage inputs (one type changes) ──────────────────────────────────────────────────
    // ClosestHitInput, AnyHitInput, MissInput:  property Context.Payload payload { ref; }
    //                                            (was Context.TraceContext.Payload)
    //                                            absent when Context.Payload is empty; use is an error (Section 9)
    // All other properties, including `record : Context.Record`, are unchanged.

    // ── the hit group (slot removed; the only group) ─────────────────────────────────────
    // A hit record binds up to three stages that share one context and one record, which is what
    // D3D12 and Vulkan call a hit group. A miss or callable record binds one shader, so those
    // sections list shaders and the draft's IMissGroup / ICallableGroup wrappers are gone. No entry
    // has a table position; how many records use it, and what each carries, is host data.
    public interface IHitGroup
    {
        associatedtype Context : IHitContext;
        associatedtype ClosestHit : IClosestHitShader;
        associatedtype AnyHit : IAnyHitShader;
        associatedtype Intersection : IIntersectionStage;
        __constraint ClosestHit.Context == Context;
        __constraint AnyHit.Context == Context;
        __constraint Intersection.Context == Context;
    }

    // ── section lists (trace-context argument and constraint removed) ────────────────────
    // Sealed: exactly the closed, open, and empty forms below exist. Membership of every entry in
    // the schema's trace context is a compiler check (Section 9), not a where clause, so the same
    // rule covers listed and linked entries.
    [sealed] public interface IHitGroupList {}
    [sealed] public interface IMissShaderList {}
    [sealed] public interface ICallableShaderList {}

    // Closed section. Function indices are the declaration ordinals, per payload for hit and miss,
    // program-wide for callables.
    public struct HitGroupList<each Group : IHitGroup>              : IHitGroupList        { public static const int count = countof(Group); }
    public struct MissShaderList<each Shader : IMissShader>         : IMissShaderList      { public static const int count = countof(Shader); }
    public struct CallableShaderList<each Shader : ICallableShader> : ICallableShaderList  { public static const int count = countof(Shader); }

    // Open section (Section 6): the listed entries plus every type in the linked program that
    // conforms to `Tag`, deduplicated by type. Listed entries take the lowest ordinals (within their
    // payload for hit and miss); linked entries follow in qualified-type-name order. `Tag` is checked by the compiler
    // to be an interface inheriting the section's entry interface (IHitGroup, IMissShader, or
    // ICallableShader).
    public struct OpenHitGroups<Tag, each Group : IHitGroup>              : IHitGroupList        { public static const int knownCount = countof(Group); }
    public struct OpenMissShaders<Tag, each Shader : IMissShader>         : IMissShaderList      { public static const int knownCount = countof(Shader); }
    public struct OpenCallableShaders<Tag, each Shader : ICallableShader> : ICallableShaderList  { public static const int knownCount = countof(Shader); }

    public struct NoHitGroups       : IHitGroupList       {}
    public struct NoMissShaders     : IMissShaderList     {}
    public struct NoCallableShaders : ICallableShaderList {}

    // ── the one schema contract (hit groups, miss shaders, callable shaders) ─────────────
    public interface ITraceProgramSchema
    {
        associatedtype TraceContext    : ITraceContext;
        associatedtype HitGroups       : IHitGroupList;
        associatedtype MissShaders     : IMissShaderList;
        associatedtype CallableShaders : ICallableShaderList;
    }

    // ── traversal description (unchanged from the draft) ─────────────────────────────────
    public struct RayTraversalDesc
    {
        public RayDesc ray;
        public float time;
        public RAY_FLAG rayFlags;
        public uint instanceMask;
        public uint sbtOffset;                          // RayContributionToHitGroupIndex
        public uint sbtStride;                          // MultiplierForGeometryContributionToHitGroupIndex
        public uint missIndex;                          // MissShaderIndex
    }

    // ── descriptor (unchanged at source level; Metal lowering in Section 4) ──────────────
    public struct TraceProgramDescriptor<Schema> where Schema : ITraceProgramSchema
    {
        internal ParameterBlock<TraceProgramDescriptorResources<Schema>> resources;   // erased on D3D12/Vulkan
    }

    // ── trace (generic on the payload; one constraint removed) ───────────────────────────
    public struct RayTracer<Schema> where Schema : ITraceProgramSchema {}

    public extension<Schema> RayTracer<Schema>
        where Schema : ITraceProgramSchema      /* plus the AccelerationStructure / Motion keying of today's eight extensions */
    {
        public void trace<Payload>(
            RayTraversalDesc desc,
            Schema.TraceContext.AccelerationStructure accelerationStructure,
            TraceProgramDescriptor<Schema> descriptor,
            inout Payload payload);                    // Payload inferred; compiler checks Payload ∈ payloads(Schema)

        public void trace(                             // the trace of entries whose payload type is empty; the
            RayTraversalDesc desc,                     // compiler checks that payloads(Schema) has one such type. How the
            Schema.TraceContext.AccelerationStructure accelerationStructure,   // native call's payload operand is
            TraceProgramDescriptor<Schema> descriptor);   // satisfied is internal to the lowering (payload model, Section 8).
    }

    // callShader keeps the draft's own extension, keyed by the schema alone (no traversal axes).
    public extension<Schema> RayTracer<Schema> where Schema : ITraceProgramSchema
    {
        public void callShader<CallableContext>(
            uint callableIndex,                        // a runtime callable RECORD index
            TraceProgramDescriptor<Schema> descriptor,
            inout CallableContext.CallableData data)
            where CallableContext : ICallableContext
            where CallableContext.TraceContext == Schema.TraceContext;   // unchanged from the draft:
                                                       // a method-level equality the front end enforces
    }
}
```

The context interfaces form one hierarchy. `IStageContext` holds what every stage has, the trace
context and the record type; `IPayloadContext` adds the payload for hit and miss stages;
`IHitContext` adds the primitive; `ICallableContext` adds the callable data instead of a payload.
Because a hit context is an `IPayloadContext`, a miss shader may name a hit context as its context
and share its record type, which removes the second context declaration in the common case; a
miss that needs its own record type declares an `IPayloadContext` of its own. The draft's
`IMissGroupContext` and `ICallableGroupContext` are renamed to `IPayloadContext` and
`ICallableContext`.

Only the hit section has groups. A hit record binds a closest-hit, an any-hit, and an intersection
stage that share one context; that bundle is `IHitGroup`, and its three equality constraints make
the shared context a checked fact rather than a convention. A miss or callable record binds one
shader, whose contract already names its context, so the draft's `IMissGroup` and `ICallableGroup`
are removed and the schema lists `MissShaders` and `CallableShaders` directly. The word *entry* in
this document means a hit group, a miss shader, or a callable shader: the unit one record binds to
and one function index names. Moving the stage context from a generic parameter to an associated
type is what makes the shader lists possible: a variadic list can require `each Shader :
IMissShader` and read `Shader.Context.Payload`, but it cannot name a per-element generic argument.
The same move costs one `typealias Context = ...;` line per stage struct in place of the generic
argument on its conformance.

```text
                       draft                                         revised
miss shader        struct SkyMiss : rt::IMissShader<Ctx> { .. }      struct SkyMiss : rt::IMissShader { typealias Context = Ctx; .. }
miss entry         struct SkyMissGroup : rt::IMissGroup              (none; the shader is the entry)
                   { typealias Context = Ctx; typealias Miss = SkyMiss; }
schema section     MissGroups = rt::MissGroupList<SkyMissGroup, ..>  MissShaders = rt::MissShaderList<SkyMiss, ..>
```

The revised intrinsic table (replaces the one in 2.2.1):

| Intrinsic | Describes |
| --- | --- |
| `ITraceContext` | Topology and motion of one trace program |
| `IStageContext`, `IPayloadContext`, `IHitContext`, `ICallableContext` | A stage's signature: trace context and record type for every stage; plus the payload for hit and miss stages; plus the primitive for hit stages; plus the callable data for callables |
| `IClosestHitShader`, `IAnyHitShader`, `IIntersectionShader`, `IMissShader`, `ICallableShader` | One stage on the context it names |
| `IHitGroup` | The only group: three hit stages on one shared context and record |
| `HitGroupList`, `MissShaderList`, `CallableShaderList` | The closed set of entries in a section |
| `OpenHitGroups`, `OpenMissShaders`, `OpenCallableShaders` | A section whose set is completed at link time from a tag interface |
| `NoHitGroups`, `NoMissShaders`, `NoCallableShaders` | An empty section |
| `ITraceProgramSchema` | The entry sets of one trace program, for one trace context |
| `payloads(Schema)` (derived, not declared) | The payload types the schema's hit groups and miss shaders serve |
| function index (reflected, not declared) | The ordinal of an entry within its section (per payload for hit and miss, program-wide for callables); the Metal table index and record identity |

### 3.2 Replaced Paragraphs

Was (2.2.1): "The group list and the group slot have complementary roles: the list declares that a
group belongs to the layout, while the group's `IShaderGroupSlot` type declares where its record
resides. ... Slots are zero-based and must be unique within their SBT section. The explicit slot,
rather than list position, is authoritative, so reordering declarations does not renumber SBT
records."

Now: the list declares the set of entries (hit groups, miss shaders, callable shaders) a trace can
reach. Where an entry's records reside, and how many there are, is host data. The compiler assigns
each entry a function index within its payload's section, which for a closed list is its
declaration ordinal, so reordering a closed list renumbers; hosts key by entry name and re-query
after each compile, as they re-query DXR shader identifiers after each state-object build.

Also replaced in 2.2.1, each by removing the slot from the sentence: "which groups belong to each
SBT section, and the slots those groups occupy in their sections" (intro); "Each group interface
describes one logical SBT record and names its slot in the corresponding SBT section ... The
group-list types declare which records belong to the three SBT sections" (now: `IHitGroup`
describes three stages on one context and each stage interface one stage; the list types declare
which entries belong to the three sections); "host code reflects the declared groups and constructs
each native hit, _Miss_, and _Callable_ record at its declared slot" (now: writes as many records as
the scene needs, each naming a reflected entry); "Slang uses the same declared group membership and
slots to synthesize the _Miss_ and _ClosestHit_ dispatch" (now: the same declared membership).

Was (2.2.1): "D3D and Vulkan map that property to native shader-record data; Metal loads it from
the descriptor's generated data buffer before dispatching the logical stage."

Now: D3D and Vulkan map `input.record` to native shader-record data. On Metal the generated trace
dispatch resolves the record from the SBT buffer and passes a pointer to the generated stage
function, which reads `Context.Record` through it. The property therefore means "the record that
selected this stage" on every target.

Was (2.2.1, _Callable_ paragraph): "Metal indexes the descriptor's _Callable_ visible-function
table. ... Metal passes the descriptor resources and record buffer through generated visible
functions so nested calls and `input.record` use the same program descriptor. ... Slang diagnoses an
incompatible group during specialization."

Now: the operation and its availability are unchanged. The index passed to `callShader` is a
runtime callable **record** index; on Metal the dispatch resolves the callable record and indexes
the callable visible-function table by that record's function index. The descriptor resources and
record buffer are still threaded through generated visible functions for nested `callShader`, but
`input.record` in a callable arrives as the record pointer resolved by the dispatch. All callable
shaders in one program still share one `CallableData` type; the mismatch is diagnosed at structural
synthesis over the linked callable set and names the offending shader (Section 9). Callables have
no payload: the callable section is one table shared by every payload.

Was (2.2.3 and the retired RAY_PAYLOAD_MODEL.md Section 9):
`ITraceContext { associatedtype Payload; ... }` and "Using one associated type guarantees that
every structurally reachable shader group agrees on the payload type."

Now: the payload is an associated type of the hit and miss **contexts**, and the trace call is
generic on the payload it passes. One schema carries as many payload types as its hit groups and
miss shaders use. The
old sentence was true because a schema could hold only one payload, which is also why a second
payload needed a second schema; the payload model's Section 5 states what is checked in its place
and what remains a host rule. The 2.2.3 example's `PrimaryTraceContext` loses `Payload`, its hit and
miss contexts gain `Payload`, and its two `typealias Slot` lines are deleted; the inference
relationship in 2.2.3 is otherwise unchanged.

### 3.3 Example: One Schema, Two Payloads, Host-Written Table

The migration example from PROPOSAL 3.1, extended with the shadow ray that a real renderer has.
Before this revision the shadow ray would have needed a second schema because its payload differs.

```slang
import slang.raytracing;

struct RadiancePayload  { float3 radiance; float3 throughput; }
struct OcclusionPayload { uint occluded; }

struct SceneTraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

// Per-record data, one per (instance, geometry, rayType). The host writes it.
struct MaterialRecord { uint materialIndex; float alphaThreshold; }

struct PrimaryMeshContext : rt::IHitContext
{
    typealias TraceContext = SceneTraceContext;
    typealias Payload = RadiancePayload;
    typealias Primitive = rt::TrianglePrimitive;
    typealias Record = MaterialRecord;
}
struct ShadowMeshContext : rt::IHitContext
{
    typealias TraceContext = SceneTraceContext;
    typealias Payload = OcclusionPayload;
    typealias Primitive = rt::TrianglePrimitive;
    typealias Record = MaterialRecord;
}
// Miss contexts with their own (empty) record. A miss shader whose record matched the hit record
// could name the hit context directly instead, since IHitContext : IPayloadContext.
struct PrimaryMissContext : rt::IPayloadContext { typealias TraceContext = SceneTraceContext; typealias Payload = RadiancePayload;  typealias Record = void; }
struct ShadowMissContext  : rt::IPayloadContext { typealias TraceContext = SceneTraceContext; typealias Payload = OcclusionPayload; typealias Record = void; }

// Stages. Each names its context. A stage whose body is the same for several payloads is generic
// over the context (AlphaTest).
struct ShadePbr : rt::IClosestHitShader
{
    typealias Context = PrimaryMeshContext;
    void invoke(rt::ClosestHitInput<PrimaryMeshContext> input)
    {
        MaterialRecord rec = input.record;                     // per record on every target
        input.payload.radiance += shade(rec.materialIndex, input.triangle.barycentricCoord) * input.payload.throughput;
    }
}
struct ShadowHit : rt::IClosestHitShader
{
    typealias Context = ShadowMeshContext;
    void invoke(rt::ClosestHitInput<ShadowMeshContext> input) { input.payload.occluded = 1; }
}
struct AlphaTest<C> : rt::IAnyHitShader                          // shared by both payloads (front-end limit today: Section 9)
    where C : rt::IHitContext
    where C.Primitive == rt::TrianglePrimitive
    where C.Record == MaterialRecord
{
    typealias Context = C;
    void invoke(rt::AnyHitInput<C> input)
    {
        if (coverage(input.primitiveIndex, input.triangle.barycentricCoord) < input.record.alphaThreshold)
            input.ignoreHit();
    }
}
struct SkyMiss    : rt::IMissShader { typealias Context = PrimaryMissContext; void invoke(rt::MissInput<PrimaryMissContext> i) { i.payload.radiance += sky(i.worldSpaceDirection) * i.payload.throughput; } }
struct ShadowMiss : rt::IMissShader { typealias Context = ShadowMissContext;  void invoke(rt::MissInput<ShadowMissContext> i)  { i.payload.occluded = 0; } }

// Hit groups carry no table position. Their payload comes from their context, which every stage
// must share. Miss shaders are listed as they are; there is no wrapper.
struct OpaqueGroup       : rt::IHitGroup { typealias Context = PrimaryMeshContext; typealias ClosestHit = ShadePbr;  typealias AnyHit = rt::NoAnyHit<PrimaryMeshContext>; typealias Intersection = rt::NoIntersection<PrimaryMeshContext>; }
struct AlphaTestedGroup  : rt::IHitGroup { typealias Context = PrimaryMeshContext; typealias ClosestHit = ShadePbr;  typealias AnyHit = AlphaTest<PrimaryMeshContext>;   typealias Intersection = rt::NoIntersection<PrimaryMeshContext>; }
struct ShadowOpaqueGroup : rt::IHitGroup { typealias Context = ShadowMeshContext;  typealias ClosestHit = ShadowHit; typealias AnyHit = rt::NoAnyHit<ShadowMeshContext>;  typealias Intersection = rt::NoIntersection<ShadowMeshContext>; }
struct ShadowAlphaGroup  : rt::IHitGroup { typealias Context = ShadowMeshContext;  typealias ClosestHit = ShadowHit; typealias AnyHit = AlphaTest<ShadowMeshContext>;    typealias Intersection = rt::NoIntersection<ShadowMeshContext>; }

// Reflected function indices, per payload and section, in declaration order:
//   RadiancePayload:  hit 0 = OpaqueGroup, 1 = AlphaTestedGroup;        miss 0 = SkyMiss
//   OcclusionPayload: hit 0 = ShadowOpaqueGroup, 1 = ShadowAlphaGroup;  miss 0 = ShadowMiss
struct SceneSchema : rt::ITraceProgramSchema
{
    typealias TraceContext    = SceneTraceContext;
    typealias HitGroups       = rt::HitGroupList<OpaqueGroup, AlphaTestedGroup, ShadowOpaqueGroup, ShadowAlphaGroup>;
    typealias MissShaders     = rt::MissShaderList<SkyMiss, ShadowMiss>;
    typealias CallableShaders = rt::NoCallableShaders;
}

// The table convention is the engine's, shared with the host as ordinary constants.
static const uint kRayTypeCount = 2, kPrimaryRayType = 0, kShadowRayType = 1;

rt::AccelerationStructure gScene;
rt::TraceProgramDescriptor<SceneSchema> gProgram;      // one descriptor for both payloads

[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTracer<SceneSchema> tracer;
    rt::RayTraversalDesc d = {};
    d.ray = cameraRay(DispatchRaysIndex().xy);
    d.instanceMask = 0xff;
    d.sbtStride = kRayTypeCount;
    d.sbtOffset = kPrimaryRayType;  d.missIndex = kPrimaryRayType;
    RadiancePayload p = { 0, 1 };
    tracer.trace(d, gScene, gProgram, p);               // Payload = RadiancePayload, inferred

    d.ray = shadowRay(...);
    d.rayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    d.sbtOffset = kShadowRayType;   d.missIndex = kShadowRayType;
    OcclusionPayload o = { 1 };
    tracer.trace(d, gScene, gProgram, o);               // Payload = OcclusionPayload, inferred
    ...
}
```

Nothing in this source says how many records exist. For every `(instance, geometry)` the host
writes two hit records at `instanceBase + geometryIndex * 2 + rayType`: for the primary ray a
record naming `OpaqueGroup` or `AlphaTestedGroup` with the surface's `MaterialRecord`, for the
shadow ray a record naming `ShadowOpaqueGroup` or `ShadowAlphaGroup` with the same data. The miss
section has two records. Adding an instance appends records (case F). A trace passing a payload no
entry serves is a compile error (Section 9); a trace passing `kShadowRayType` with the radiance
payload compiles, because where the host put the occlusion records is not in the program (Section
2.3).

## 4. Revisions To PROPOSAL Section 2.2.2: Metal Lowering Of The Descriptor

### 4.1 Native Layout (2.2.2.1)

Was: "`visible_function_table_1` is the _ClosestHit_ table ... entry `logicalHitSlot` -> generated
_ClosestHit_ function", that is, one table entry per record; and "The first Metal layout uses four
header words in `descriptorData`. They contain the word offsets of the instance hit-group-offset
table and the hit, _Miss_, and _Callable_ record tables. Each record table entry is a byte offset
... to that record's data." Records carried no identity of their own because the table index *was*
the slot; the descriptor had exactly five fields.

Now: tables are per payload and indexed by function index; the record buffer is a shader binding
table in the D3D12 sense. Because each payload has its own `ray_data` type, it has its own
visible-function-table element types and its own intersection function table; the callable table
and the record buffer are shared.

```text
TraceProgramDescriptor<Schema> on Metal, one table set per payload P0 .. Pn-1
    (the compiler orders the sets by first appearance in HitGroups, then MissShaders, linked entries
     after listed ones by qualified type name; reflection names each field's payload, Section 8.3)
    for each payload Pi:
        intersection_function_table<payloadTags_i>        intersectionFunctions_i    // per-kind dispatchers at 0 / 1 / 2 (Section 5)
        visible_function_table<MissSig_i>                 missFunctions_i            // one entry per miss shader of Pi
        visible_function_table<ClosestHitSig_i>           closestHitFunctions_i      // one entry per hit group of Pi
    visible_function_table<CallableSig>                   callableFunctions          // one entry per callable shader
    uint32_t device const*                                records                    // the SBT (below)
```

The resource shape of the `TraceProgramDescriptorResources<Schema>` placeholder in
`descriptor.slang` is unchanged; on Metal the lowering already rewrites it, and now synthesizes the
struct above for the schema's payload count instead of rewriting five fixed field types. Reflection
reports the field order (Section 8.3). A payload partition with no candidate logic keeps its
intersection-table field (the field count is always `3 * payloadCount + 2`, Section 11) but reflects
no dispatchers, and a trace of that payload passes no table to `intersect`, exactly as the draft does
today with its single unused `intersectionFunctions` field; the host binds an empty table (Section
8.1, step 4).

Record buffer:

```text
Header (16 bytes, four u32 byte offsets from the buffer base)
   0  instanceTableOffset    u32 per TLAS instance: hit-record base for that instance;
                             absent for a primitive-only trace context
   4  hitOffset
   8  missOffset
  12  callableOffset

Hit record  i  at hitOffset      + i * HIT_STRIDE        i = instanceContribution + geometryIndex * desc.sbtStride + desc.sbtOffset
Miss record m  at missOffset     + m * MISS_STRIDE       m = desc.missIndex
Callable rec k at callableOffset + k * CALLABLE_STRIDE   k = the callShader argument

Record (same shape in every section)
  +0  u32 functionIndex      index into the payload's (or the callable) visible-function table;
                             for hit records also the case selector of that payload's dispatcher;
                             0xFFFFFFFF = empty record, no stage runs (like a null DXR identifier)
  +4  padding to 16 bytes    keeps the data 16-byte aligned for Metal's natural layout
  +16 Context.Record         in the reflected record layout, padded to the section stride
```

The strides are compile-time constants: `16 + size of the largest Record` in that section across all
payloads, rounded up to 16, reflected as `getHitRecordStride()` and friends and baked into the
generated dispatch; the host never writes them. The four offsets are host data because they are
running sums of `count * stride` and the counts are the scene's. A host helper computes them from
reflection:

```text
instanceTableOffset = 16
hitOffset           = align16(instanceTableOffset + instanceCount * 4)
missOffset          = align16(hitOffset + hitCount * hitStride)
callableOffset      = align16(missOffset + missCount * missStride)
```

**Relationship to PROPOSAL.md.** PROPOSAL 2.2.2.1's four header words were word offsets of
per-record *offset tables*; the version-1 record had no identity because the table index was the
slot. This layout keeps four offset words but makes records fixed-pitch with the function index in
the record, so the tables can be per entry. The former implementation plan described the version-1
buffer; the current [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) incorporates the layout above.

### 4.2 Lowering Strategy (2.2.2.1)

Was:

```slang
if (result.isNone)
{
    missFns[desc.missIndex](payload, descriptorData, desc.missIndex);
}
else
{
    uint logicalHitSlot = instanceOffset + geometryId * desc.sbtStride + desc.sbtOffset;
    closestHitFns[logicalHitSlot](payload, descriptorData, logicalHitSlot, result);
}
```

Now, for a trace whose payload is `P1` (the payload is a compile-time type at the call site, so the
table set is selected statically; the record index uses the runtime `desc` fields as today):

```metal
// Internal Metal-shaped pseudocode for a trace of payload P1 in a two-payload program.
thread Schema_P1_rayData rayData;                 // holds OcclusionPayload
rayData.payload = payload;
rayData.descriptorData = res->records;            // present when a reachable stage dispatches a callable
                                                  //   or this payload partition has candidate logic (Section 5)
rayData.sbtOffset = desc.sbtOffset;               // both present only when the program has candidate
rayData.sbtStride = desc.sbtStride;               //   logic: the dispatchers recompute the hit record

auto R = intersector.intersect(ray, scene, desc.instanceMask, res->intersectionFunctions_1, rayData);

uint32_t device const* h = res->records;
if (R.type == intersection_type::none)
{
    uchar device const* rec = (uchar device const*)h + h[2] + desc.missIndex * MISS_STRIDE;
    uint fn = *(uint device const*)rec;
    if (fn != 0xFFFFFFFFu)
        res->missFunctions_1[fn](&rayData, ..., rec + 16);
}
else if ((desc.rayFlags & RAY_FLAG_SKIP_CLOSEST_HIT_SHADER) == 0)
{
    uint recordIndex = h[(h[0] >> 2) + R.instance_id] + R.geometry_id * desc.sbtStride + desc.sbtOffset;
    uchar device const* rec = (uchar device const*)h + h[1] + recordIndex * HIT_STRIDE;
    uint fn = *(uint device const*)rec;
    if (fn != 0xFFFFFFFFu)
        res->closestHitFunctions_1[fn](&rayData, R.distance, ..., rec + 16, kernelContext);
}
payload = rayData.payload;
```

Two dependent loads on the hit path (instance contribution, function index), the same count as the
draft implementation today; one on the miss path. A generated stage function reads its record from
the pointer:

```metal
[[visible]] void ShadowHit(Schema_P1_rayData thread* rayData, float distance, float2 barycentrics,
                           uint primitiveIndex, uchar device const* record, uchar thread* kernelContext)
{
    MaterialRecord rec = *((MaterialRecord device const*)record);
    rayData->payload.occluded = 1;
}
```

Callable dispatch changes the same way: `rec = h[3] + callableIndex * CALLABLE_STRIDE`, then
`callableFunctions[*(uint*)rec](...)`.

### 4.3 Gaps, Fixes, And Constraints (2.2.2.1)

Gap 1 (tables do not dispatch every stage) is unchanged; its constraint now reads: for each payload
the host creates a miss table with one entry per reflected miss shader and a closest-hit table with
one entry per reflected hit group of that payload, installing each function at its function index
and the reflected no-op function for `NoClosestHit` groups (Section 5); one callable table for the
program. Tables are
built once per pipeline and never change when records are added, removed, or repointed.

Gap 2 ("Native function-table indexing is not the portable hit-slot formula") remains a fact about
Metal: the IFT entry is selected from acceleration-structure offsets with no ray-type term. Its
constraint is removed. Was: "the host must construct acceleration-structure function-table offsets
and function table contents so every `metalIFTIndex` selected by traversal maps to exactly one
`logicalHitSlot` ... Separate `TraceProgramDescriptor` values per ray type are also a natural way to
keep this mapping simple." Now: candidate functions are selected by the record through the per-kind
dispatcher of the tracing payload (Section 5). The host sets every geometry's
`intersectionFunctionTableOffset` to its primitive-kind constant (triangle 0, bounding box 1, curve
2) and every instance's to 0. Nothing in the acceleration structure depends on materials, ray types,
or records, and one descriptor serves every payload.

### 4.4 Concrete Example (2.2.2.1)

Was: the `sbtStride = 2, sbtOffset = 1` example mapping `metalIFTIndex` 8 and 12 to logical slots
1 and 3.

Now: the Section 3.3 program with one instance holding two geometries, `MaterialRecord` of 8 bytes,
so `HIT_STRIDE = 32`, `Record = void` for miss so `MISS_STRIDE = 16`; the engine's convention is
stride 2:

```text
bytes 0..15    header: 16, 32, 160, 192          (instance table, hit, miss, callable offsets)
byte  16       instance table: 0                  (padded to 32)
byte  32       hit record 0: geometry 0, primary  fn=1 (AlphaTestedGroup, RadiancePayload table)   data {materialIndex 7, alpha 0.5}
byte  64       hit record 1: geometry 0, shadow   fn=1 (ShadowAlphaGroup, OcclusionPayload table)  same data
byte  96       hit record 2: geometry 1, primary  fn=0 (OpaqueGroup, RadiancePayload table)        data {materialIndex 3, alpha 0}
byte  128      hit record 3: geometry 1, shadow   fn=0 (ShadowOpaqueGroup, OcclusionPayload table) same data
byte  160      miss record 0: primary             fn=0 (SkyMiss, RadiancePayload table)
byte  176      miss record 1: shadow              fn=0 (ShadowMiss, OcclusionPayload table)
byte  192      callable section: empty
```

replacing the version-1 word array the draft Metal test host writes today
(`{4, 5, 7, 9, 0, 36, 40, 44, 48, 100, 200, 300, 400}`, `metal-test-host.mm:409-423`). At engine
scale, 20,000 hit records at 32 bytes is 640 KB and each closest-hit table still has one entry per
group of its payload.

### 4.5 Future IFB Lowering (2.2.2.2)

Remains a future item, but its slot-indexed text is superseded in the same direction as 2.2.2.1:
`closestHitFns[logicalHitSlot]` becomes `closestHitFns_i[record.functionIndex]`, the function buffer
holds one candidate handle per **record** (the hit section of this buffer) rather than per logical
slot, "slot maps" is deleted from the descriptor-side data, and Pattern B in 3.3 installs by
function index. Note added: the hit section already has the properties an
`intersection_function_buffer` needs (per-record identity at a fixed stride, aligned data), and
record data would surface as `[[user_data]]`. The IFB entry encoding has not been verified against
the MSL specification, so this is an intent, not a commitment.

## 5. Revisions To PROPOSAL Section 2.3.1: Metal Candidate Functions

Was: "The Metal binding unit is the **generated candidate function for the whole hit group**, not
an individual source _AnyHit_ or _Intersection_ stage. Each hit group therefore reflects at most one
Metal candidate function ... If it is non-null, the host installs it as the hit group's IFT entry."

Why this changes: on DXR the record names all three shaders of a hit group, so the _AnyHit_ that
runs for a candidate depends on the ray type and the instance contribution. Metal selects the
intersection function from `geometry.intersectionFunctionTableOffset +
instance.intersectionFunctionTableOffset`, with no ray-type term and no record. A per-group IFT
entry therefore cannot express alpha-tested geometry that runs `AlphaTest` for both rays but a
different closest hit for each, or one geometry instanced with two materials.

Now: for each payload partition that contains candidate logic, the compiler generates at most one
`[[intersection(...)]]` function per primitive kind, at fixed IFT indices (triangle 0, bounding box
1, curve 2) in that payload's intersection function table. Each dispatcher recomputes the hit
record from the runtime operands the trace stored in `ray_data` and switches on the record's
function index over the hit groups of that payload only. A payload partition with no candidate
logic reflects no dispatchers, and a trace of that payload passes no IFT. The per-group bodies that
today's candidate adapters produce become ordinary functions called from the arms. For
`OcclusionPayload` in the Section 3.3 program:

```metal
// Internal Metal-shaped pseudocode.
[[intersection(triangle, instancing, triangle_data)]]
bool SceneSchema_P1_triangleCandidate(
    uint primitiveIndex [[primitive_id]], uint geometryIndex [[geometry_id]], uint instanceIndex [[instance_id]],
    float2 barycentrics [[barycentric_coord]], float distance [[distance]], bool opaque [[opaque]],
    ray_data SceneSchema_P1_rayData& rayData [[payload]])
{
    uint32_t device const* h = rayData.descriptorData;
    uint recordIndex = h[(h[0] >> 2) + instanceIndex] + geometryIndex * rayData.sbtStride + rayData.sbtOffset;
    uchar device const* record = (uchar device const*)h + h[1] + recordIndex * HIT_STRIDE;
    uint fn = *(uint device const*)record;
    switch (fn)
    {
    case 1:  return ShadowAlphaGroup_anyHit(record + 16, primitiveIndex, barycentrics, distance, rayData);   // AlphaTest<ShadowMeshContext>
    case 0:  return true;                   // ShadowOpaqueGroup has no AnyHit: accept, as DXR does
    case 0xFFFFFFFFu: return true;          // empty record: no shader runs
    default: return false;                  // out-of-contract function index: drop the candidate
    }
}
```

A bounding-box dispatcher composes each group's _Intersection_ and _AnyHit_ inside its arm with the
existing `reportHit` accumulator and returns a rejection for `default` and empty records.

What changes and what does not inside 2.3.1:

- The `reportHit` accumulator semantics ("Lowering `reportHit` on Metal", steps 1 to 5, the Boolean
  control-flow contract, committed attributes) are unchanged.
- The generated per-group function is no longer itself an `[[intersection(bounding_box)]]` function
  installed in the IFT. It becomes an ordinary Metal function called from the dispatcher arm for
  that group's function index. Accordingly, in the composition table every Metal-lowering cell that
  reads "... and put that function in the IFT" becomes "... as an arm of the per-kind dispatcher";
  the sphere example's `generatedSphereHitGroupCandidate` loses its `[[intersection(bounding_box)]]`
  attribute and gains explicit parameters; and the binding lines
  `IFT[metalIFTIndex] = generatedSphereHitGroupCandidate` /
  `ClosestHitVFT[logicalHitSlot] = generatedSphereClosestHit` become
  `IFT_i[1] = <bounding-box dispatcher of payload i>` /
  `ClosestHitVFT_i[functionIndex] = generatedSphereClosestHit`.
- "The generated candidate function is specialized per concrete hit group. If two groups reuse the
  same source _Intersection_ type with different _AnyHit_ types, Slang generates two candidate
  functions" becomes: two arms of the one dispatcher, not two IFT entries.
- "`NoAnyHit`, `NoClosestHit`, and `NoIntersection` ... do not consume native shader entries" and
  "when [_ClosestHit_] is absent, Slang emits no such dispatch" are revised for `NoClosestHit`: such
  a group still has a function index, and the compiler emits one reflected no-op visible function
  for every payload whose hit section contains at least one placeholder closest hit, including an
  all-placeholder section. The host installs that shared no-op at every placeholder ordinal, so no
  table entry the dispatch can reach is null. `NoAnyHit` and `NoIntersection` still consume
  nothing.

The reflected candidate table in 2.3.1 becomes:

| Payload-partition contents | Reflected Metal candidate dispatchers |
| --- | --- |
| The payload has candidate logic (some group has _AnyHit_ or _Intersection_ logic), so its trace passes an IFT | Triangle and bounding-box dispatchers, always, so every geometry kind this payload can trace has an entry; a kind with no groups gets a reject-all stub |
| The payload has a curve group and candidate logic | Curve dispatcher in addition |
| The payload has no candidate logic | None; a trace of this payload passes no IFT |

Consequences: ray-type- and instance-specific _AnyHit_ behavior works as on DXR because the record
is the single source of truth for candidate and closest-hit selection; the host never chooses IFT
indices; candidates pay the record lookup and a switch on a value that is uniform per geometry, over
the groups of one payload only. The existing restriction that candidate logic cannot reference
global shader parameters on Metal is carried over; records are the way to feed candidates data.

One further Metal change follows from records being runtime data: the committed custom-attribute
field in `ray_data` is keyed by the **attributes type** (`Context.Primitive.Attributes`) rather than
by group type, so one generated _ClosestHit_ function can serve every group that uses the same
stage, matching the D3D/Vulkan adapter identity `(stage kind, invoke)`.

## 6. New: Open Sections

This section has no counterpart in PROPOSAL.md. It covers case C: material or geometry modules
compiled separately from the ray-generation module and linked later.

The mechanism is Slang linking, not a declared target ABI. A shared context module declares the
payloads, contexts, and a tag interface:

```slang
module scene_contexts;
import slang.raytracing;

public struct RadiancePayload  { public float3 radiance; public float3 throughput; }
public struct OcclusionPayload { public uint occluded; }
public struct SceneTraceContext : rt::ITraceContext
{
    public typealias AccelerationStructure = rt::AccelerationStructure;
    public typealias Motion = rt::NoMotion;
}
public struct PrimaryMeshContext : rt::IHitContext { public typealias TraceContext = SceneTraceContext; public typealias Payload = RadiancePayload;  ... }
public struct ShadowMeshContext  : rt::IHitContext { public typealias TraceContext = SceneTraceContext; public typealias Payload = OcclusionPayload; ... }

// Any group conforming to this tag joins the hit section of every schema that opens on it.
public interface IPbrHitGroup : rt::IHitGroup {}
```

The ray-generation module opens its hit section on the tag and may list groups it knows:

```slang
struct SceneSchema : rt::ITraceProgramSchema
{
    typealias TraceContext   = SceneTraceContext;
    typealias HitGroups       = rt::OpenHitGroups<IPbrHitGroup, OpaqueGroup, ShadowOpaqueGroup>;   // open
    typealias MissShaders     = rt::MissShaderList<SkyMiss, ShadowMiss>;                            // closed
    typealias CallableShaders = rt::NoCallableShaders;
}
```

A plugin module, compiled without seeing the ray-generation module, contributes groups by
conforming to the tag, one per payload it supports:

```slang
import slang.raytracing;
import scene_contexts;

struct GlassGroup       : IPbrHitGroup { typealias Context = PrimaryMeshContext; typealias ClosestHit = GlassClosestHit; typealias AnyHit = GlassAlphaTest<PrimaryMeshContext>; typealias Intersection = rt::NoIntersection<PrimaryMeshContext>; }
struct GlassShadowGroup : IPbrHitGroup { typealias Context = ShadowMeshContext;  typealias ClosestHit = ShadowHit;       typealias AnyHit = GlassAlphaTest<ShadowMeshContext>;  typealias Intersection = rt::NoIntersection<ShadowMeshContext>; }
```

The host composes the ray-generation module and every plugin module into one program and links
it, as for any Slang program. Structural synthesis, the compiler phase that runs on the linked IR
before dead-code elimination and turns each trace site's schema into generated stage functions and
dispatch (IMPLEMENTATION_PLAN.md Section 5), then enumerates every witness table in the linked IR
whose conformance type is the tag, forms the set union with the listed entries (deduplicated by
type, so a listed entry that also conforms appears once), derives `payloads(Schema)` and partitions
the result by payload, and continues exactly as for a closed list: tag inference, `ray_data` sizing,
table signatures, and the candidate dispatchers all see the complete set. Nothing about the target
ABI is declared in source, so METAL_TAG_LIST_ANALYSIS.md's inference decision stands for open sets.
A linked entry may introduce a payload no listed entry uses; its tables are generated and reflected,
and a trace in the ray-generation module can use it only by importing the payload type.

Function indices in the open part: within each payload, listed entries take the lowest ordinals in
declaration order and linked entries follow in qualified-type-name order. Tables are always dense.
Adding or removing a plugin can renumber the linked entries, exactly as rebuilding a DXR state
object changes its shader identifiers, so hosts key records by entry name and resolve name to index
after each compile. Nothing in shader source names a table position, for closed and open sections
alike; an earlier draft had a per-entry pin attribute for this and it was dropped as a second way
to declare a slot.

Two requirements and one limit:

- A tagged conformance must survive IR linking, which only retains entry-point-reachable and
  exported instructions. The front end marks witness tables whose target is a section tag with the
  same keep-alive and export decorations the type-conformance component uses, and synthesis strips
  them from tables it did not claim. Until that lands, `export struct GlassGroup : IPbrHitGroup` or
  a type-conformance component provides the retention.
- Adding a plugin means linking again and producing a new target binary. On Metal that is a new
  compute pipeline with a larger linked-function set, which the platform requires in any case.
- Linking independently compiled Metal binaries (a plugin metallib into a foreign kernel) is not
  supported: generated visible functions bind the ray-generation kernel's global-parameter block by
  layout, and Metal does not validate table signatures when a function is installed. Slang-level
  linking is the supported open-set path. On D3D12 and Vulkan, native pipeline libraries with
  per-stage entry points continue to work, and reflection supplies the payload and attribute sizes
  their pipeline interface needs.

## 7. Revisions To PROPOSAL Sections 1.3 And 2.1

Was (2.1, step 2): "Declare the SBT layout. `ITraceProgramLayout` maps hit, _Miss_, and _Callable_
shader groups to logical SBT slots."

Now: "Declare the trace program's entry sets. `ITraceProgramSchema` lists the hit groups, _Miss_
shaders, and _Callable_ shaders a trace can reach, for one trace context; each hit group's and miss
shader's context names the payload its stages read. It assigns no table positions; the host builds
the table from the reflected sets."

Was (2.1, step 3): "The trace context carries the shared payload and traversal shape".

Now: "The trace context carries the traversal shape; each hit and miss context carries its
payload."

Was (1.3 and 2.2.1): 1.3 defines reachability as the SBT entries one trace call can select and
observes that the existing model cannot know it from source; 2.2.1 answers that "explicit layout
intrinsics keep the SBT schema finite, reviewable, and directly reflectable".

Now: that property holds for closed sections, and it is sharper than before: the entries a trace of
payload `P` reaches, when the host follows the rule of the payload model's Section 6, are the hit
groups and miss shaders of `P`, not the whole schema. For a section opened
on a tag, the reachable set is finite and reflectable once the program is linked, not from the
ray-generation source alone. The schema author opts into that per section. This is a deliberate
revision of the 1.3 argument, made because the alternative, a declared target ABI in source, cannot
fix the Metal table signatures without also declaring callable presence, global-parameter use, and
candidate-driven `ray_data` contents. [DYNAMIC_SBT_DESIGN.md](DYNAMIC_SBT_DESIGN.md) explains why
the linked schema and runtime records are separate sources of information.

PROPOSAL 2.5's inference sources remain valid, but aggregation is payload-aware:

```text
PayloadMetalTags<Schema, P> = normalize(
    Schema.TraceContext.AccelerationStructure.sharedRequirements,
    Schema.TraceContext.Motion.requirements,
    union(ReachableStage<Schema, P>.requirements),
    SelectedCapabilities.requirements,
    Lowering.requirements)
```

Topology and motion are fixed by the schema's trace context. Optional stage-data requirements are
collected only from the hit groups and miss shaders of `P`, including linked entries. The primitive
selector remains local to each generated candidate dispatcher.

## 8. Revisions To PROPOSAL Section 3: Migration And Reflection

### 8.1 Metal Host Migration (3.1)

Was, steps 2 to 4: populate the _Miss_ and _ClosestHit_ visible-function tables from the reflected
slots; put each hit group's reflected candidate function in the IFT; choose native IFT indices per
logical hit slot and build acceleration-structure offsets so traversal selects them.

Now:

1. Query the schema through reflection.
2. Link every reflected stage function, each payload's reflected no-op closest-hit function when
   present, and every reflected candidate dispatcher into the pipeline, by their exact reflected
   names.
3. For each payload, create a miss table with one entry per reflected miss shader and a closest-hit
   table with one entry per reflected hit group of that payload, and install each function at its
   function index (the no-op function for `NoClosestHit` groups). Create one callable table for the
   program.
4. For every payload partition that reflects candidate dispatchers, create an IFT with three
   entries and install that payload's dispatchers at their table indices. For a payload partition
   without candidate logic, bind an empty table in its intersection-table field; a trace of that
   payload passes no IFT, as with today's unused IFT.
5. Set every geometry's `intersectionFunctionTableOffset` to its primitive-kind constant and every
   instance's to 0.
6. Write the SBT buffer (Section 4.1): the four offsets; the instance table (omitted for a
   primitive-only trace context); one hit record per `(instance, geometry, rayType)` at the position
   the engine's convention gives it, naming a group by function index and carrying its record bytes
   packed from the reflected record layout; one miss record per
   `missIndex` the shader uses; one callable record per callable. Per-frame changes rewrite records
   only.
7. Write the descriptor: the per-payload tables in reflected field order, the callable table, and
   the records buffer.

### 8.2 D3D/Vulkan Host Migration (3.2)

Was, steps 2 to 3: "For each reflected _Miss_ group, add a _Miss_ record at its declared slot. For
each reflected hit group, add a hit group record at its declared slot."

Now: build the pipeline with one native hit group per reflected hit group, take
`MaxPayloadSizeInBytes` from the largest reflected payload (D3D12 counts 4 bytes per scalar and
ignores the field once payload access qualifiers are emitted on Shader Model 6.7+; Vulkan's
`maxPipelineRayPayloadSize` is the byte size and is supplied only for pipeline-library pipelines)
and `MaxAttributeSizeInBytes` from `getNativeHitAttributeSize()`, and write the SBT as a
conventional DXR engine does: one record per `(instance, geometry, rayType)` with the group's shader
identifier and its record bytes, one miss record per `missIndex`, and
`InstanceContributionToHitGroupIndex` per instance. Nothing changes in code generation on these
targets: `trace` still lowers to the native `TraceRay` with the three `RayTraversalDesc` fields.

slang-rhi's `ShaderRecordOverwrite` carries at most 8 inline bytes at a hard-coded offset, so the
revision adds, append-only, a per-record data pointer on `ShaderTableDesc`
(`ShaderRecordData { const void* data; uint32_t size; }` as `hitGroupRecordData`,
`missShaderRecordData`, `callableShaderRecordData`), copied immediately after the shader
identifier. Until it lands, the D3D12/Vulkan tests for per-record data use a raw SBT.

### 8.3 Host Reflection Patterns (3.3)

Was:

```cpp
struct ReflectedHitGroup
{
    int slot;
    ...
    EntryPointReflection* metalCandidateFunction;
    EntryPointReflection* metalClosestHitVisibleFunction;
};
```

Now (expected shape; names are illustrative as before):

```cpp
struct ReflectedTraceProgramSchema
{
    TypeReflection* traceContextType;
    List<ReflectedPayload> payloads;             // payloads(Schema)
    List<ReflectedCallableShader> callableShaders;   // in function-index order
    bool hitGroupsOpen, missShadersOpen, callableShadersOpen;
    uint nativeHitAttributeSize;                 // includes the 8-byte built-in triangle attributes
    uint hitRecordStride, missRecordStride, callableRecordStride;   // compile-time; Metal buffer layout, also usable on D3D12/Vulkan
    // Metal only
    uint metalRecordHeaderSize;                  // 16
    List<ReflectedDescriptorField> metalDescriptorFields;           // every synthesized field of Section 4.1 with its binding and, for
                                                                    //   per-payload tables, the payload it serves
};

struct ReflectedPayload
{
    const char* name;                            // qualified payload type name; the stable key for hosts
    TypeReflection* payloadType;
    uint payloadSize;                            // under the target's payload rules
    List<ReflectedHitGroup> hitGroups;           // in function-index order within this payload
    List<ReflectedMissShader> missShaders;       // { functionIndex, name, linked, contextType, recordType, recordTypeLayout, miss }
    // Metal only
    ReflectedDescriptorField* metalClosestHitTable;             // the descriptor fields holding this payload's tables;
    ReflectedDescriptorField* metalMissTable;                   //   the host binds by field, never by a number
    ReflectedDescriptorField* metalIntersectionTable;           //   (always present; an empty table when there is no candidate logic)
    List<ReflectedMetalDispatcher> metalCandidateDispatchers;   // { entryPointName, primitiveKind, tableIndex }
    EntryPointReflection* metalEmptyClosestHit;                 // null when no NoClosestHit group exists for this payload
};

struct ReflectedHitGroup
{
    int functionIndex;                           // ordinal within this payload's hit section; the Metal table index and record identity
    const char* name;                            // qualified group type name; the stable key for hosts
    bool linked;                                 // joined through an open section
    TypeReflection* contextType;
    TypeReflection* recordType;
    TypeLayoutReflection* recordTypeLayout;      // the exact layout generated code reads on this target
    TypeReflection* intersectionAttributesType;
    EntryPointReflection* closestHit;            // exact emitted names on every target
    EntryPointReflection* anyHit;
    EntryPointReflection* intersection;
};
```

The `getSlot()` C entry points and wrappers are deleted rather than stubbed: the structural
ray-tracing reflection exists only on the unreleased branch, so no released caller can depend on
them, and a stub returning -1 would be a second spelling of "no slot". Hit groups and miss shaders
are reached through their payload because the function index is meaningful only within one.
Program-wide enumeration of entry declarations independent of any schema (plugin discovery) is also
exposed; on such entries `getFunctionIndex()` returns -1 and `isLinked()` returns false.

Pattern A (native SBT) and Pattern C (Metal IFT) in 3.3 become the loops in 8.1 and 8.2 above.
Pattern B (future IFB) loses `slot` the same way. Pattern D (manual host construction without
reflection) remains possible for schemas whose sections are all closed: the payloads and the
per-payload declaration order are visible in source, and positions were always the host's. A schema
with an open section requires reflection of the linked program.

## 9. Diagnostics

Removed: `InvalidStructuralRayTracingGroupSlot` and `DuplicateStructuralRayTracingGroupSlot` (the
negative and duplicate slot errors).

Added, all at structural synthesis, where linked entries are first visible:

| Diagnostic | Trigger |
| --- | --- |
| `StructuralRayTracingEntryTraceContextMismatch` | a listed or linked entry's `Context.TraceContext` is not the schema's trace context |
| `StructuralRayTracingPayloadNotServed` | the payload passed to `trace` is not `Context.Payload` of any hit group or miss shader in the schema |
| `StructuralRayTracingEmptyPayloadValue` | a value of an empty payload type appears in user code: as a `trace` argument, as an explicit generic argument to `trace`, or as `input.payload` in a stage whose context names it |
| `StructuralRayTracingAmbiguousEmptyPayload` | `payloads(Schema)` contains two or more distinct empty types, so `trace(desc, as, descriptor)` would not name one partition; the contexts must share one empty type |
| `DuplicateStructuralRayTracingEntry` | the same entry type listed twice in one pack (a listed entry that also conforms to the section's tag is not a duplicate) |
| `StructuralRayTracingOpenTagNotEntryInterface` | the `Tag` of an open list does not inherit the section's entry interface (`IHitGroup`, `IMissShader`, or `ICallableShader`) |
| `StructuralRayTracingRecordNotPlainData` | a `Record` or `Payload` type is outside the portable plain-data subset: opaque resource types, runtime-sized arrays, pointers, atomics, non-copyable types, or `void` as a payload |

Changed: `StructuralRayTracingCallableDataMismatch` now reports the callable shader's name instead
of the slot and runs over the linked callable set;
`StructuralRayTracingMetalCandidateGlobalParameter` (Metal finalize) now also covers the dispatcher
arms.

The trace-context check is needed regardless of open sections: the draft module's `where expand
each Group.Context.TraceContext == TraceContext` constraint is silently unenforced by the current
front end. An entry whose context names a trace context with a different payload, listed in a
schema for another payload, compiles for SPIR-V and Metal without a diagnostic, and the Metal output
installs a function that writes the second payload's field into a `ray_data` struct holding the
first. This was reproduced with the draft compiler for both a hit group in a `HitGroupList` and a
miss group in the draft's `MissGroupList`. That is a front-end bug to fix in its own right; with the
constraints removed from the lists, the synthesis check is the one mechanism, and it also covers
linked entries.

A front-end bug limits generic stages, independently of this revision. A generic stage whose
`where` clauses constrain associated types of its context parameter, such as the example's
`AlphaTest<C>` (`C.Primitive == rt::TrianglePrimitive`, `C.Record == MaterialRecord`) or a stage
with `C.Payload : ISurfacePayload`, is accepted as a variable type and as an ordinary generic
argument, but its specialization is rejected with E38029 when it is the witness of a hit group's
associated type (`typealias AnyHit = AlphaTest<PrimaryMeshContext>;`). The draft's
generic-parameter stage contracts fail identically, and a constraint of the form `C : ISomething`
passes, so the defect is in how associated-type witnesses are checked against a generic's
`where` clauses, not in the shape chosen here. Generic-stage coverage depends on #12822 / PR
#12827; this proposal does not define a workaround.

What these checks do and do not establish about payloads is stated in the payload model, Section
5: every stage is typed on its own context's payload, a hit group's three stages on one, and no
trace names a payload the schema does not serve; whether the host placed a `P` entry where a `P`
trace looks is not checkable at compile time and is one host rule that Slang does not check; a
validation mode that would detect it at run time is recorded as future work (payload model, Section
13). What no compiler can check on
any native API otherwise stays with the host, optionally aided by a host-side SBT validator built
from reflection: record indices inside their section, every record naming a populated table entry
of its own payload or the empty sentinel, record bytes matching the record's entry, and every
geometry kind in a traced acceleration structure having a dispatcher in the tracing program.

## 10. Unchanged Sections And Affected Companion Documents

Unchanged in PROPOSAL.md: 1.1, 1.2, 1.4; 2.2.2 (the descriptor as an opaque, compiler-specialized
resource); 2.5's inference sources (aggregation now runs over the linked set per payload); the
stage semantics of 2.3, and within 2.3.1
the composition rules and the `reportHit` accumulator semantics (Section 5 lists what does change
in 2.3.1); 2.4; every `RayTraversalDesc` field list.

Affected in PROPOSAL.md beyond the sections above: 2.2.3's `ITraceContext` (loses `Payload`), its
hit and miss contexts (gain `Payload`), and its example's two `typealias Slot` lines; 2.5.1's "one
trace context has one `AccelerationStructure` type" argument is unchanged, but the payload sentences
in 2.2.3 move to the contexts.

Companion-document disposition for this revision:

- METAL_TAG_LIST_ANALYSIS.md: its `ITraceContext` snippet loses `associatedtype Payload` and its
  example trace context loses `typealias Payload = RadiancePayload`; the inference decision is
  unaffected.
- RAY_PAYLOAD_MODEL.md is retired. PAYLOAD_SAFE_TRACE_VARIANTS.md is the payload model under this
  revision: one payload per hit or miss context, `trace` generic on the payload it passes, the
  portable payload rules carried over from the retired document, and an explicit statement of what
  is and is not checked.
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) incorporates the schema terminology, associated
  stage contexts, atomic payload-and-record cutover, function-indexed Metal lowering, and revised
  tests described here. References to the former slot-based plan are historical, not pending work.
- CALLABLE_SHADER_CONCERNS.md: the `typealias Slot = rt::CallableSlot<0>` example, the phrases
  "invoke a dynamically selected slot", "each slot resolves its own record", "runtime SBT slot", and
  "per-slot records", the rename `ICallableGroupContext` to `ICallableContext`, and the removal of
  `ICallableGroup` (the schema lists `ICallableShader` types, each naming its `Context`). The stage
  semantics are unchanged.
- TUTORIAL.md sections 2 (trace context loses the payload), 3 and 5 (contexts gain a payload;
  `IMissGroupContext` becomes `IPayloadContext`; every stage declares `typealias Context` in place
  of a generic argument; the miss and callable wrappers disappear and the schema lists the shaders),
  4, 5, and 6 (slot lines), and 10 (table sizing and record placement).
- The draft implementation: `ITraceProgramLayout` becomes `ITraceProgramSchema`,
  `program-layout.slang` becomes `program-schema.slang`, and `SlangReflectionTraceProgramLayout`
  with its `spReflectionTraceProgramLayout_*` entry points become `...Schema...`; a mechanical
  rename over the public module surface, feature-owned compiler metadata and reflection, and
  `tests/ray-tracing-2/`, done before the contract ships rather than as a breaking rename after.
  Internal IR operands, accessors such as `getProgramLayout()`, and lowering vocabulary remain until
  the semantic cutover.
- The workspace README.md ("logical slot and context", "logical-slot tables") and the Cornell demo
  README ("allocate each native table through the largest slot"); the demo's ray-type convention
  constants become `kPrimaryRayType` and `kShadowRayType`.

## 11. Revised Open Design Questions

Retained from PROPOSAL 4: exact reflection API names and ownership model; whether and how to add
the IFB lowering; how much runtime validation Slang should provide. Generated entry-point naming is
resolved in principle, reflection returns the exact emitted symbol on every target, but the
emitter-side export path and the qualification rule for a stage that participates in several
schemas or payloads with different `ray_data` types must be fixed before the names hosts read are
frozen.

Added:

- **Whether the table convention should be reflectable from the shader.** `sbtOffset`, `sbtStride`,
  and `missIndex` are runtime fields and the engine keeps its convention on both sides by itself.
  A shader-declared, reflected convention was tried in an intermediate revision and dropped because
  it fixed the interleave the host owns and added a declaration whose only remaining check the
  entries already provide. If wanted later, it should be an optional annotation, not a required
  list.
- **A validation mode** that detects a record whose entry belongs to another payload (a payload tag
  in each Metal record compared by the generated dispatch; a tagged payload wrapper on D3D12 and
  Vulkan compared by the generated entry point). Not in scope for this revision.
- **A warning for a payload served only by hit groups or only by miss shaders** in a schema. Legal
  today; the host writes the empty sentinel for records nothing should select.
- **Payload access qualifier inference** from the stages of a payload, now that the compiler knows
  exactly which stages are typed on a payload (the only stages a trace of it invokes when the host
  follows the rule of the payload model's Section 6).
- **Adopting the traversal-axes split** of PAYLOAD_SAFE_TRACE_VARIANTS.md Section 11 (acceleration
  structure inferred from the trace argument, motion from the traversal descriptor), which would
  remove `ITraceContext` entirely.
- **Metal descriptor synthesis.** The lowering moves from rewriting five fixed field types to
  synthesizing a struct with `3 * payloadCount + 2` fields. The placeholder in `descriptor.slang`
  can stay fixed only if the lowering replaces the struct wholesale; otherwise the module needs a
  per-payload nested placeholder. The host writes the fields in reflected order either way.
- Metal dispatch cost of the record indirection (one extra dependent load on the miss path; record
  lookup plus switch in candidates). To be measured with the Cornell perf harness before the
  contract change lands.
- Link-time enumeration of tagged conformances and construction of entry metadata from IR witness
  tables rather than AST witnesses; the largest new compiler piece.
- Whether the front end accepts an interface type as the `Tag` generic argument as written, or
  needs an explicit rule.
- The dispatcher under `max_levels<N>` (innermost instance id) and the MSL attribute form for
  intersection functions in that configuration.
- Record layout portability: D3D/Vulkan records use constant-buffer packing, Metal natural layout.
  Whether to mandate one portable rule for `Record`.
- Whether candidate logic on Metal should gain access to global parameters through an argument
  buffer bound to the dispatcher.
