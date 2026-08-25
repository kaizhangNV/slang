# Tutorial: Declaring A Structural Ray-Tracing Program

This tutorial builds a minimal triangle hit/miss pipeline with the structural ray-tracing API. The
same source layout maps to native D3D/Vulkan SBT records and generated Metal function-table
dispatch.

## 1. Import The Module

The API is an experimental standard module:

```slang
import slang.raytracing;
```

Compile with `-experimental-feature`. Importing the module is explicit; ordinary shaders do not pay
its loading or lowering cost.

## 2. Define The Trace Context

The trace context fixes the payload, acceleration-structure model, and motion model shared by one
trace program:

```slang
struct RadiancePayload
{
    float3 color;
}

struct PrimaryTraceContext : rt::ITraceContext
{
    typealias Payload = RadiancePayload;
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}
```

`rt::AccelerationStructure` is the portable instanced scene model. Metal-only programs can instead
select `rt::MultiLevelAccelerationStructure<N>` when they require a different hierarchy depth.

## 3. Write A Hit Stage

First define the context shared by the stages in one hit group:

```slang
struct TriangleHitContext : rt::IHitContext
{
    typealias TraceContext = PrimaryTraceContext;
    typealias Primitive = rt::TrianglePrimitive;
    typealias Record = void;
}
```

Then implement _ClosestHit_ as a method rather than a free-standing entry point:

```slang
struct ShadeTriangle : rt::IClosestHitShader<TriangleHitContext>
{
    void invoke(rt::ClosestHitInput<TriangleHitContext> input)
    {
        float2 barycentrics = input.triangle.barycentricCoord;
        input.payload.color = float3(barycentrics, 1.0);
    }
}
```

The input is a compiler-provided view. Its members are intrinsic properties, not stored fields.
Only properties reachable from `invoke` contribute native entry parameters or Metal tags.

## 4. Declare The Hit Group

A hit group gives the logical SBT record a slot and identifies its stage implementations:

```slang
struct TriangleHitGroup : rt::IHitGroup
{
    typealias Slot = rt::HitGroupSlot<0>;
    typealias Context = TriangleHitContext;
    typealias ClosestHit = ShadeTriangle;
    typealias AnyHit = rt::NoAnyHit<TriangleHitContext>;
    typealias Intersection = rt::NoIntersection<TriangleHitContext>;
}
```

The placeholders state that this group has no source _AnyHit_ or _Intersection_ logic. They do not
consume physical SBT or Metal function-table entries.

For procedural geometry, use `rt::BoundingBoxPrimitive<Attributes>` in the hit context and provide
an `IIntersectionShader`. The shader reports candidates through `input.reportHit(...)`; an optional
_AnyHit_ stage can accept, ignore, or end the search for each reported candidate.

## 5. Declare The Miss Group

_Miss_ has its own context and record type:

```slang
struct PrimaryMissContext : rt::IMissGroupContext
{
    typealias TraceContext = PrimaryTraceContext;
    typealias Record = void;
}

struct ShadeMiss : rt::IMissShader<PrimaryMissContext>
{
    void invoke(rt::MissInput<PrimaryMissContext> input)
    {
        input.payload.color = float3(0.0);
    }
}

struct PrimaryMissGroup : rt::IMissGroup
{
    typealias Slot = rt::MissSlot<0>;
    typealias Context = PrimaryMissContext;
    typealias Miss = ShadeMiss;
}
```

## 6. Assemble The Program Layout

`ITraceProgramLayout` is the source-of-truth for the logical SBT:

```slang
struct PrimaryProgramLayout : rt::ITraceProgramLayout
{
    typealias TraceContext = PrimaryTraceContext;
    typealias HitGroups =
        rt::HitGroupList<PrimaryTraceContext, TriangleHitGroup>;
    typealias MissGroups =
        rt::MissGroupList<PrimaryTraceContext, PrimaryMissGroup>;
    typealias CallableGroups =
        rt::NoCallableGroups<PrimaryTraceContext>;
}
```

Each list is finite and statically typed. Slang uses it to retain the selected source stages,
synthesize physical entries, report reflection, and reject duplicate or incompatible slots.

## 7. Bind The Descriptor

The program descriptor is a resource whose physical target layout depends on the selected program:

```slang
struct FrameParameters
{
    rt::AccelerationStructure scene;
    rt::TraceProgramDescriptor<PrimaryProgramLayout> program;
}

ParameterBlock<FrameParameters> frame;
```

On D3D/Vulkan, the native pipeline and SBT own stage dispatch, so the descriptor has no additional
physical shader binding. On Metal, it specializes to parameter-block-like IFT, visible-function
table, and record-buffer resources. Normal binding reflection exposes that physical layout.

## 8. Trace A Ray

Ray-generation code supplies the ordinary trace parameters and payload:

```slang
[shader("raygeneration")]
void rayGen()
{
    rt::RayTraversalDesc desc = {};
    desc.ray.origin = float3(0.0, 0.0, 0.0);
    desc.ray.direction = float3(0.0, 0.0, 1.0);
    desc.ray.tMin = 0.001;
    desc.ray.tMax = 1000.0;
    desc.rayFlags = RAY_FLAG_NONE;
    desc.instanceMask = 0xff;
    desc.sbtOffset = 0;
    desc.sbtStride = 1;
    desc.missIndex = 0;

    RadiancePayload payload = {};
    rt::RayTracer<PrimaryProgramLayout> tracer;
    tracer.trace(desc, frame.scene, frame.program, payload);
}
```

The layout type is the key that associates this trace with its possible stages. D3D/Vulkan lower
the call to their existing native trace operation. Metal lowers it to traversal followed by
generated _ClosestHit_ or _Miss_ visible-function dispatch.

Runtime-valued `RAY_FLAG` bits are portable. Metal emits a small helper sequence that configures the
corresponding intersector controls before traversal.

## 9. Compile A Stage By Itself

A stage struct can also be selected without compiling a complete layout:

```text
slangc shader.slang -experimental-feature \
    -entry ShadeTriangle -stage closesthit -target spirv
```

The entry-point name is the struct name. Slang synthesizes the native signature from the reachable
input properties and discards unrelated stages after entry synthesis.

## 10. Build Target Resources

Use structural reflection to enumerate logical hit, miss, and callable slots.

- D3D12/Vulkan: compile the synthesized native stage entries, create the pipeline, and build SBT
  records whose indices match the reflected slots.
- Metal: install generated candidate functions in the IFT, install _ClosestHit_, _Miss_, and
  _Callable_ functions in their visible-function tables, and populate the reflected record buffer.

The host never binds source _AnyHit_ and _Intersection_ stages as separate Metal resources. Slang
combines the required candidate logic into the generated IFT function for each hit group.

Complete executable cases are under
[`tests/ray-tracing-2/runtime/shaders`](../../../../tests/ray-tracing-2/runtime/shaders), with
focused target and diagnostic coverage in the rest of `tests/ray-tracing-2`.
