//TEST:SIMPLE(filecheck=CHECK): -experimental-feature -entry TraceEntry -stage miss -target metal -profile metallib_2_4 -validate-ir -dump-ir-after specializeModule
//TEST:SIMPLE(filecheck=LIVENESS): -experimental-feature -entry TraceEntry -stage miss -target metal -profile metallib_2_4 -validate-ir -dump-ir-after specializeModule
//TEST:COMPILE: tests/ray-tracing-2/ir/identity/schema-contract.slang -experimental-feature -o tests/ray-tracing-2/ir/identity/schema-contract-cache.slang-module
//TEST:SIMPLE_EX(filecheck=CHECK): -experimental-feature tests/ray-tracing-2/ir/identity/schema-contract-cache.slang-module -entry TraceEntry -stage miss -target metal -profile metallib_2_4 -validate-ir -dump-ir-after specializeModule
//TEST:SIMPLE_EX(filecheck=LIVENESS): -experimental-feature tests/ray-tracing-2/ir/identity/schema-contract-cache.slang-module -entry TraceEntry -stage miss -target metal -profile metallib_2_4 -validate-ir -dump-ir-after specializeModule
//TEST:SIMPLE(filecheck=CHECK,CUDA): -experimental-feature -entry TraceEntry -stage miss -target cuda -validate-ir -dump-ir-after specializeModule
//TEST:SIMPLE(filecheck=LIVENESS): -experimental-feature -entry TraceEntry -stage miss -target cuda -validate-ir -dump-ir-after specializeModule
//TEST:SIMPLE_EX(filecheck=CHECK,CUDA): -experimental-feature tests/ray-tracing-2/ir/identity/schema-contract-cache.slang-module -entry TraceEntry -stage miss -target cuda -validate-ir -dump-ir-after specializeModule
//TEST:SIMPLE_EX(filecheck=LIVENESS): -experimental-feature tests/ray-tracing-2/ir/identity/schema-contract-cache.slang-module -entry TraceEntry -stage miss -target cuda -validate-ir -dump-ir-after specializeModule
//TEST:SIMPLE(filecheck=PTX): -experimental-feature -entry TraceEntry -stage miss -target ptx -validate-ir -Xnvrtc -I"./external/optix-dev/include/"
//TEST:SIMPLE_EX(filecheck=PTX): -experimental-feature tests/ray-tracing-2/ir/identity/schema-contract-cache.slang-module -entry TraceEntry -stage miss -target ptx -validate-ir -Xnvrtc -I"./external/optix-dev/include/"

// The descriptor operands and list decorations must contain concrete types after specialization,
// including after loading a serialized module. They must not retain unselected shader bodies.

// Check both calls use the same specialized descriptor type. Its operands name the three
// concrete shader-list types checked below.
// CHECK: result code = 0
// CHECK: ### AFTER specializeModule:
// CHECK: func %copyDescriptor : Func(TraceProgramDescriptor(%Schema, [[WITNESS:%[A-Za-z0-9_]+]],
// CHECK-SAME: RaytracingAccelerationStructure(0 : Int), %rtx5FNoMotion, [[HITS:%[A-Za-z0-9_]+]], [[MISSES:%[A-Za-z0-9_]+]], [[CALLABLES:%[A-Za-z0-9_]+]]),
// CHECK-SAME: TraceProgramDescriptor(%Schema, [[WITNESS]],
// CHECK-SAME: RaytracingAccelerationStructure(0 : Int), %rtx5FNoMotion, [[HITS]], [[MISSES]], [[CALLABLES]]))
// CHECK: param %descriptor : TraceProgramDescriptor(%Schema, [[WITNESS]],
// CHECK-SAME: RaytracingAccelerationStructure(0 : Int), %rtx5FNoMotion, [[HITS]], [[MISSES]], [[CALLABLES]])):

// Miss and callable lists retain only their entry types and Record types. Hit-only operands
// are empty packs. The refined Miss interface still resolves to Record = uint.
// CHECK: [PortableRTShaderListDecoration(TypePack(%Callable), TypePack(Vec(Float, 4 : Int)), TypePack, TypePack, TypePack, TypePack, TypePack, TypePack)]
// CHECK-NEXT: [KnownBuiltinDecoration({{[0-9]+}} : Int)]
// CHECK-NEXT: [nameHint("rt.CallableShaderList")]
// CHECK-NEXT: struct [[CALLABLES]] : Type;
// CHECK: [PortableRTShaderListDecoration(TypePack(%Miss), TypePack(UInt), TypePack, TypePack, TypePack, TypePack, TypePack, TypePack)]
// CHECK-NEXT: [KnownBuiltinDecoration({{[0-9]+}} : Int)]
// CHECK-NEXT: [nameHint("rt.MissShaderList")]
// CHECK-NEXT: struct [[MISSES]] : Type;

// Each operand has one concrete type per group, in list order. SharedHit fills both stage
// slots in group zero; group one has only Intersection; group two has no stages.
// CHECK: [PortableRTShaderListDecoration(TypePack(%HitGroup, %ProceduralGroup, %OpaqueGroup),
// CHECK-SAME: TypePack(Void, Void, Void),
// CHECK-SAME: TypePack(%Payload, %Payload, %Payload),
// CHECK-SAME: TypePack(%rtx5FTrianglePrimitive, %rtx5FBoundingBoxPrimitive, %rtx5FTrianglePrimitive),
// CHECK-SAME: TypePack(%rtx5FTriangleData, %Attributes, %rtx5FTriangleData),
// CHECK-SAME: TypePack(%SharedHit, %rtx5FNoClosestHit, %rtx5FNoClosestHit),
// CHECK-SAME: TypePack(%SharedHit, %rtx5FNoAnyHit, %rtx5FNoAnyHit),
// CHECK-SAME: TypePack(%rtx5FNoIntersection, %Intersection, %rtx5FNoIntersection))]
// CHECK-NEXT: [KnownBuiltinDecoration({{[0-9]+}} : Int)]
// CHECK-NEXT: [nameHint("rt.HitGroupList")]
// CHECK-NEXT: struct [[HITS]] : Type;
// CHECK: let %program : TraceProgramDescriptor(%Schema, [[WITNESS]],
// CHECK-SAME: RaytracingAccelerationStructure(0 : Int), %rtx5FNoMotion, [[HITS]], [[MISSES]], [[CALLABLES]]) = global_param
// CHECK: let {{%[A-Za-z0-9_]+}} : TraceProgramDescriptor(%Schema, [[WITNESS]],
// CHECK-SAME: RaytracingAccelerationStructure(0 : Int), %rtx5FNoMotion, [[HITS]], [[MISSES]], [[CALLABLES]]) = call %copyDescriptor(%program)

// Scan the entire specialized module separately: the descriptor's type dependencies must not
// retain the unselected invoke bodies or their shared helper. Metal intentionally
// keeps global resource parameters for layout, so stageBodyOutput itself may remain.
// LIVENESS: result code = 0
// LIVENESS: ### AFTER specializeModule:
// LIVENESS-NOT: referencedStageBody
// LIVENESS-NOT: UnrelatedMiss
// LIVENESS-NOT: nameHint("{{(SharedHit|Intersection|Miss|Callable)}}.invoke")
// LIVENESS: ###

import slang.raytracing.portable;

RWStructuredBuffer<uint> stageBodyOutput;
[noinline]
void referencedStageBody()
{
    stageBodyOutput[0] = 42;
}

struct Payload { uint value; }
// One concrete type supplies both contracts. Ordinary lookup expressions must use
// the declared conformance even when the concrete type also implements another interface.
struct TraceContext : rt::ITraceContext, rt::IPayloadContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
    typealias TraceContext = ::TraceContext;
    typealias Payload = ::Payload;
    typealias Record = uint;
}
struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = ::Payload;
    typealias Record = void;
    typealias Primitive = rt::TrianglePrimitive;
}
typealias MissContext = TraceContext;
struct CallableContext : rt::ICallableContext
{
    typealias TraceContext = ::TraceContext;
    typealias CallableData = float2;
    typealias Record = float4;
}
struct Attributes { float coordinate; }
struct ProceduralContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = ::Payload;
    typealias Record = void;
    typealias Primitive = rt::BoundingBoxPrimitive<Attributes>;
}
// Both group members name this type. Keep both stage slots without retaining either method.
struct SharedHit : rt::IClosestHitShader, rt::IAnyHitShader
{
    typealias Context = HitContext;
    static void invoke(rt::ClosestHitInput<Context> input)
    {
        referencedStageBody();
        input.payload.value = 1;
    }
    static void invoke(rt::AnyHitInput<Context> input)
    {
        referencedStageBody();
        input.payload.value = 4;
    }
}
struct HitGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = SharedHit;
    typealias AnyHit = SharedHit;
    typealias Intersection = rt::NoIntersection<Context>;
}
struct Intersection : rt::IIntersectionShader
{
    typealias Context = ProceduralContext;
    static void invoke(rt::IntersectionInput<Context> input)
    {
        referencedStageBody();
    }
}
struct ProceduralGroup : rt::IHitGroup
{
    typealias Context = ProceduralContext;
    typealias ClosestHit = rt::NoClosestHit<Context>;
    typealias AnyHit = rt::NoAnyHit<Context>;
    typealias Intersection = ::Intersection;
}
struct OpaqueGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = rt::NoClosestHit<Context>;
    typealias AnyHit = rt::NoAnyHit<Context>;
    typealias Intersection = rt::NoIntersection<Context>;
}
interface IRefinedMiss : rt::IMissShader {}
struct Miss : IRefinedMiss
{
    typealias Context = MissContext;
    static void invoke(rt::MissInput<Context> input)
    {
        referencedStageBody();
        input.payload.value = 2;
    }
}
struct UnrelatedMiss : rt::IMissShader
{
    typealias Context = MissContext;
    static void invoke(rt::MissInput<Context> input)
    {
        input.payload.value = 3;
    }
}
struct Callable : rt::ICallableShader
{
    typealias Context = CallableContext;
    static void invoke(rt::CallableInput<Context> input)
    {
        referencedStageBody();
        input.data += input.record.xy;
    }
}
struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::HitGroupList<HitGroup, ProceduralGroup, OpaqueGroup>;
    typealias MissShaders = rt::MissShaderList<Miss>;
    typealias CallableShaders = rt::CallableShaderList<Callable>;
}
rt::TraceProgramDescriptor<Schema> program;
rt::AccelerationStructure scene;

// This descriptor is first lowered with an unknown schema. Specializing the helper
// must reuse the concrete descriptor type above, including all five schema-type operands.
[noinline]
rt::TraceProgramDescriptor<S> copyDescriptor<S : rt::ITraceProgramSchema>(
    rt::TraceProgramDescriptor<S> descriptor)
{
    return descriptor;
}

struct TraceEntry : rt::IMissShader
{
    typealias Context = MissContext;
    static void invoke(rt::MissInput<Context> input)
    {
        rt::RayTracer<Schema> tracer;
        tracer.trace({}, scene, program, input.payload);
        tracer.trace({}, scene, copyDescriptor<Schema>(program), input.payload);
    }
}

// Both descriptor spellings also lower to ordinary OptiX traces after specialization.
// CUDA-COUNT-2: optixTrace(
// CUDA: __miss__TraceEntry_miss()
// CUDA: optixSetPayload_0(
// PTX: .visible .entry __miss__TraceEntry_miss()
// PTX-COUNT-2: _optix_trace_
// PTX: _optix_set_payload
