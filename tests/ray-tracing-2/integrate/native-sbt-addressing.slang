//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry RayGeneration -stage raygeneration -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry RayGeneration -stage raygeneration -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -target metal -entry WrongHit -stage closesthit -entry FirstHit -stage closesthit -entry SecondHit -stage closesthit -entry WrongMiss -stage miss -entry SelectedMiss -stage miss -entry WrongCallable -stage callable -entry SelectedCallable -stage callable -profile metallib_2_4 -validate-ir
//DIAGNOSTIC_TEST:SIMPLE(filecheck=METAL_RAYGEN): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry RayGeneration -stage raygeneration -target cuda -entry WrongHit -stage closesthit -entry FirstHit -stage closesthit -entry SecondHit -stage closesthit -entry WrongMiss -stage miss -entry SelectedMiss -stage miss -entry WrongCallable -stage callable -entry SelectedCallable -stage callable -validate-ir
//TEST:SIMPLE(filecheck=PTX): -experimental-feature -entry RayGeneration -stage raygeneration -target ptx -entry WrongHit -stage closesthit -entry FirstHit -stage closesthit -entry SecondHit -stage closesthit -entry WrongMiss -stage miss -entry SelectedMiss -stage miss -entry WrongCallable -stage callable -entry SelectedCallable -stage callable -validate-ir

// Metal stage code generation is checked below; full raygen compilation diagnoses the missing
// portable launch-index query because legacy DispatchRaysIndex() does not support Metal.
import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint4;
    typealias Record = void;
    typealias Primitive = rt::TrianglePrimitive;
}

struct CallableContext : rt::ICallableContext
{
    typealias TraceContext = ::TraceContext;
    typealias CallableData = uint;
    typealias Record = void;
}

struct WrongHit : rt::IClosestHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload.xy = uint2(999, input.geometryIndex);
    }
}

struct FirstHit : rt::IClosestHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload.xy = uint2(11, input.geometryIndex);
    }
}

struct SecondHit : rt::IClosestHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload.xy = uint2(22, input.geometryIndex);
    }
}

struct WrongMiss : rt::IMissShader
{
    typealias Context = HitContext;
    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload.xy = uint2(998, 0xffffffff);
    }
}

struct SelectedMiss : rt::IMissShader
{
    typealias Context = HitContext;
    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload.xy = uint2(33, 0xffffffff);
    }
}

struct WrongCallable : rt::ICallableShader
{
    typealias Context = CallableContext;
    static void invoke(in rt::CallableInput<Context> input)
    {
        input.data = 997;
    }
}

struct SelectedCallable : rt::ICallableShader
{
    typealias Context = CallableContext;
    static void invoke(in rt::CallableInput<Context> input)
    {
        input.data += 77;
    }
}

struct WrongGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = WrongHit;
    typealias AnyHit = rt::NoAnyHit<Context>;
    typealias Intersection = rt::NoIntersection<Context>;
}

struct FirstGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = FirstHit;
    typealias AnyHit = rt::NoAnyHit<Context>;
    typealias Intersection = rt::NoIntersection<Context>;
}

struct SecondGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = SecondHit;
    typealias AnyHit = rt::NoAnyHit<Context>;
    typealias Intersection = rt::NoIntersection<Context>;
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::HitGroupList<WrongGroup, FirstGroup, SecondGroup>;
    typealias MissShaders = rt::MissShaderList<WrongMiss, SelectedMiss>;
    typealias CallableShaders = rt::CallableShaderList<WrongCallable, SelectedCallable>;
}

rt::AccelerationStructure sceneBVH;
rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<uint4> resultBuffer;

// The host puts two geometries in one BLAS and gives its instance an SBT contribution of one.
// Hit records are intentionally sparse: 1 + geometryIndex * 2 + 1 selects physical records 2/4,
// while the corresponding shader functions have schema indices 1/2. Neither index is a binding.
[shader("raygeneration")]
void RayGeneration()
{
    uint index = DispatchRaysIndex().x;
    rt::RayTraversalDesc desc = {};
    desc.ray.Origin = float3(0.25 + float(index) * 2.0, 0.25, 0.0);
    desc.ray.Direction = float3(0.0, 0.0, 1.0);
    desc.ray.TMax = 10.0;
    desc.instanceMask = 0xff;
    desc.sbtOffset = 1;
    desc.sbtStride = 2;
    desc.missIndex = 1;

    uint4 payload = uint4(0);
    rt::RayTracer<Schema> tracer;
    tracer.trace(desc, sceneBVH, program, payload);
    uint data = 5;
    tracer.callShader<CallableContext>(1, program, data);
    resultBuffer[index] = uint4(payload.xy, data, index);
}

// HLSL-DAG: TraceRay(
// HLSL-DAG: CallShader(1U,
// SPIRV-DAG: OpTraceRayKHR
// SPIRV-DAG: OpExecuteCallableKHR

// METAL-DAG: visible]] void WrongHit_closesthit(
// METAL-DAG: visible]] void FirstHit_closesthit(
// METAL-DAG: visible]] void SecondHit_closesthit(
// METAL-DAG: visible]] void WrongMiss_miss(
// METAL-DAG: visible]] void SelectedMiss_miss(
// METAL-DAG: visible]] void WrongCallable_callable(
// METAL-DAG: visible]] void SelectedCallable_callable(

// METAL_RAYGEN: error[E36107]: unavailable features in entry point
// METAL_RAYGEN: note: see using of 'DispatchRaysIndex'

// OptiX preserves the SBT addressing operands and exports every shader selected by the host.
// CUDA: optixGetSbtGASIndex()
// CUDA: __raygen__RayGeneration()
// CUDA: ->sbtOffset_{{[0-9]+}} = 1U;
// CUDA: ->sbtStride_{{[0-9]+}} = 2U;
// CUDA: ->missIndex_{{[0-9]+}} = 1U;
// CUDA: optixTrace({{.*}}.sbtOffset_{{[0-9]+}}, {{.*}}.sbtStride_{{[0-9]+}}, {{.*}}.missIndex_
// CUDA: optixDirectCall<void>(1U,
// CUDA: __closesthit__FirstHit_closesthit()
// CUDA: __closesthit__SecondHit_closesthit()
// CUDA: __miss__SelectedMiss_miss()
// CUDA: __direct_callable__SelectedCallable_callable(
// PTX: .visible .entry __raygen__RayGeneration()
// PTX: _optix_trace_typed_
// PTX: _optix_call_direct_callable
// PTX: .visible .entry __closesthit__WrongHit_closesthit()
// PTX: _optix_read_sbt_gas_idx
// PTX: .visible .entry __closesthit__FirstHit_closesthit()
// PTX: _optix_read_sbt_gas_idx
// PTX: .visible .entry __closesthit__SecondHit_closesthit()
// PTX: _optix_read_sbt_gas_idx
// PTX: .visible .entry __miss__WrongMiss_miss()
// PTX: .visible .entry __miss__SelectedMiss_miss()
// PTX: .visible .func __direct_callable__WrongCallable_callable(
// PTX: .visible .func __direct_callable__SelectedCallable_callable(
