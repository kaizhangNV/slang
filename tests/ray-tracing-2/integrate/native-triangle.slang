//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry RayGeneration -stage raygeneration -entry TriangleClosestHit -stage closesthit -entry BackgroundMiss -stage miss -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry RayGeneration -stage raygeneration -entry TriangleClosestHit -stage closesthit -entry BackgroundMiss -stage miss -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry TriangleClosestHit -stage closesthit -entry BackgroundMiss -stage miss -target metal -profile metallib_2_4 -validate-ir
//DIAGNOSTIC_TEST:SIMPLE(filecheck=METAL_RAYGEN): -experimental-feature -entry RayGeneration -stage raygeneration -entry TriangleClosestHit -stage closesthit -entry BackgroundMiss -stage miss -target metal -profile metallib_2_4
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry RayGeneration -stage raygeneration -entry TriangleClosestHit -stage closesthit -entry BackgroundMiss -stage miss -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTX): -experimental-feature -entry RayGeneration -stage raygeneration -entry TriangleClosestHit -stage closesthit -entry BackgroundMiss -stage miss -target ptx -validate-ir

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
    typealias Payload = float4;
    typealias Primitive = rt::TrianglePrimitive;
    typealias Record = void;
}

struct MissContext : rt::IPayloadContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = float4;
    typealias Record = void;
}

struct TriangleClosestHit : rt::IClosestHitShader
{
    typealias Context = HitContext;

    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        // Reading the attributes verifies that the adapter uses native hit-attribute storage,
        // not an uninitialized local that merely has the same float2 type.
        let triangle = input.attributes;
        input.payload = float4(triangle.barycentricCoord, input.distance, 1.0);
    }
}

struct BackgroundMiss : rt::IMissShader
{
    typealias Context = MissContext;

    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload = float4(-1.0, -1.0, -1.0, 0.0);
    }
}

struct TriangleGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = TriangleClosestHit;
    typealias AnyHit = rt::NoAnyHit<Context>;
    typealias Intersection = rt::NoIntersection<Context>;
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::HitGroupList<TriangleGroup>;
    typealias MissShaders = rt::MissShaderList<BackgroundMiss>;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

rt::AccelerationStructure sceneBVH;
rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<float4> resultBuffer;

// The host builds a single triangle at z = 1 and one hit/miss SBT record. The first ray hits
// barycentrics (0.25, 0.25) at distance 1; the second misses. Neither result equals the initial
// payload, so the runtime test checks both native stage dispatch paths and caller-visible writes.
[shader("raygeneration")]
void RayGeneration()
{
    uint index = DispatchRaysIndex().x;
    rt::RayTraversalDesc desc = {};
    desc.ray.Origin = index == 0 ? float3(0.25, 0.25, 0.0) : float3(2.0, 2.0, 0.0);
    desc.ray.Direction = float3(0.0, 0.0, 1.0);
    desc.ray.TMin = 0.0;
    desc.ray.TMax = 10.0;
    desc.instanceMask = 0xff;
    desc.sbtStride = 1;

    float4 payload = float4(17.0);
    rt::RayTracer<Schema> tracer;
    tracer.trace(desc, sceneBVH, program, payload);
    resultBuffer[index] = payload;
}

// HLSL-DAG: TraceRay(
// HLSL-DAG: [shader("closesthit")]void TriangleClosestHit_closesthit(
// HLSL-DAG: [shader("miss")]void BackgroundMiss_miss(
// SPIRV-DAG: OpEntryPoint RayGenerationKHR
// SPIRV-DAG: OpEntryPoint ClosestHitKHR
// SPIRV-DAG: OpEntryPoint MissKHR
// SPIRV-DAG: OpVariable {{.*}} HitAttributeKHR
// SPIRV-DAG: OpTraceRayKHR

// METAL: visible]] void TriangleClosestHit_closesthit(
// METAL: TriangleClosestHit_invoke_{{[0-9]+}}((float4 thread*)((ulong)(payload_{{[0-9]+}})), barycentricCoord_{{[0-9]+}}, distance_
// METAL: visible]] void BackgroundMiss_miss(
// METAL: BackgroundMiss_invoke_{{[0-9]+}}((float4 thread*)((ulong)(payload_

// METAL_RAYGEN: error[E36107]: unavailable features in entry point
// METAL_RAYGEN: note: see using of 'DispatchRaysIndex'

// OptiX supplies triangle attributes to closest-hit and writes both stages' payload results back.
// CUDA: __raygen__RayGeneration()
// CUDA: optixTrace(
// CUDA: __closesthit__TriangleClosestHit_closesthit()
// CUDA: optixGetPayload_0()
// CUDA: optixGetAttribute_0()
// CUDA: optixGetAttribute_1()
// CUDA: TriangleClosestHit_invoke_
// CUDA: optixSetPayload_0(
// CUDA: __miss__BackgroundMiss_miss()
// CUDA: BackgroundMiss_invoke_
// CUDA: optixSetPayload_0(
// PTX: .visible .entry __raygen__RayGeneration()
// PTX: _optix_trace_typed_
// PTX: .visible .entry __closesthit__TriangleClosestHit_closesthit()
// PTX: _optix_get_attribute_0
// PTX: _optix_get_attribute_1
// PTX: _optix_set_payload
// PTX: .visible .entry __miss__BackgroundMiss_miss()
// PTX: _optix_set_payload
