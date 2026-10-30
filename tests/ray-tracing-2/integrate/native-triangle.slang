//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry RayGeneration -stage raygeneration -entry TriangleClosestHit -stage closesthit -entry BackgroundMiss -stage miss -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry RayGeneration -stage raygeneration -entry TriangleClosestHit -stage closesthit -entry BackgroundMiss -stage miss -target spirv-assembly -emit-spirv-directly

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
