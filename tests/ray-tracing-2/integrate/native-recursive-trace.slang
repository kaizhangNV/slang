//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry RayGeneration -stage raygeneration -entry RecursiveClosestHit -stage closesthit -entry SecondaryMiss -stage miss -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry RayGeneration -stage raygeneration -entry RecursiveClosestHit -stage closesthit -entry SecondaryMiss -stage miss -target spirv-assembly -emit-spirv-directly -validate-ir
//TEST:SIMPLE(filecheck=DXIL): -experimental-feature -entry RecursiveClosestHit -stage closesthit -target dxil -profile sm_6_6 -validate-ir

import slang.raytracing.portable;

#ifndef TEST_NOINLINE_TRACE
#define TEST_NOINLINE_TRACE 1
#endif

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
    typealias Payload = uint;
    typealias Record = void;
}

// The secondary ray starts beyond the only triangle and travels away from it. Its uint payload
// therefore reaches only SecondaryMiss; the outer float4 payload reaches only RecursiveClosestHit.
// Runtime SBT selection, not the schema, determines the concrete payload contract of each trace.
// Only the D3D runtime harness disables this attribute: pipeline creation fails on its driver,
// although DXIL validation succeeds. See https://github.com/shader-slang/slang/issues/14077.
// The DXIL compile-only test and other runtime targets keep the retained helper.
#if TEST_NOINLINE_TRACE
[noinline]
#endif
void traceSecondary(inout uint payload)
{
    rt::RayTraversalDesc desc = {};
    desc.ray.Origin = float3(0.25, 0.25, 2.0);
    desc.ray.Direction = float3(0.0, 0.0, 1.0);
    desc.ray.TMax = 10.0;
    desc.instanceMask = 0xff;
    desc.sbtStride = 1;
    rt::RayTracer<Schema> tracer;
    tracer.trace(desc, sceneBVH, program, payload);
}

struct RecursiveClosestHit : rt::IClosestHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        // Writes before and after the recursive trace distinguish its payload from the caller's.
        // Builtins read afterward must still describe this outer committed triangle.
        input.payload.x = 100.0 + float(input.dispatchRaysIndex.x);
        uint secondaryPayload = 5;
        traceSecondary(secondaryPayload);
        input.payload.y = float(secondaryPayload);
        input.payload.z = input.distance;
        input.payload.w = input.attributes.barycentricCoord.x + float(input.dispatchRaysDimensions.x);
    }
}

struct SecondaryMiss : rt::IMissShader
{
    typealias Context = MissContext;
    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload += 10 + input.dispatchRaysIndex.x + 100 * input.dispatchRaysDimensions.x;
    }
}

struct TriangleGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = RecursiveClosestHit;
    typealias AnyHit = rt::NoAnyHit<Context>;
    typealias Intersection = rt::NoIntersection<Context>;
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::HitGroupList<TriangleGroup>;
    typealias MissShaders = rt::MissShaderList<SecondaryMiss>;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

rt::AccelerationStructure sceneBVH;
rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<float4> resultBuffer;

// Both launch lanes hit barycentrics (0.25, 0.25) at distance 1. The host permits depth two and
// checks each outer payload after ClosestHit -> secondary trace -> Miss has completed.
[shader("raygeneration")]
void RayGeneration()
{
    uint index = DispatchRaysIndex().x;
    rt::RayTraversalDesc desc = {};
    desc.ray.Origin = float3(0.25, 0.25, 0.0);
    desc.ray.Direction = float3(0.0, 0.0, 1.0);
    desc.ray.TMax = 10.0;
    desc.instanceMask = 0xff;
    desc.sbtStride = 1;
    float4 payload = float4(17.0);
    rt::RayTracer<Schema> tracer;
    tracer.trace(desc, sceneBVH, program, payload);
    resultBuffer[index] = payload;
}

// HLSL-DAG: TraceRay(
// HLSL-DAG: traceSecondary
// HLSL-DAG: [shader("closesthit")]void RecursiveClosestHit_closesthit(
// HLSL-DAG: [shader("miss")]void SecondaryMiss_miss(
// SPIRV-DAG: OpEntryPoint RayGenerationKHR
// SPIRV-DAG: OpEntryPoint ClosestHitKHR
// SPIRV-DAG: OpEntryPoint MissKHR
// SPIRV-DAG: OpVariable {{.*}} HitAttributeKHR
// SPIRV-DAG: OpTraceRayKHR
// DXIL-DAG: Function Attrs: noinline
// DXIL-DAG: traceSecondary
// DXIL-DAG: @dx.op.traceRay
