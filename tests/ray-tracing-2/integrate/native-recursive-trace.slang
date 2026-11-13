//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry RayGeneration -stage raygeneration -entry RecursiveClosestHit -stage closesthit -entry SecondaryMiss -stage miss -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry RayGeneration -stage raygeneration -entry RecursiveClosestHit -stage closesthit -entry SecondaryMiss -stage miss -target spirv-assembly -emit-spirv-directly -validate-ir
//TEST:SIMPLE(filecheck=DXIL): -experimental-feature -entry RecursiveClosestHit -stage closesthit -target dxil -profile sm_6_6 -validate-ir
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RecursiveClosestHit -stage closesthit -entry SecondaryMiss -stage miss -target metal -profile metallib_2_4 -validate-ir
//DIAGNOSTIC_TEST:SIMPLE(filecheck=METAL_RAYGEN): -experimental-feature -entry RayGeneration -stage raygeneration -entry RecursiveClosestHit -stage closesthit -entry SecondaryMiss -stage miss -target metal -profile metallib_2_4
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry RayGeneration -stage raygeneration -entry RecursiveClosestHit -stage closesthit -entry SecondaryMiss -stage miss -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTX): -experimental-feature -entry RayGeneration -stage raygeneration -entry RecursiveClosestHit -stage closesthit -entry SecondaryMiss -stage miss -target ptx -validate-ir

// Metal stage code generation is checked below; full raygen compilation diagnoses the missing
// portable launch-index query because legacy DispatchRaysIndex() does not support Metal.
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

// METAL-DAG: metal::raytracing::intersector<
// METAL-DAG: [_slang_miss_function_index]({{.*}}invocationValue_{{[0-9]+}}, invocationValue_
// METAL-DAG: visible]] void RecursiveClosestHit_closesthit(
// METAL-DAG: RecursiveClosestHit_invoke_{{[0-9]+}}((float4 thread*)((ulong)(payload_{{[0-9]+}})), dispatchRaysIndex_{{[0-9]+}}, dispatchRaysDimensions_
// METAL-DAG: visible]] void SecondaryMiss_miss(
// METAL-DAG: SecondaryMiss_invoke_{{[0-9]+}}(dispatchRaysIndex_{{[0-9]+}}, dispatchRaysDimensions_{{[0-9]+}}, (uint thread*)((ulong)(payload_

// METAL_RAYGEN: error[E36107]: unavailable features in entry point
// METAL_RAYGEN: note: see using of 'DispatchRaysIndex'

// OptiX keeps the secondary trace in its helper and restores the outer invocation after the call.
// CUDA: __raygen__RayGeneration()
// CUDA: optixTrace(
// CUDA: __noinline__ void [[TRACE:traceSecondary_[0-9]+]](uint * [[SECONDARY:payload_[0-9]+]])
// CUDA: optixTrace({{.*}}, [[SECONDARY]]);
// CUDA: RecursiveClosestHit_invoke_
// CUDA: [[TRACE]](&secondaryPayload_
// CUDA: __closesthit__RecursiveClosestHit_closesthit()
// CUDA: optixGetAttribute_0()
// CUDA: optixSetPayload_0(
// CUDA: __miss__SecondaryMiss_miss()
// CUDA: optixSetPayload_0(
// PTX: .visible .entry __raygen__RayGeneration()
// PTX: _optix_trace_typed_
// PTX: .func {{.*}}traceSecondary_
// PTX: _optix_trace_typed_
// PTX: .visible .entry __closesthit__RecursiveClosestHit_closesthit()
// PTX: _optix_get_ray_tmax
// PTX: _optix_get_launch_dimension_x
// PTX: _optix_set_payload
// PTX: .visible .entry __miss__SecondaryMiss_miss()
// PTX: _optix_get_launch_index_x
// PTX: _optix_set_payload
