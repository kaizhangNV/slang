//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry RayGeneration -stage raygeneration -entry ProceduralClosestHit -stage closesthit -entry FilterCandidate -stage anyhit -entry TwoCandidates -stage intersection -entry BackgroundMiss -stage miss -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry RayGeneration -stage raygeneration -entry ProceduralClosestHit -stage closesthit -entry FilterCandidate -stage anyhit -entry TwoCandidates -stage intersection -entry BackgroundMiss -stage miss -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry RayGeneration -stage raygeneration -entry ProceduralClosestHit -stage closesthit -entry FilterCandidate -stage anyhit -entry TwoCandidates -stage intersection -entry BackgroundMiss -stage miss -target cuda
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -DTEST_NOINLINE_FILTER=1 -entry RayGeneration -stage raygeneration -entry ProceduralClosestHit -stage closesthit -entry FilterCandidate -stage anyhit -entry TwoCandidates -stage intersection -entry BackgroundMiss -stage miss -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -DTEST_NOINLINE_FILTER=1 -entry RayGeneration -stage raygeneration -entry ProceduralClosestHit -stage closesthit -entry FilterCandidate -stage anyhit -entry TwoCandidates -stage intersection -entry BackgroundMiss -stage miss -target cuda
//TEST:SIMPLE(filecheck=DXIL): -experimental-feature -entry RayGeneration -stage raygeneration -entry ProceduralClosestHit -stage closesthit -entry FilterCandidate -stage anyhit -entry TwoCandidates -stage intersection -entry BackgroundMiss -stage miss -target dxil -profile sm_6_6
//TEST:SIMPLE(filecheck=PTX): -experimental-feature -DTEST_NOINLINE_FILTER=1 -entry RayGeneration -stage raygeneration -entry ProceduralClosestHit -stage closesthit -entry FilterCandidate -stage anyhit -entry TwoCandidates -stage intersection -entry BackgroundMiss -stage miss -target ptx

import slang.raytracing.portable;

#ifndef TEST_NOINLINE_FILTER
#define TEST_NOINLINE_FILTER 0
#endif

struct Attributes
{
    uint candidate;
    uint value;
}

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint4;
    typealias Primitive = rt::BoundingBoxPrimitive<Attributes>;
    typealias Record = void;
}

struct MissContext : rt::IPayloadContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint4;
    typealias Record = void;
}

RWStructuredBuffer<uint4> resultBuffer;

// The first candidate is rejected, so traversal must continue to the farther candidate. The
// payload bit mask also verifies that AnyHit changes remain visible after IgnoreHit returns
// control to the intersection stage; it is not a discarded copy of the caller's payload.
// DXC rejects IgnoreHit in a retained ordinary helper even for legacy shaders (#14070):
// https://github.com/shader-slang/slang/issues/14070. The ordinary D3D runtime path inlines this
// helper; SPIR-V and OptiX additionally exercise the explicitly retained control-flow boundary.
#if TEST_NOINLINE_FILTER
[noinline]
#endif
void filter(in rt::AnyHitInput<HitContext> input)
{
    input.payload.z |= input.attributes.candidate;
    if (input.attributes.candidate == 1)
        input.ignoreHit();
}

struct FilterCandidate : rt::IAnyHitShader
{
    typealias Context = HitContext;

    static void invoke(in rt::AnyHitInput<Context> input)
    {
        let local = input;
        filter(local);
    }
}

struct TwoCandidates : rt::IIntersectionShader
{
    typealias Context = HitContext;

    static void invoke(in rt::IntersectionInput<Context> input)
    {
        Attributes nearAttributes = {1, 13};
        Attributes farAttributes = {2, 29};
        bool nearAccepted = input.reportHit(1.0, 7, nearAttributes);
        bool farAccepted = input.reportHit(3.0, 9, farAttributes);
        // Repeated implementation-level intersection visits may overwrite this same result,
        // but each visit must reject the near candidate and accept the far candidate.
        resultBuffer[2] = uint4(nearAccepted, farAccepted, 123, 0);
    }
}

struct ProceduralClosestHit : rt::IClosestHitShader
{
    typealias Context = HitContext;

    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload.x = uint(input.distance);
        input.payload.y = input.attributes.value;
        input.payload.w = input.hitKind;
    }
}

struct BackgroundMiss : rt::IMissShader
{
    typealias Context = MissContext;

    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload = uint4(0, 0, 0, 99);
    }
}

struct ProceduralGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = ProceduralClosestHit;
    typealias AnyHit = FilterCandidate;
    typealias Intersection = TwoCandidates;
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::HitGroupList<ProceduralGroup>;
    typealias MissShaders = rt::MissShaderList<BackgroundMiss>;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

rt::AccelerationStructure sceneBVH;
rt::TraceProgramDescriptor<Schema> program;

[shader("raygeneration")]
void RayGeneration()
{
    uint index = DispatchRaysIndex().x;
    rt::RayTraversalDesc desc = {};
    desc.ray.Origin = index == 0 ? float3(0.0) : float3(2.0, 2.0, 0.0);
    desc.ray.Direction = float3(0.0, 0.0, 1.0);
    desc.ray.TMin = 0.0;
    desc.ray.TMax = 10.0;
    desc.instanceMask = 0xff;
    desc.sbtStride = 1;

    uint4 payload = uint4(0);
    rt::RayTracer<Schema> tracer;
    tracer.trace(desc, sceneBVH, program, payload);
    resultBuffer[index] = payload;
}

// HLSL-DAG: TraceRay(
// HLSL-DAG: ReportHit(
// HLSL-DAG: IgnoreHit()
// HLSL-DAG: [shader("intersection")]void TwoCandidates_intersection()
// HLSL-DAG: [shader("anyhit")]void FilterCandidate_anyhit(
// SPIRV-DAG: OpEntryPoint IntersectionKHR
// SPIRV-DAG: OpEntryPoint AnyHitKHR
// SPIRV-DAG: OpReportIntersectionKHR
// SPIRV-DAG: OpIgnoreIntersectionKHR
// CUDA-DAG: __intersection__TwoCandidates_intersection
// CUDA-DAG: __anyhit__FilterCandidate_anyhit
// CUDA-DAG: optixReportIntersection(
// CUDA-DAG: optixIgnoreIntersection()
// DXIL-DAG: dx.op.traceRay
// DXIL-DAG: dx.op.reportHit
// DXIL-DAG: dx.op.ignoreHit
// PTX-DAG: __intersection__TwoCandidates_intersection
// PTX-DAG: __anyhit__FilterCandidate_anyhit
// PTX-DAG: _optix_report_intersection
// PTX-DAG: _optix_ignore_intersection
