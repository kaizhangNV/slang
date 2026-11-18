//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry main -target hlsl -validate-ir
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry main -target spirv-assembly -emit-spirv-directly -validate-ir
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry main -target cuda -validate-ir
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry main -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct CallableContext : rt::ICallableContext
{
    typealias TraceContext = ::TraceContext;
    typealias CallableData = uint;
    typealias Record = void;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = void;
    typealias Primitive = rt::TrianglePrimitive;
}

struct ClosestHit : rt::IClosestHitShader
{
    typealias Context = HitContext;
    static void invoke(rt::ClosestHitInput<Context> input) { input.payload = 1; }
}

struct HitGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = ::ClosestHit;
    typealias AnyHit = rt::NoAnyHit<Context>;
    typealias Intersection = rt::NoIntersection<Context>;
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::HitGroupList<HitGroup>;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

rt::AccelerationStructure scene;
rt::TraceProgramDescriptor<Schema, 2> twoRayTypes;
rt::TraceProgramDescriptor<Schema, 3> threeRayTypes;
rt::TraceProgramDescriptor<Schema, 2, rt::HitRecordLayout.SharedAcrossGeometries> sharedRecords;
RWStructuredBuffer<uint> output;
uniform uint rayType;

// A generic helper must retain count and layout through specialization. Explicit payload and
// callable context arguments still occupy their original first generic-argument positions.
void traceProgram<let N : int, let Layout : rt::HitRecordLayout>(
    rt::TraceProgramDescriptor<Schema, N, Layout> program, inout uint payload)
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.instanceMask = 255;
    ray.rayType = rayType;
    rt::RayTracer<Schema> tracer;
    tracer.trace<uint>(ray, scene, program, payload);
    tracer.callShader<CallableContext>(0, program, payload);
}

[shader("raygeneration")]
void main()
{
    uint payload = 0;
    traceProgram(twoRayTypes, payload);
    traceProgram(threeRayTypes, payload);
    traceProgram(sharedRecords, payload);
    output[0] = payload;
}

// The same runtime ray selector is retained while each descriptor supplies its own multiplier.
// HLSL-DAG: TraceRay({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, 2U,
// HLSL-DAG: TraceRay({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, 3U,
// HLSL-DAG: TraceRay({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, 0U,
// HLSL: CallShader(
// SPIRV-DAG: %[[TWO:[^ ]+]] = OpConstant %uint 2{{$}}
// SPIRV-DAG: %[[THREE:[^ ]+]] = OpConstant %uint 3{{$}}
// SPIRV-DAG: %[[ZERO:[^ ]+]] = OpConstant %uint 0{{$}}
// SPIRV-DAG: OpTraceRayKHR {{%[^ ]+ %[^ ]+ %[^ ]+ %[^ ]+}} %[[TWO]]
// SPIRV-DAG: OpTraceRayKHR {{%[^ ]+ %[^ ]+ %[^ ]+ %[^ ]+}} %[[THREE]]
// SPIRV-DAG: OpTraceRayKHR {{%[^ ]+ %[^ ]+ %[^ ]+ %[^ ]+}} %[[ZERO]]
// CUDA-DAG: optixTrace({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, 2U,
// CUDA-DAG: optixTrace({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, 3U,
// CUDA-DAG: optixTrace({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, 0U,
// CUDA: optixDirectCall<void>(
// METAL-DAG: kernel]] void main
// METAL-DAG: _slang_result.geometry_id * 2U +
// METAL-DAG: _slang_result.geometry_id * 3U +
// METAL-DAG: _slang_result.geometry_id * 0U +
