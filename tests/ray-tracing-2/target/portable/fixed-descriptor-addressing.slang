//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry main -target hlsl -validate-ir
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry main -target spirv-assembly -emit-spirv-directly -validate-ir
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry main -target cuda -validate-ir
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry main -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=METAL_LARGE): -experimental-feature -entry main -target metal -profile metallib_2_4 -validate-ir -DLARGE_COUNT
//TEST:SIMPLE(filecheck=HLSL_CALLABLE): -experimental-feature -entry main -target hlsl -validate-ir -DLARGE_COUNT -DCALLABLE_ONLY
//TEST:SIMPLE(filecheck=SPIRV_CALLABLE): -experimental-feature -entry main -target spirv-assembly -emit-spirv-directly -validate-ir -DLARGE_COUNT -DCALLABLE_ONLY
//TEST:SIMPLE(filecheck=CUDA_CALLABLE): -experimental-feature -entry main -target cuda -validate-ir -DLARGE_COUNT -DCALLABLE_ONLY
//TEST:SIMPLE(filecheck=METAL_CALLABLE): -experimental-feature -entry main -target metal -profile metallib_2_4 -validate-ir -DLARGE_COUNT -DCALLABLE_ONLY

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
#ifdef LARGE_COUNT
// Classic IFT uses these constants in shader arithmetic, without truncating them to four bits.
rt::TraceProgramDescriptor<Schema, 16> firstProgram;
rt::TraceProgramDescriptor<Schema, 32> secondProgram;
rt::TraceProgramDescriptor<Schema, 32, rt::HitRecordLayout.SharedAcrossGeometries> sharedRecords;
#else
rt::TraceProgramDescriptor<Schema, 2> firstProgram;
rt::TraceProgramDescriptor<Schema, 3> secondProgram;
rt::TraceProgramDescriptor<Schema, 2, rt::HitRecordLayout.SharedAcrossGeometries> sharedRecords;
#endif
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
#ifndef CALLABLE_ONLY
    tracer.trace<uint>(ray, scene, program, payload);
#endif
    // Callable indexing does not use the descriptor's geometry multiplier on any target.
    tracer.callShader<CallableContext>(0, program, payload);
}

[shader("raygeneration")]
void main()
{
    uint payload = 0;
    traceProgram(firstProgram, payload);
    traceProgram(secondProgram, payload);
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
// METAL_LARGE-DAG: kernel]] void main
// METAL_LARGE-DAG: _slang_result.geometry_id * 16U +
// METAL_LARGE-DAG: _slang_result.geometry_id * 32U +
// METAL_LARGE-DAG: _slang_result.geometry_id * 0U +
// HLSL_CALLABLE: CallShader(
// SPIRV_CALLABLE: OpExecuteCallableKHR
// CUDA_CALLABLE: optixDirectCall<void>(
// METAL_CALLABLE: [_slang_callable_function_index](
