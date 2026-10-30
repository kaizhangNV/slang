//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4 -validate-ir

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

struct Callable : rt::ICallableShader
{
    typealias Context = CallableContext;
    static void invoke(in rt::CallableInput<Context> input)
    {
        input.data += input.dispatchRaysIndex.x + input.dispatchRaysDimensions.x;
    }
}

struct CallableSchema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::EmptyHitGroupList;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::CallableShaderList<Callable>;
}

rt::TraceProgramDescriptor<CallableSchema> callableProgram;

[noinline]
void callHelper(inout uint value)
{
    rt::RayTracer<CallableSchema> tracer;
    tracer.callShader<CallableContext>(0, callableProgram, value);
}

struct MissContext : rt::IPayloadContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = void;
}

struct Miss : rt::IMissShader
{
    typealias Context = MissContext;
    static void invoke(in rt::MissInput<Context> input)
    {
        callHelper(input.payload);
    }
}

struct InnerSchema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::EmptyHitGroupList;
    typealias MissShaders = rt::MissShaderList<Miss>;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

rt::TraceProgramDescriptor<InnerSchema> innerProgram;
rt::AccelerationStructure scene;

[noinline]
void traceHelper(inout uint value)
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 1, 0);
    ray.ray.TMax = 10;
    ray.instanceMask = 255;
    rt::RayTracer<InnerSchema> tracer;
    tracer.trace(ray, scene, innerProgram, value);
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
    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        traceHelper(input.payload);
    }
}

struct HitGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = ::ClosestHit;
    typealias AnyHit = rt::NoAnyHit<Context>;
    typealias Intersection = rt::NoIntersection<Context>;
}

struct OuterSchema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::HitGroupList<HitGroup>;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

rt::TraceProgramDescriptor<OuterSchema> outerProgram;
RWStructuredBuffer<uint> output;

// Only the innermost callable reads launch builtins. Requirements must cross the actual
// outer-trace -> ClosestHit -> inner-trace -> Miss -> callable chain before any ABI is frozen.
[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.instanceMask = 255;
    ray.sbtStride = 1;
    uint payload = 1;
    rt::RayTracer<OuterSchema> tracer;
    tracer.trace(ray, scene, outerProgram, payload);
    output[0] = payload;
}

// METAL-DAG: uint3 dispatchIndex
// METAL-DAG: uint3 dispatchDimensions
// METAL-DAG: thread_position_in_grid
// METAL-DAG: threads_per_grid
// METAL-DAG: traceHelper
// METAL-DAG: callHelper
// METAL-DAG: _slang_callable_function_index
// METAL-NOT: metalRTGetDispatchRays
