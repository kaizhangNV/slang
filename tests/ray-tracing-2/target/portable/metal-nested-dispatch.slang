//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=HIT): -experimental-feature -entry ClosestHit -stage closesthit -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=MISS): -experimental-feature -entry Miss -stage miss -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=CALLABLE): -experimental-feature -entry Callable -stage callable -target metal -profile metallib_2_4 -validate-ir

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

rt::TraceProgramDescriptor<CallableSchema, 1> callableProgram;

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

rt::TraceProgramDescriptor<InnerSchema, 1> innerProgram;
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

rt::TraceProgramDescriptor<OuterSchema, 1> outerProgram;
RWStructuredBuffer<uint> output;

// Only the innermost callable reads launch builtins. Each independently compiled stage uses
// the fixed contract to forward those values through the outer trace, inner trace, and callable
// dispatch, including through ordinary helpers that have no direct builtin reads.
[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.instanceMask = 255;
    uint payload = 1;
    rt::RayTracer<OuterSchema> tracer;
    tracer.trace(ray, scene, outerProgram, payload);
    output[0] = payload;
}

// METAL-NOT: visible]]
// METAL-DAG: thread_position_in_grid
// METAL-DAG: threads_per_grid
// METAL-DAG: _slang_invocation.dispatchRaysIndex_{{[0-9]+}} = dispatchIndex_{{[0-9]+}};
// METAL-DAG: _slang_invocation.dispatchRaysDimensions_{{[0-9]+}} = dispatchDimensions_{{[0-9]+}};
// METAL-DAG: [_slang_hit_function_index](&_slang_invocation);
// METAL-NOT: visible]]
// METAL-NOT: traceHelper
// METAL-NOT: callHelper

// Match the actual arguments across every call boundary. A nested dispatch must reuse the
// visible entry's context fields, not append another pair that its VFT caller omits.
// HIT: void traceHelper{{[_0-9]*}}(uint thread*{{[^,]*}}, uint3 [[HIT_HELPER_INDEX:[A-Za-z_][A-Za-z0-9_]*]], uint3 [[HIT_HELPER_DIMENSIONS:[A-Za-z_][A-Za-z0-9_]*]]
// HIT: _slang_invocation.dispatchRaysIndex_{{[0-9]+}} = [[HIT_HELPER_INDEX]];
// HIT: _slang_invocation.dispatchRaysDimensions_{{[0-9]+}} = [[HIT_HELPER_DIMENSIONS]];
// HIT: [_slang_miss_function_index](&_slang_invocation);
// HIT: void ClosestHit_invoke{{[_0-9]*}}(uint3 [[HIT_INVOKE_INDEX:[A-Za-z_][A-Za-z0-9_]*]], uint3 [[HIT_INVOKE_DIMENSIONS:[A-Za-z_][A-Za-z0-9_]*]], uint thread*
// HIT: traceHelper{{[_0-9]*}}({{.*}}, [[HIT_INVOKE_INDEX]], [[HIT_INVOKE_DIMENSIONS]],
// HIT: visible]] void ClosestHit_closesthit(ClosestHitInvocationContext_{{[0-9]+}} thread* [[CONTEXT:invocation_[0-9]+]])
// HIT: ClosestHit_invoke{{[_0-9]*}}([[CONTEXT]]->dispatchRaysIndex_{{[0-9]+}}, [[CONTEXT]]->dispatchRaysDimensions_{{[0-9]+}},
// HIT-NOT: metalRTGetDispatchRays

// MISS: void callHelper{{[_0-9]*}}(uint thread*{{[^,]*}}, uint3 [[MISS_HELPER_INDEX:[A-Za-z_][A-Za-z0-9_]*]], uint3 [[MISS_HELPER_DIMENSIONS:[A-Za-z_][A-Za-z0-9_]*]]
// MISS: _slang_invocation.dispatchRaysIndex_{{[0-9]+}} = [[MISS_HELPER_INDEX]];
// MISS: _slang_invocation.dispatchRaysDimensions_{{[0-9]+}} = [[MISS_HELPER_DIMENSIONS]];
// MISS: [_slang_callable_function_index](&_slang_invocation);
// MISS: void Miss_invoke{{[_0-9]*}}(uint3 [[MISS_INVOKE_INDEX:[A-Za-z_][A-Za-z0-9_]*]], uint3 [[MISS_INVOKE_DIMENSIONS:[A-Za-z_][A-Za-z0-9_]*]], uint thread*
// MISS: callHelper{{[_0-9]*}}({{.*}}, [[MISS_INVOKE_INDEX]], [[MISS_INVOKE_DIMENSIONS]],
// MISS: visible]] void Miss_miss(MissInvocationContext_{{[0-9]+}} thread* [[CONTEXT:invocation_[0-9]+]])
// MISS: Miss_invoke{{[_0-9]*}}([[CONTEXT]]->dispatchRaysIndex_{{[0-9]+}}, [[CONTEXT]]->dispatchRaysDimensions_{{[0-9]+}},
// MISS-NOT: metalRTGetDispatchRays

// CALLABLE: rt_CallableInput_dispatchRaysIndex_get{{.*}}rt_CallableInput_dispatchRaysDimensions_get
// CALLABLE: visible]] void Callable_callable(CallableInvocationContext_{{[0-9]+}} thread* [[CONTEXT:invocation_[0-9]+]])
// CALLABLE: Callable_invoke{{[_0-9]*}}([[CONTEXT]]->dispatchRaysIndex_{{[0-9]+}}, [[CONTEXT]]->dispatchRaysDimensions_{{[0-9]+}},
// CALLABLE-NOT: metalRTGetDispatchRays
