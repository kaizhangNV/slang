//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry RayGeneration -stage raygeneration -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry RayGeneration -stage raygeneration -target spirv-assembly -emit-spirv-directly

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
