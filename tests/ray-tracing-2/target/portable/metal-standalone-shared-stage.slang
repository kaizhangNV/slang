//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -stage raygeneration -entry SharedHit -stage closesthit -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}
struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = void;
    typealias Primitive = rt::TrianglePrimitive;
}

// This ordinary helper is reached both from the schema's visible adapter and from the explicitly
// selected standalone stage. Ordinary instancing uses a scalar ID in both ABIs, not an instance
// path in one and a scalar in the other. Both callers must update their own payload reference.
[noinline]
void addInstanceID(rt::ClosestHitInput<HitContext> input)
{
    let copy = input;
    copy.payload += copy.instanceID;
}
struct SharedHit : rt::IClosestHitShader
{
    typealias Context = HitContext;
    static void invoke(rt::ClosestHitInput<Context> input) { addInstanceID(input); }
}
struct HitGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = SharedHit;
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
rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<uint> output;
[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 100;
    ray.instanceMask = 255;
    ray.sbtStride = 1;
    uint payload = 1;
    rt::RayTracer<Schema> tracer;
    tracer.trace(ray, scene, program, payload);
    output[0] = payload;
}

// METAL-DAG: addInstanceID
// METAL-DAG: void SharedHit_closesthit(
// METAL-DAG: __slang_portable_rt_0_closesthit_0
// METAL-NOT: metalRTGetInstanceID
