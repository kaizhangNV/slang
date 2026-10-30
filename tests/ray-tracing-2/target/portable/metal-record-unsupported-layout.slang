//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}
struct Material { uint value; }
struct MissContext : rt::IPayloadContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    // A resource container is a valid associated type, but it does not describe fixed-size
    // application bytes in Metal's runtime SBT. Diagnose that at the target layout boundary.
    typealias Record = ConstantBuffer<Material>;
}
struct Miss : rt::IMissShader
{
    typealias Context = MissContext;
    static void invoke(rt::MissInput<Context> input) { input.payload = 1; }
}
struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::EmptyHitGroupList;
    typealias MissShaders = rt::MissShaderList<Miss>;
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
    uint payload = 0;
    rt::RayTracer<Schema> tracer;
    tracer.trace(ray, scene, program, payload);
    output[0] = payload;
}

// CHECK: Metal ray-tracing Record requires a fixed-size inline storage layout supported by the descriptor ABI
