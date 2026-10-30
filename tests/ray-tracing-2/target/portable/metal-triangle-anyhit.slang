//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = float;
    typealias Record = void;
    typealias Primitive = rt::TrianglePrimitive;
}

// This helper is genuinely shared by the visible ClosestHit and the candidate AnyHit. Creating
// all invocation roots before threading builtins gives both callers the same ordinary signature.
[noinline]
float readBarycentric(rt::TriangleData attributes)
{
    return attributes.barycentricCoord.x;
}

struct ClosestHit : rt::IClosestHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload = readBarycentric(input.attributes);
    }
}

struct AnyHit : rt::IAnyHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::AnyHitInput<Context> input)
    {
        if (readBarycentric(input.attributes) < 0.25)
            input.ignoreHit();
        input.payload += 2;
    }
}

struct HitGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = ::ClosestHit;
    typealias AnyHit = ::AnyHit;
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
RWStructuredBuffer<float> output;

[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.instanceMask = 255;
    ray.sbtStride = 1;
    float payload = 0;
    rt::RayTracer<Schema> tracer;
    tracer.trace(ray, scene, program, payload);
    output[0] = payload;
}

// METAL-DAG: intersection(triangle,
// METAL-DAG: barycentric_coord
// METAL-DAG: accept_intersection
// METAL-DAG: continue_search
// METAL-DAG: visible]]
// METAL-DAG: readBarycentric
// METAL-NOT: metalRTIgnoreHit
