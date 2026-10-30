//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -capability metallib_3_1 -validate-ir
//TEST:SIMPLE(filecheck=PROVIDER): -experimental-feature -entry AnyHit -stage anyhit -target metal -capability metallib_3_1 -validate-ir

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
    typealias Primitive = rt::CurvePrimitive;
}

// CurveData is a zero-storage view. The same non-inlined helper receives the native curve
// parameter from both the candidate AnyHit adapter and the committed ClosestHit adapter.
[noinline]
float readCurveParameter(rt::CurveData attributes)
{
    return attributes.parameter;
}

struct ClosestHit : rt::IClosestHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload = readCurveParameter(input.attributes);
    }
}

struct AnyHit : rt::IAnyHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::AnyHitInput<Context> input)
    {
        if (readCurveParameter(input.attributes) < 0.25)
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

// METAL-DAG: intersection(curve, metal::raytracing::instancing, metal::raytracing::curve_data{{[^)]*}})
// METAL-DAG: curve_parameter
// METAL-DAG: accept_intersection
// METAL-DAG: continue_search
// METAL-DAG: anyHitFunctions
// METAL-NOT: readCurveParameter
// PROVIDER-DAG: visible]]
// PROVIDER-DAG: readCurveParameter
// METAL-NOT: metalRTIgnoreHit
