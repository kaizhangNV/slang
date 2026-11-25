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
rt::TraceProgramDescriptor<Schema, 1> program;
RWStructuredBuffer<float> output;

[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.instanceMask = 255;
    float payload = 0;
    rt::RayTracer<Schema> tracer;
    tracer.trace(ray, scene, program, payload);
    output[0] = payload;
}

// Traversal selects a host-bound table; selecting RayGeneration does not compile AnyHit.
// METAL-DAG: intersection_function_table<
// METAL-DAG: tables
// METAL-DAG: anyHitFunctions
// METAL-NOT: readCurveParameter
// METAL-NOT: {{\[\[intersection\(}}
// METAL-NOT: metalRTIgnoreHit
// The selected AnyHit is the native callback and directly reaches its ordinary helper.
// PROVIDER-DAG: intersection(curve,
// PROVIDER-DAG: readCurveParameter
// PROVIDER-DAG: accept_intersection
// PROVIDER-DAG: continue_search
// PROVIDER-DAG: curve_parameter
// PROVIDER-NOT: visible]]
// PROVIDER-NOT: metalRTIgnoreHit
