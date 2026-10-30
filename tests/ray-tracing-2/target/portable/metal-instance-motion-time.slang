//TEST:SIMPLE(filecheck=INSTANCE): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=COMBINED): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4 -DCOMBINED_MOTION=1 -validate-ir

import slang.raytracing.portable;

#ifndef COMBINED_MOTION
#define COMBINED_MOTION 0
#endif

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
#if COMBINED_MOTION
    typealias Motion = rt::PrimitiveAndInstanceMotion;
#else
    typealias Motion = rt::InstanceMotion;
#endif
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = float;
    typealias Record = void;
    typealias Primitive = rt::TrianglePrimitive;
}

struct ClosestHit : rt::IClosestHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload += input.time;
    }
}

struct AnyHit : rt::IAnyHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::AnyHitInput<Context> input)
    {
        if (input.time < 0.25)
            input.ignoreHit();
        input.payload += input.time;
    }
}

struct Miss : rt::IMissShader
{
    typealias Context = HitContext;
    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload = -input.time;
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
    typealias MissShaders = rt::MissShaderList<Miss>;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

struct Parameters
{
    rt::AccelerationStructure scene;
    float time;
}

ParameterBlock<Parameters> parameters;
rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<float> output;

// The selected motion signature must reach the actual bound resource and this retained helper,
// not just the final trace operand. Time is observed by candidate, hit, and miss dispatch.
[noinline]
rt::AccelerationStructure forwardScene(rt::AccelerationStructure scene)
{
    return scene;
}

[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.instanceMask = 255;
    ray.sbtStride = 1;
    ray.time = parameters.time;
    float payload = 0;
    rt::RayTracer<Schema> tracer;
    tracer.trace(ray, forwardScene(parameters.scene), program, payload);
    output[0] = payload;
}

// INSTANCE-DAG: acceleration_structure<metal::raytracing::instancing, metal::raytracing::instance_motion>
// INSTANCE-DAG: intersector<metal::raytracing::instancing, metal::raytracing::triangle_data, metal::raytracing::world_space_data, metal::raytracing::instance_motion>
// INSTANCE-DAG: intersection_function_table<metal::raytracing::instancing, metal::raytracing::triangle_data, metal::raytracing::world_space_data, metal::raytracing::instance_motion>
// INSTANCE-DAG: float time_
// INSTANCE-DAG: intersection(triangle, metal::raytracing::instancing, metal::raytracing::triangle_data, metal::raytracing::world_space_data, metal::raytracing::instance_motion)
// INSTANCE-DAG: instanceMask_{{[0-9]+}}, {{.*}}time_{{[0-9]+}}, {{.*}}intersectionFunctions_
// INSTANCE-NOT: metalRTGetRayTime
// COMBINED-DAG: acceleration_structure<metal::raytracing::instancing, metal::raytracing::primitive_motion, metal::raytracing::instance_motion>
// COMBINED-DAG: intersector<metal::raytracing::instancing, metal::raytracing::triangle_data, metal::raytracing::world_space_data, metal::raytracing::primitive_motion, metal::raytracing::instance_motion>
// COMBINED-DAG: intersection_function_table<metal::raytracing::instancing, metal::raytracing::triangle_data, metal::raytracing::world_space_data, metal::raytracing::primitive_motion, metal::raytracing::instance_motion>
// COMBINED-DAG: float time_
// COMBINED-DAG: intersection(triangle, metal::raytracing::instancing, metal::raytracing::triangle_data, metal::raytracing::world_space_data, metal::raytracing::primitive_motion, metal::raytracing::instance_motion)
// COMBINED-DAG: instanceMask_{{[0-9]+}}, {{.*}}time_{{[0-9]+}}, {{.*}}intersectionFunctions_
// COMBINED-NOT: metalRTGetRayTime
