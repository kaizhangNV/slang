//TEST:SIMPLE(filecheck=PLAIN): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=PRIMITIVE): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -DLEVELS=1 -validate-ir
//TEST:SIMPLE(filecheck=MULTI): -experimental-feature -entry RayGeneration -target metal -capability metallib_3_1 -DLEVELS=3 -validate-ir
//TEST:SIMPLE(filecheck=MOTION): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -DMOTION=1 -validate-ir
//TEST:SIMPLE(filecheck=EXTENDED): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -capability metal_raytracing_extended_limits -validate-ir
//TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry RayGeneration -target metal -capability metallib_3_1 -DLEVELS=0
//TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry RayGeneration -target metal -capability metallib_3_1 -DLEVELS=33
//TEST:SIMPLE(filecheck=VERSION): -experimental-feature -entry RayGeneration -target metal -capability metallib_3_0 -DLEVELS=3
//TEST:SIMPLE(filecheck=INSTANCE): -experimental-feature -entry RayGeneration -target metal -capability metallib_3_1 -DLEVELS=1 -DMOTION=2

import slang.raytracing.portable;

#ifndef MOTION
#define MOTION 0
#endif

struct TraceContext : rt::ITraceContext
{
#ifdef LEVELS
    typealias AccelerationStructure = rt::MultiLevelAccelerationStructure<LEVELS>;
#else
    typealias AccelerationStructure = rt::AccelerationStructure;
#endif
#if MOTION == 1
    typealias Motion = rt::PrimitiveMotion;
#elif MOTION == 2
    typealias Motion = rt::InstanceMotion;
#else
    typealias Motion = rt::NoMotion;
#endif
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::EmptyHitGroupList;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

struct Scene
{
    TraceContext.AccelerationStructure first;
    TraceContext.AccelerationStructure second;
}

struct Parameters
{
    Scene scene;
    bool selectFirst;
}

ParameterBlock<Parameters> parameters;
rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<uint> output;

// The trace operand is not the resource declaration: it crosses a nested ParameterBlock,
// a selected value, a noinline helper's parameter and return, and a local aggregate. All
// representations must acquire the same physical Metal motion type.
[noinline]
TraceContext.AccelerationStructure chooseScene(
    TraceContext.AccelerationStructure first,
    TraceContext.AccelerationStructure second,
    bool selectFirst)
{
    return selectFirst ? first : second;
}

[shader("raygeneration")]
void RayGeneration()
{
    Scene local = {
        chooseScene(parameters.scene.first, parameters.scene.second, parameters.selectFirst),
        parameters.scene.second,
    };
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.instanceMask = 255;
    ray.sbtStride = 1;
    ray.time = 0.5;
    uint payload = 17;
    rt::RayTracer<Schema> tracer;
    tracer.trace(ray, local.first, program, payload);
    output[0] = payload;
}

// PLAIN-DAG: acceleration_structure<metal::raytracing::instancing>
// PLAIN-DAG: intersector<metal::raytracing::instancing>
// PRIMITIVE-DAG: acceleration_structure<>
// PRIMITIVE-DAG: intersector<>
// MULTI-DAG: acceleration_structure<metal::raytracing::instancing>
// MULTI-DAG: intersector<metal::raytracing::instancing, metal::raytracing::max_levels<3>>
// MOTION-DAG: acceleration_structure<metal::raytracing::instancing, metal::raytracing::primitive_motion>
// MOTION-DAG: intersector<metal::raytracing::instancing, metal::raytracing::primitive_motion>
// EXTENDED-DAG: acceleration_structure<metal::raytracing::instancing>
// EXTENDED-DAG: intersector<metal::raytracing::instancing, metal::raytracing::extended_limits>
// INVALID: error[E38096]
// VERSION: error[E38097]
// INSTANCE: error[E38098]
