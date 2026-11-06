//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::PrimitiveMotion;
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::EmptyHitGroupList;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

struct Parameters
{
    rt::AccelerationStructure scenes[2];
    uint sceneIndex;
}

ParameterBlock<Parameters> parameters;
rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<uint> output;

// An element access gets its native type from the homogeneous array element type. The
// compiler cannot retag just the final extraction: the parameter field, local array, and
// helper ABI must all preserve the required primitive_motion resource type.
[noinline]
rt::AccelerationStructure selectScene(rt::AccelerationStructure scenes[2], uint index)
{
    return scenes[index];
}

[shader("raygeneration")]
void RayGeneration()
{
    rt::AccelerationStructure localScenes[2] = {
        parameters.scenes[0],
        parameters.scenes[1],
    };
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.time = 0.5;
    uint payload = 17;
    rt::RayTracer<Schema> tracer;
    tracer.trace(ray, selectScene(localScenes, parameters.sceneIndex & 1), program, payload);
    output[0] = payload;
}

// METAL-DAG: acceleration_structure<metal::raytracing::instancing, metal::raytracing::primitive_motion>
// METAL-DAG: selectScene
// METAL-NOT: acceleration_structure<metal::raytracing::instancing>
