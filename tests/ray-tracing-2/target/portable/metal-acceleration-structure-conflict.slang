//TEST:SIMPLE(filecheck=INDEPENDENT): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=CONFLICT): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -DSHARED_SCENE=1

import slang.raytracing.portable;

struct TraceContext<M : rt::IRayMotion> : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = M;
}

struct Schema<M : rt::IRayMotion> : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext<M>;
    typealias HitGroups = rt::EmptyHitGroupList;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

rt::AccelerationStructure staticScene;
rt::AccelerationStructure movingScene;
rt::TraceProgramDescriptor<Schema<rt::NoMotion>> staticProgram;
rt::TraceProgramDescriptor<Schema<rt::PrimitiveMotion>> movingProgram;
RWStructuredBuffer<uint> output;

[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.instanceMask = 255;
    ray.sbtStride = 1;
    ray.time = 0.5;
    uint payload = 17;
    rt::RayTracer<Schema<rt::NoMotion>> staticTracer;
    staticTracer.trace(ray, staticScene, staticProgram, payload);
    rt::RayTracer<Schema<rt::PrimitiveMotion>> movingTracer;
#ifdef SHARED_SCENE
    // The conflict is on one resource value, not on the common source AS type. Separate
    // globals above may have different native signatures; one shared global cannot.
    movingTracer.trace(ray, staticScene, movingProgram, payload);
#else
    movingTracer.trace(ray, movingScene, movingProgram, payload);
#endif
    output[0] = payload;
}

// INDEPENDENT-DAG: acceleration_structure<metal::raytracing::instancing>
// INDEPENDENT-DAG: acceleration_structure<metal::raytracing::instancing, metal::raytracing::primitive_motion>
// CONFLICT: error[E38099]
