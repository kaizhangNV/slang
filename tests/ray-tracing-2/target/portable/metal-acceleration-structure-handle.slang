//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -validate-ir -DSTORAGE=0
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -validate-ir -DSTORAGE=1
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -validate-ir -DSTORAGE=2
//TEST:SIMPLE(filecheck=MOTION): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -validate-ir -DSTORAGE=0 -DMOTION=1
//TEST:SIMPLE(filecheck=MOTION): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -validate-ir -DSTORAGE=1 -DMOTION=1
//TEST:SIMPLE(filecheck=MOTION): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -validate-ir -DSTORAGE=2 -DMOTION=1

import slang.raytracing.portable;

#ifndef MOTION
#define MOTION 0
#endif

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
#if MOTION
    typealias Motion = rt::PrimitiveMotion;
#else
    typealias Motion = rt::NoMotion;
#endif
}

struct Record
{
    uint value;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = ::Record;
    typealias Primitive = rt::TrianglePrimitive;
}

struct ClosestHit : rt::IClosestHitShader
{
    typealias Context = HitContext;
    static void invoke(rt::ClosestHitInput<Context> input)
    {
        input.payload = input.record.value;
    }
}

struct HitGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = ::ClosestHit;
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

struct Parameters
{
    rt::AccelerationStructure.Handle scenes[2];
}

#if STORAGE == 0
uniform rt::AccelerationStructure.Handle scene;
#elif STORAGE == 1
ParameterBlock<Parameters> parameters;
#else
StructuredBuffer<Parameters> parameters;
#endif
uniform uint selectedRow;
uniform uint selectedScene;
rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<uint> output;

// A stored handle identifies the argument buffer containing both the native AS and its instance
// contributions. Loading it must not recover only the native AS. The dynamic indices also prevent
// the compiler from replacing the stored-handle cases with one statically selected declaration.
[shader("raygeneration")]
void RayGeneration()
{
#if STORAGE == 0
    rt::AccelerationStructure selected = scene;
#elif STORAGE == 1
    rt::AccelerationStructure selected = parameters.scenes[selectedScene & 1];
#else
    rt::AccelerationStructure selected = parameters[selectedRow].scenes[selectedScene & 1];
#endif
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.instanceMask = 255;
    ray.sbtStride = 1;
    ray.time = 0.5;
    uint payload = 0;
    rt::RayTracer<Schema> tracer;
    tracer.trace(ray, selected, program, payload);
    output[0] = payload;
}

// METAL-DAG: struct PortableRTAccelerationStructureResources
// METAL-DAG: acceleration_structure<metal::raytracing::instancing>
// METAL-DAG: uint device* instanceContributionMetadata
// METAL-DAG: _slang_result.instance_id
// METAL-DAG: closestHitFunctions

// The handle's stored resource type and the traced resource must acquire the same motion type.
// MOTION-DAG: struct PortableRTAccelerationStructureResources
// MOTION-DAG: acceleration_structure<metal::raytracing::instancing, metal::raytracing::primitive_motion>
// MOTION-DAG: uint device* instanceContributionMetadata
// MOTION-DAG: _slang_result.instance_id
// MOTION-DAG: closestHitFunctions
