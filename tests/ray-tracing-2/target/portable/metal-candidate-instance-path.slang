//TEST:SIMPLE(filecheck=PRIMITIVE): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -capability metallib_3_1 -D LEVELS=1 -validate-ir
//TEST:SIMPLE(filecheck=PROVIDER): -experimental-feature -entry AnyHit -stage anyhit -target metal -capability metallib_3_1 -D LEVELS=3 -validate-ir
//TEST:SIMPLE(filecheck=MULTI): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -capability metallib_3_1 -D LEVELS=3 -validate-ir

import slang.raytracing.portable;

#ifndef LEVELS
#define LEVELS 3
#endif

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::MultiLevelAccelerationStructure<LEVELS>;
    typealias Motion = rt::NoMotion;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = void;
    typealias Primitive = rt::TrianglePrimitive;
}

[noinline]
uint readPath(in rt::AnyHitInput<HitContext> input)
{
    uint value = input.instanceCount;
    for (uint level = 0; level < input.instanceCount; ++level)
        value += input.getInstanceIndex(level) + input.getInstanceID(level);
    // Without instancing these are identity matrices, not nonexistent native builtin parameters.
    return value + uint(input.objectToWorld[0][0] + input.worldToObject[2][2]);
}

struct AnyHit : rt::IAnyHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::AnyHitInput<Context> input)
    {
        input.payload += readPath(input);
    }
}

struct HitGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = rt::NoClosestHit<Context>;
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

TraceContext.AccelerationStructure scene;
rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<uint> output;

[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.instanceMask = 255;
    ray.sbtStride = 1;
    uint payload = 0;
    rt::RayTracer<Schema> tracer;
    tracer.trace(ray, scene, program, payload);
    output[0] = payload;
}

// PRIMITIVE: intersection(triangle, metal::raytracing::triangle_data)
// PRIMITIVE-NOT: instance_id
// PRIMITIVE-NOT: user_instance_id
// PRIMITIVE-NOT: object_to_world_transform
// PRIMITIVE-NOT: world_to_object_transform
// MULTI-DAG: max_levels<3>
// MULTI-DAG: metal::array_ref<uint>
// MULTI-DAG: instance_id
// MULTI-DAG: user_instance_id
// MULTI-DAG: object_to_world_transform
// MULTI-DAG: world_to_object_transform

// PROVIDER-DAG: visible]]
// PROVIDER-DAG: readPath
// PROVIDER-DAG: instanceIndices
// PROVIDER-DAG: instanceIDs
