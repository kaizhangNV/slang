//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=PROVIDER): -experimental-feature -entry Miss -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
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
        input.payload += input.record.value;
    }
}

struct AnyHit : rt::IAnyHitShader
{
    typealias Context = HitContext;
    static void invoke(rt::AnyHitInput<Context> input)
    {
        if (input.record.value == 0)
            input.ignoreHit();
    }
}

struct Miss : rt::IMissShader
{
    typealias Context = HitContext;
    static void invoke(rt::MissInput<Context> input)
    {
        input.payload = output[0];
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

rt::AccelerationStructure first;
rt::AccelerationStructure second;
RaytracingAccelerationStructure queryScene;
rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<uint> output;

// Both structures may report instance index zero, while the host supplies contributions 2 and
// 7 in their respective metadata buffers. Sharing program records must not make both traces
// select the same hit record. The native ray query below uses neither metadata nor a wrapper.
[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.instanceMask = 255;
    ray.sbtStride = 1;
    uint firstPayload = 0;
    uint secondPayload = 0;
    rt::RayTracer<Schema> tracer;
    tracer.trace(ray, first, program, firstPayload);
    tracer.trace(ray, second, program, secondPayload);

    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(queryScene, RAY_FLAG_NONE, 255, ray.ray);
    while (query.Proceed()) {}
    output[0] = firstPayload;
    output[1] = secondPayload;
    output[2] = query.CommittedInstanceIndex();
}

// METAL-DAG: uint device* instanceContributionMetadata
// METAL-DAG: first{{[^;]*}}instanceContributionMetadata
// METAL-DAG: second{{[^;]*}}instanceContributionMetadata
// METAL-DAG: intersection(triangle,
// METAL-DAG: get_committed_instance_id
// METAL-DAG: queryScene{{.*}}buffer(

// The separately selected provider reads output without tracing. Its shared resource context
// must retain the portable AS argument buffers and the direct native AS from this component.
// PROVIDER-DAG: instanceContributionMetadata
// PROVIDER-DAG: accelerationStructure
// PROVIDER-DAG: queryScene
// PROVIDER-DAG: visible]] void Miss_miss
