//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4 -D VIA_INTERFACE=1
//TEST:SIMPLE(filecheck=HIT): -experimental-feature -entry ClosestHit -stage closesthit -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=MISS): -experimental-feature -entry Miss -stage miss -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = float4;
    typealias Record = void;
    typealias Primitive = rt::TrianglePrimitive;
}

struct ClosestHit : rt::IClosestHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        // The aggregate attribute view has no stored barycentrics. Its property reads the
        // current invocation, including through a noinline ordinary helper and an input copy.
        update(input);
    }

    [noinline]
    static void update(in rt::ClosestHitInput<Context> input)
    {
        let copy = input;
        input.payload = float4(copy.attributes.barycentricCoord, copy.distance, 1.0);
    }
}

struct Miss : rt::IMissShader
{
    typealias Context = HitContext;
    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload = float4(-1.0);
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
    typealias MissShaders = rt::MissShaderList<Miss>;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

rt::AccelerationStructure scene;
rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<float4> output;
uniform uint rayFlags;

interface ITracer { float4 trace(); }

struct Tracer : ITracer
{
    float4 trace()
    {
        rt::RayTraversalDesc desc = {};
        desc.ray.Origin = float3(0.25, 0.25, 0.0);
        desc.ray.Direction = float3(0.0, 0.0, 1.0);
        desc.ray.TMax = 10.0;
        desc.rayFlags = rayFlags;
        desc.instanceMask = 255;
        desc.sbtStride = 1;
        float4 payload = float4(17.0);
        rt::RayTracer<Schema> tracer;
        tracer.trace(desc, scene, program, payload);
        return payload;
    }
}

float4 viaInterface<T>(T tracer) where T : ITracer
{
    return tracer.trace();
}

[shader("raygeneration")]
void RayGeneration()
{
#if defined(VIA_INTERFACE)
    output[0] = viaInterface(Tracer());
#else
    output[0] = Tracer().trace();
#endif
}

// METAL-NOT: visible]]
// METAL-DAG: metal::visible_function_table<
// METAL-DAG: metal::raytracing::intersector<metal::raytracing::instancing, metal::raytracing::triangle_data, metal::raytracing::world_space_data>
// METAL-DAG: _slang_intersector.force_opacity(
// METAL-DAG: _slang_result.triangle_barycentric_coord
// METAL-DAG: kernel]]
// METAL-NOT: visible]]
// METAL-NOT: intersection(

// HIT: void ClosestHit_update{{.*}}(float4 thread*
// HIT: float4({{.*}}rt_ClosestHitInput_distance_get
// HIT: visible]] void ClosestHit_closesthit(void thread* payload
// HIT-SAME: void thread* attributes
// HIT-SAME: uchar thread* globalResources
// HIT: ClosestHit_invoke{{.*}}((float4 thread*){{.*}}payload{{.*}}, barycentricCoord{{.*}}, distance
// HIT-NOT: visible]]

// MISS: float4(-1.0f)
// MISS: visible]] void Miss_miss(void thread* payload
// MISS-SAME: uchar thread* globalResources
// MISS: Miss_invoke{{.*}}((float4 thread*){{.*}}payload
// MISS-NOT: visible]]
