//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=RADIANCE): -experimental-feature -entry RadianceClosestHit -stage closesthit -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=SHADOW): -experimental-feature -entry ShadowClosestHit -stage closesthit -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

struct RadiancePayload
{
    float3 color;
    uint count;
}

struct ShadowPayload
{
    uint visible;
}

struct RadianceAttributes
{
    float3 normal;
}

struct ShadowAttributes
{
    uint flags;
}

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct HitContext<PayloadType, AttributeType> : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = PayloadType;
    typealias Record = void;
    typealias Primitive = rt::BoundingBoxPrimitive<AttributeType>;
}

typealias RadianceContext = HitContext<RadiancePayload, RadianceAttributes>;
typealias ShadowContext = HitContext<ShadowPayload, ShadowAttributes>;

struct RadianceIntersection : rt::IIntersectionShader
{
    typealias Context = RadianceContext;
    static void invoke(in rt::IntersectionInput<Context> input)
    {
        input.reportHit(2.0, 7, RadianceAttributes(float3(0, 0, 1)));
    }
}

struct ShadowIntersection : rt::IIntersectionShader
{
    typealias Context = ShadowContext;
    static void invoke(in rt::IntersectionInput<Context> input)
    {
        input.reportHit(3.0, 9, ShadowAttributes(123));
    }
}

struct RadianceClosestHit : rt::IClosestHitShader
{
    typealias Context = RadianceContext;
    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload.color += input.attributes.normal;
        input.payload.count += input.hitKind;
    }
}

struct ShadowClosestHit : rt::IClosestHitShader
{
    typealias Context = ShadowContext;
    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload.visible = input.attributes.flags;
    }
}

struct RadianceGroup : rt::IHitGroup
{
    typealias Context = RadianceContext;
    typealias ClosestHit = RadianceClosestHit;
    typealias AnyHit = rt::NoAnyHit<Context>;
    typealias Intersection = RadianceIntersection;
}

struct ShadowGroup : rt::IHitGroup
{
    typealias Context = ShadowContext;
    typealias ClosestHit = ShadowClosestHit;
    typealias AnyHit = rt::NoAnyHit<Context>;
    typealias Intersection = ShadowIntersection;
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::HitGroupList<RadianceGroup, ShadowGroup>;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

rt::AccelerationStructure scene;
rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<float4> output;

[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.instanceMask = 255;
    ray.sbtStride = 2;
    rt::RayTracer<Schema> tracer;

    // Both calls use one descriptor and one physical SBT. The host records select the matching
    // groups; payload and procedural attribute storage must remain distinct for the two calls.
    RadiancePayload radiance = {};
    tracer.trace(ray, scene, program, radiance);
    ray.sbtOffset = 1;
    ShadowPayload shadow = {};
    tracer.trace(ray, scene, program, shadow);
    output[0] = float4(radiance.color, radiance.count);
    output[1] = float4(shadow.visible);
}

// METAL-NOT: visible]]
// METAL-DAG: RadiancePayload{{.*}} payload_
// METAL-DAG: ShadowPayload{{.*}} payload_
// METAL-DAG: RadianceAttributes{{.*}} customAttributes_
// METAL-DAG: ShadowAttributes{{.*}} customAttributes_
// METAL-DAG: intersection(bounding_box,
// METAL-NOT: visible]]
// METAL-NOT: metalTraceRay

// RADIANCE: ->color{{.*}} = {{.*}}->color{{.*}} + {{.*}}->normal
// RADIANCE: ->count{{.*}}rt_ClosestHitInput_hitKind_get
// RADIANCE: visible]] void RadianceClosestHit_closesthit(void thread* payload{{.*}}, void thread* attributes
// RADIANCE: (RadiancePayload{{.*}} thread*){{.*}}payload
// RADIANCE: (RadianceAttributes{{.*}} thread*){{.*}}attributes
// RADIANCE: RadianceClosestHit_invoke{{.*}}hitKind
// RADIANCE-NOT: PortableRTRayData

// SHADOW: ->visible{{.*}} = {{.*}}->flags
// SHADOW: visible]] void ShadowClosestHit_closesthit(void thread* payload{{.*}}, void thread* attributes
// SHADOW: (ShadowPayload{{.*}} thread*){{.*}}payload
// SHADOW: (ShadowAttributes{{.*}} thread*){{.*}}attributes
// SHADOW-NOT: PortableRTRayData
