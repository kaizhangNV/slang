//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

struct EmptyPayloadA {}
struct EmptyPayloadB {}

struct Attributes
{
    uint value;
}

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct HitContext<PayloadType> : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = PayloadType;
    typealias Record = void;
    typealias Primitive = rt::BoundingBoxPrimitive<Attributes>;
}

RWStructuredBuffer<uint> observed;

struct Intersection<PayloadType> : rt::IIntersectionShader
{
    typealias Context = HitContext<PayloadType>;

    static void invoke(in rt::IntersectionInput<Context> input)
    {
        input.reportHit(2.0, 7, Attributes(123));
    }
}

struct ClosestHit<PayloadType> : rt::IClosestHitShader
{
    typealias Context = HitContext<PayloadType>;

    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        // Empty payload values remain ordinary source values. Keeping the nominal read/write
        // legal must not create a different physical ray_data layout for each empty type.
        let payload = input.payload;
        input.payload = payload;
        observed[0] = input.attributes.value + input.hitKind;
    }
}

struct AnyHit<PayloadType> : rt::IAnyHitShader
{
    typealias Context = HitContext<PayloadType>;

    static void invoke(in rt::AnyHitInput<Context> input)
    {
        let payload = input.payload;
        input.payload = payload;
        if (input.attributes.value != 123)
            input.ignoreHit();
    }
}

struct Group<PayloadType> : rt::IHitGroup
{
    typealias Context = HitContext<PayloadType>;
    typealias ClosestHit = ::ClosestHit<PayloadType>;
    typealias AnyHit = ::AnyHit<PayloadType>;
    typealias Intersection = ::Intersection<PayloadType>;
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::HitGroupList<Group<void>, Group<EmptyPayloadA>, Group<EmptyPayloadB>>;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

rt::AccelerationStructure scene;
rt::TraceProgramDescriptor<Schema> program;

[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.instanceMask = 255;
    ray.sbtStride = 3;
    rt::RayTracer<Schema> tracer;

    // The payload-less overload supplies the library's own empty type. Every selected group
    // still needs the same custom-attribute storage as the explicitly empty payload traces.
    tracer.trace(ray, scene, program);
    ray.sbtOffset = 1;
    EmptyPayloadA payloadA = {};
    tracer.trace(ray, scene, program, payloadA);
    ray.sbtOffset = 2;
    EmptyPayloadB payloadB = {};
    tracer.trace(ray, scene, program, payloadB);
}

// Every call and callback must share one carrier, including its procedural attribute storage.
// METAL: struct PortableRTRayData_{{[0-9]+}}
// METAL: Attributes{{.*}} customAttributes
// METAL-NOT: struct PortableRTRayData_
