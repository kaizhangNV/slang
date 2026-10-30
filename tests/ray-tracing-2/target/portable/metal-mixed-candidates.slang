//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct Attributes { uint value; }

struct HitContext<P : rt::IIntersectionPrimitive> : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = void;
    typealias Primitive = P;
}

typealias TriangleContext = HitContext<rt::TrianglePrimitive>;
typealias BoxContext = HitContext<rt::BoundingBoxPrimitive<Attributes>>;

struct TriangleAnyHit : rt::IAnyHitShader
{
    typealias Context = TriangleContext;
    static void invoke(in rt::AnyHitInput<Context> input)
    {
        input.payload += 1;
    }
}

struct BoxAnyHit : rt::IAnyHitShader
{
    typealias Context = BoxContext;
    static void invoke(in rt::AnyHitInput<Context> input)
    {
        input.payload += input.hitKind;
    }
}

struct Intersection : rt::IIntersectionShader
{
    typealias Context = BoxContext;
    static void invoke(in rt::IntersectionInput<Context> input)
    {
        input.reportHit(1.0, 7, Attributes(3));
    }
}

struct TriangleGroup : rt::IHitGroup
{
    typealias Context = TriangleContext;
    typealias ClosestHit = rt::NoClosestHit<Context>;
    typealias AnyHit = TriangleAnyHit;
    typealias Intersection = rt::NoIntersection<Context>;
}

struct BoxGroup : rt::IHitGroup
{
    typealias Context = BoxContext;
    typealias ClosestHit = rt::NoClosestHit<Context>;
    typealias AnyHit = BoxAnyHit;
    typealias Intersection = ::Intersection;
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::HitGroupList<TriangleGroup, BoxGroup>;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

rt::AccelerationStructure scene;
rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<uint> output;

// Only the procedural source reads hitKind. A shared candidate requirement can nevertheless
// add front_facing to the triangle IFT signature, which must carry triangle_data consistently.
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

// METAL-DAG: intersection(triangle, metal::raytracing::instancing, metal::raytracing::triangle_data)
// METAL-DAG: intersection(bounding_box, metal::raytracing::instancing, metal::raytracing::triangle_data)
// METAL-NOT: metalRTGetHitKind
