//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry Intersection -stage intersection -target hlsl -validate-ir
//TEST:SIMPLE(filecheck=PROVIDER): -experimental-feature -entry Intersection -stage intersection -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=FILTER): -experimental-feature -entry FilteredAnyHit -stage anyhit -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

struct Attributes
{
    uint candidate;
}

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = void;
    typealias Primitive = rt::BoundingBoxPrimitive<Attributes>;
}

RWStructuredBuffer<uint> observed;

// A retained value-returning helper must exit its entire AnyHit invocation, not return a fake
// value that the caller could consume. Payload writes before IgnoreHit still take effect.
[noinline]
uint filter(in rt::AnyHitInput<HitContext> input)
{
    input.payload += 3;
    if (input.attributes.candidate == 1)
        input.ignoreHit();
    if (input.attributes.candidate == 2)
        input.acceptHitAndEndSearch();
    return input.hitKind;
}

struct FilteredAnyHit : rt::IAnyHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::AnyHitInput<Context> input)
    {
        input.payload += filter(input);
        observed[2] += 1;
    }
}

struct AcceptingAnyHit : rt::IAnyHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::AnyHitInput<Context> input)
    {
        input.payload += input.attributes.candidate;
    }
}

[noinline]
float readCurrentLimit(in rt::IntersectionInput<HitContext> input)
{
    return input.objectSpaceRay.TMax;
}

struct Intersection : rt::IIntersectionShader
{
    typealias Context = HitContext;
    static void invoke(in rt::IntersectionInput<Context> input)
    {
        float before = readCurrentLimit(input);
        bool first = input.reportHit(5, 7, Attributes(0));
        float after = readCurrentLimit(input);
        observed[0] = asuint(before - after);
        // The middle report is rejected in one group. Its attributes must not replace the first
        // accepted candidate. The third report terminates that group's Intersection invocation.
        if (first)
            input.reportHit(3, 8, Attributes(1));
        input.reportHit(2, 9, Attributes(2));
        observed[1] += 1;
    }
}

struct ClosestHit : rt::IClosestHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload += input.attributes.candidate + input.hitKind;
    }
}

struct FilteredGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = ::ClosestHit;
    typealias AnyHit = FilteredAnyHit;
    typealias Intersection = ::Intersection;
}

struct AcceptingGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = ::ClosestHit;
    typealias AnyHit = AcceptingAnyHit;
    typealias Intersection = ::Intersection;
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::HitGroupList<FilteredGroup, AcceptingGroup>;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

rt::AccelerationStructure scene;
rt::TraceProgramDescriptor<Schema> program;
uniform uint rayFlags;

[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.rayFlags = rayFlags;
    ray.instanceMask = 255;
    ray.sbtStride = 1;
    uint payload = 0;
    rt::RayTracer<Schema> tracer;
    tracer.trace(ray, scene, program, payload);
    observed[3] = payload;
}

// METAL-DAG: accept_intersection
// METAL-DAG: continue_search
// METAL-DAG: intersection(bounding_box,
// METAL-DAG: min_distance
// METAL-DAG: max_distance
// METAL-DAG: opaque
// METAL-DAG: buffer(
// METAL-DAG: anyHitFunctions
// METAL-DAG: intersectionShaders
// METAL-NOT: readCurrentLimit
// METAL-NOT: filter_
// METAL-NOT: metalRTReportHit
// METAL-NOT: metalRTIgnoreHit
// PROVIDER-DAG: visible]]
// PROVIDER-DAG: readCurrentLimit
// PROVIDER-DAG: portableRTReportHit
// PROVIDER-NOT: metalRTReportHit
// FILTER-DAG: visible]]
// FILTER-DAG: filter
// FILTER-NOT: metalRTIgnoreHit
// HLSL: ReportHit(
