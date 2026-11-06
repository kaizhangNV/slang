//TEST:SIMPLE(filecheck=ANYHIT): -experimental-feature -entry StandaloneAnyHit -stage anyhit -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=INTERSECTION): -experimental-feature -entry StandaloneIntersection -stage intersection -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

// Neither a schema nor the other stage's compiled implementation is needed to select a stage.
struct Attributes { uint value; }
struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}
struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = uint;
    typealias Primitive = rt::BoundingBoxPrimitive<Attributes>;
}
struct Resources
{
    RWStructuredBuffer<uint> observed;
    uint threshold;
}
ParameterBlock<Resources> resources;

[noinline]
uint candidateFilter(in rt::AnyHitInput<HitContext> input)
{
    input.payload += input.record;
    if (input.attributes.value < resources.threshold)
        input.ignoreHit();
    if (input.attributes.value == resources.threshold)
        input.acceptHitAndEndSearch();
    return input.hitKind;
}
struct StandaloneAnyHit : rt::IAnyHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::AnyHitInput<Context> input)
    {
        input.payload += candidateFilter(input);
        resources.observed[0] = input.payload;
    }
}

[noinline]
bool candidateReport(in rt::IntersectionInput<HitContext> input, float distance)
{
    return input.reportHit(distance, 7, Attributes(input.record));
}
struct StandaloneIntersection : rt::IIntersectionShader
{
    typealias Context = HitContext;
    static void invoke(in rt::IntersectionInput<Context> input)
    {
        float before = input.objectSpaceRay.TMax;
        bool accepted = candidateReport(input, 2);
        float after = input.objectSpaceRay.TMax;
        resources.observed[1] = asuint(before - after) + uint(accepted);
    }
}

// ANYHIT-DAG: visible]]
// ANYHIT-DAG: uint StandaloneAnyHit
// ANYHIT-DAG: candidateFilter
// ANYHIT-DAG: globalResourceContext
// ANYHIT-NOT: metalRTIgnoreHit
// ANYHIT-NOT: metalRTAcceptHitAndEndSearch
// INTERSECTION-DAG: visible]]
// INTERSECTION-DAG: uint StandaloneIntersection
// INTERSECTION-DAG: candidateReport
// INTERSECTION-DAG: globalResourceContext
// INTERSECTION-DAG: portableRTReportHit_{{[0-9]+}}({{.*}}visible_function_table<uint(void thread*, uchar thread*)> [[FILTER:_S[0-9]+]], uint [[SLOT:_S[0-9]+]],
// INTERSECTION-DAG: [[FILTER]]{{\[}}[[SLOT]]{{\]}}(
// INTERSECTION-NOT: metalRTReportHit
