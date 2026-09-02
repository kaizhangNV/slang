//TEST:SIMPLE: -experimental-feature -entry main -stage compute -no-codegen

import slang.raytracing;

using namespace rt;

struct TestPayload {}
struct TestRecord {}

struct TestTraceContext : rt::ITraceContext
{
    typealias Payload = TestPayload;
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct TestHitContext : rt::IHitContext
{
    typealias TraceContext = TestTraceContext;
    typealias Primitive = rt::TrianglePrimitive;
    typealias Record = TestRecord;
}

struct TestClosestHit : rt::IClosestHitShader<TestHitContext>
{
    void invoke(rt::ClosestHitInput<TestHitContext> input) {}
}

struct TestHitGroup : rt::IHitGroup
{
    typealias Slot = rt::HitGroupSlot<0>;
    typealias Context = TestHitContext;
    typealias ClosestHit = TestClosestHit;
    typealias AnyHit = rt::NoAnyHit<TestHitContext>;
    typealias Intersection = rt::NoIntersection<TestHitContext>;
}

struct TestLayout : rt::ITraceProgramLayout
{
    typealias TraceContext = TestTraceContext;
    typealias HitGroups = rt::HitGroupList<TraceContext, TestHitGroup>;
    typealias MissGroups = rt::NoMissGroups<TraceContext>;
    typealias CallableGroups = rt::NoCallableGroups<TraceContext>;
}

[numthreads(1, 1, 1)]
void main()
{
    // The structural traversal descriptor uses the core RayDesc type and its existing field names
    // directly; importing slang.raytracing does not introduce a second ray representation.
    rt::RayTraversalDesc desc = {};
    desc.ray.Origin = float3(0.0, 0.0, 0.0);
    desc.ray.TMin = 0.001;
    desc.ray.Direction = float3(0.0, 0.0, 1.0);
    desc.ray.TMax = 1000.0;

    // The namespace alias denotes the same core type, so importing it unqualified introduces no
    // duplicate-type ambiguity and preserves ordinary assignment.
    RayDesc ray = desc.ray;
    desc.ray = ray;
}
