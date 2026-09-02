//DIAGNOSTIC_TEST:SIMPLE(diag=CHECK): -experimental-feature -entry main -stage raygeneration -target hlsl

import slang.raytracing;

struct PayloadData {}

struct MyTraceContext : rt::ITraceContext
{
    typealias Payload = PayloadData;
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct FakeHitGroups : rt::IHitGroupList<MyTraceContext>
//CHECK:                   ^^^^^^^^^^^^^ declaration not accessible
//CHECK:                   ^^^^^^^^^^^^^ 'IHitGroupList' is not accessible from the current context.
{
    static const int count = 1;
}

struct FakeMissGroups : rt::IMissGroupList<MyTraceContext>
//CHECK:                    ^^^^^^^^^^^^^^ declaration not accessible
//CHECK:                    ^^^^^^^^^^^^^^ 'IMissGroupList' is not accessible from the current context.
{
    static const int count = 1;
}

struct FakeCallableGroups : rt::ICallableGroupList<MyTraceContext>
//CHECK:                        ^^^^^^^^^^^^^^^^^^ declaration not accessible
//CHECK:                        ^^^^^^^^^^^^^^^^^^ 'ICallableGroupList' is not accessible from the current context.
{
    static const int count = 1;
}

struct Layout : rt::ITraceProgramLayout
{
    typealias TraceContext = MyTraceContext;
    typealias HitGroups = rt::NoHitGroups<TraceContext>;
    typealias MissGroups = rt::NoMissGroups<TraceContext>;
    typealias CallableGroups = rt::NoCallableGroups<TraceContext>;
}

rt::AccelerationStructure scene;
rt::TraceProgramDescriptor<Layout> descriptor;

[shader("raygeneration")]
void main()
{
    PayloadData payload;
    rt::RayTracer<Layout> tracer;
    tracer.trace({}, scene, descriptor, payload);
}
