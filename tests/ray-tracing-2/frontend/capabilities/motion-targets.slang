// The source motion kinds describe acceleration-structure contents. Metal uses distinct shader
// tags for those kinds, while Vulkan and OptiX use one timed trace operation for every enabled
// kind. Keep the target checks at semantic time because portable descriptor lowering belongs to a
// later implementation slice.

//TEST:SIMPLE: -experimental-feature -target spirv -entry main -stage raygeneration -no-codegen -DPRIMITIVE_MOTION
//TEST:SIMPLE: -experimental-feature -target spirv -entry main -stage raygeneration -no-codegen -DINSTANCE_MOTION
//TEST:SIMPLE: -experimental-feature -target spirv -entry main -stage raygeneration -no-codegen -DCOMBINED_MOTION
//TEST:SIMPLE: -experimental-feature -target cuda -entry main -stage raygeneration -no-codegen -DPRIMITIVE_MOTION
//TEST:SIMPLE: -experimental-feature -target cuda -entry main -stage raygeneration -no-codegen -DINSTANCE_MOTION
//TEST:SIMPLE: -experimental-feature -target cuda -entry main -stage raygeneration -no-codegen -DCOMBINED_MOTION
//DIAGNOSTIC_TEST:SIMPLE(diag=HLSL,non-exhaustive): -experimental-feature -target hlsl -profile sm_6_5 -entry main -stage raygeneration -no-codegen -DPRIMITIVE_MOTION
//DIAGNOSTIC_TEST:SIMPLE(diag=HLSL,non-exhaustive): -experimental-feature -target hlsl -profile sm_6_5 -entry main -stage raygeneration -no-codegen -DINSTANCE_MOTION
//DIAGNOSTIC_TEST:SIMPLE(diag=HLSL,non-exhaustive): -experimental-feature -target hlsl -profile sm_6_5 -entry main -stage raygeneration -no-codegen -DCOMBINED_MOTION
//TEST:SIMPLE: -experimental-feature -target hlsl -profile sm_6_5 -entry main -stage raygeneration -no-codegen

import slang.raytracing;

struct TestPayload {}

struct TestTraceContext : rt::ITraceContext
{
    typealias Payload = TestPayload;
    typealias AccelerationStructure = rt::AccelerationStructure;
#if defined(PRIMITIVE_MOTION)
    typealias Motion = rt::PrimitiveMotion;
#elif defined(INSTANCE_MOTION)
    typealias Motion = rt::InstanceMotion;
#elif defined(COMBINED_MOTION)
    typealias Motion = rt::PrimitiveAndInstanceMotion;
#else
    typealias Motion = rt::NoMotion;
#endif
}

struct Layout : rt::ITraceProgramLayout
{
    typealias TraceContext = TestTraceContext;
    typealias HitGroups = rt::NoHitGroups<TestTraceContext>;
    typealias MissGroups = rt::NoMissGroups<TestTraceContext>;
    typealias CallableGroups = rt::NoCallableGroups<TestTraceContext>;
}

rt::AccelerationStructure scene;
rt::TraceProgramDescriptor<Layout> program;

[shader("raygeneration")]
void main()
// HLSL: ^^^^ unavailable features in entry point
{
    rt::RayTraversalDesc desc = {};
    TestPayload payload = {};
    rt::RayTracer<Layout> tracer;
    tracer.trace(desc, scene, program, payload);
}
