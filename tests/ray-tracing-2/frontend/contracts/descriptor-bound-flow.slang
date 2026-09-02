//TEST:SIMPLE: -experimental-feature -entry main -stage compute -target hlsl -no-codegen
//DIAGNOSTIC_TEST:SIMPLE(diag=CODEGEN): -experimental-feature -entry main -stage compute -target hlsl
//DIAGNOSTIC_TEST:SIMPLE(diag=CODEGEN): -experimental-feature -entry main -stage compute -target spirv
//DIAGNOSTIC_TEST:SIMPLE(diag=CODEGEN): -experimental-feature -entry main -stage compute -target cuda
//DIAGNOSTIC_TEST:SIMPLE(diag=CODEGEN): -experimental-feature -entry main -stage compute -target metal
//DIAGNOSTIC_TEST:SIMPLE(diag=CODEGEN): -experimental-feature -entry main -stage compute -target hlsl -minimum-slang-optimization
//CODEGEN: structural ray tracing is not supported on this target
//CODEGEN: the current code generation target has no lowering for structural ray-tracing program descriptors

import slang.raytracing;

struct Payload {}

struct TestTraceContext : rt::ITraceContext
{
    typealias Payload = Payload;
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct Layout : rt::ITraceProgramLayout
{
    typealias TraceContext = TestTraceContext;
    typealias HitGroups = rt::NoHitGroups<TestTraceContext>;
    typealias MissGroups = rt::NoMissGroups<TestTraceContext>;
    typealias CallableGroups = rt::NoCallableGroups<TestTraceContext>;
}

rt::TraceProgramDescriptor<Layout> descriptor;

T identity<T>(T value)
{
    return value;
}

[numthreads(1, 1, 1)]
void main()
{
    // Opaque means the descriptor has no source-visible representation or public constructor. It
    // remains a value handle, so ordinary generic code may copy one supplied by host binding.
    rt::TraceProgramDescriptor<Layout> copy = identity(descriptor);
}
