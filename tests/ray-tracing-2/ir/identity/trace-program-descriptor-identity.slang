//TEST:SIMPLE(filecheck=CHECK): -experimental-feature -entry main -stage compute -dump-ir -no-codegen

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

// The imported serialized module must preserve the compiler-owned operation, including its
// program-layout type argument and conformance witness.
// CHECK: [nameHint("descriptor")]
// CHECK-NEXT: [export({{.*}})]
// CHECK-NEXT: let %descriptor : TraceProgramDescriptor(%Layout, %{{[0-9]+}}) = global_param
// CHECK-NOT: TraceProgramDescriptorResources

[numthreads(1, 1, 1)]
void main() {}
