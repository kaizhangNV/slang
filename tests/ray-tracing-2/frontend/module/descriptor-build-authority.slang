//TEST:SIMPLE(filecheck=CHECK): -compile-slang-raytracing-module -entry main -stage compute -dump-ir -no-codegen

module raytracing;

namespace rt
{
public interface ITraceProgramLayout {}

// Only the standard-module build is allowed to establish the descriptor's AST and IR identities.
__magic_type(TraceProgramDescriptorType)
__intrinsic_type(TraceProgramDescriptor)
public struct TraceProgramDescriptor<ProgramLayout>
    where ProgramLayout : ITraceProgramLayout
{
    private __init(int unavailable) {}
}
}

struct Layout : rt::ITraceProgramLayout {}

rt::TraceProgramDescriptor<Layout> descriptor;

// CHECK: TraceProgramDescriptor(

[numthreads(1, 1, 1)]
void main() {}
