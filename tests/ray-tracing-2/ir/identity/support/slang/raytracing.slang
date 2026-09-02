//TEST_IGNORE_FILE:

module raytracing;

namespace rt
{
public interface ITraceProgramLayout {}

public interface IClosestHitShader<Context>
{
    void invoke();
}

public struct TraceProgramDescriptor<ProgramLayout>
    where ProgramLayout : ITraceProgramLayout
{
    int ordinaryField;
}
}
