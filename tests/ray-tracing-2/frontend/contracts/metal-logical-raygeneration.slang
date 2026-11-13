//TEST:SIMPLE: -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -no-codegen
//TEST:SIMPLE: -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -no-codegen -DTEST_DIRECT=1
//TEST:SIMPLE: -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4 -no-codegen -DTEST_NO_ATTRIBUTE=1
//TEST:SIMPLE: -experimental-feature -entry RayGeneration -target hlsl -profile sm_6_6 -no-codegen
//DIAGNOSTIC_TEST:SIMPLE(diag=LEGACY,non-exhaustive): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -no-codegen -DTEST_LEGACY=1

// Raygeneration permits Metal, and portable trace supports it through direct and generic calls.
// Each call still contributes its own requirements: adding DispatchRaysIndex to a portable trace
// must reject Metal because that intrinsic requires native ray tracing.
import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::EmptyHitGroupList;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

rt::AccelerationStructure scene;
rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<uint> output;

interface ITracer
{
    uint trace();
}

struct Tracer : ITracer
{
    uint trace()
    {
        uint payload = 0;
        rt::RayTracer<Schema> tracer;
        tracer.trace({}, scene, program, payload);
        return payload;
    }
}

uint viaInterface<T : ITracer>(T tracer)
{
    return tracer.trace();
}

#if !defined(TEST_NO_ATTRIBUTE)
[shader("raygeneration")]
#endif
void RayGeneration()
/*LEGACY:
     ^^^^^^^^^^^^^ unavailable features in entry point
*/
{
#if defined(TEST_LEGACY)
    output[0] = Tracer().trace() + DispatchRaysIndex().x;
/*LEGACY:
                                   ^^^^^^^^^^^^^^^^^ see using of 'DispatchRaysIndex'
*/
#elif defined(TEST_DIRECT)
    output[0] = Tracer().trace();
#else
    output[0] = viaInterface(Tracer());
#endif
}
