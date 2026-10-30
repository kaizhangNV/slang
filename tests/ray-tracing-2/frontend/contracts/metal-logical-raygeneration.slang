//TEST:SIMPLE: -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -no-codegen
//TEST:SIMPLE: -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -no-codegen -DTEST_DIRECT=1
//TEST:SIMPLE: -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4 -no-codegen -DTEST_NO_ATTRIBUTE=1
//TEST:SIMPLE: -experimental-feature -entry RayGeneration -target hlsl -profile sm_6_6 -no-codegen
//DIAGNOSTIC_TEST:SIMPLE(diag=LEGACY,non-exhaustive): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -no-codegen -DTEST_LEGACY=1
//DIAGNOSTIC_TEST:SIMPLE(diag=NATIVE,non-exhaustive): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -no-codegen -DTEST_NATIVE_PROFILE=1

// Native raygeneration includes native ray-tracing target requirements. On Metal the portable
// path instead has a logical raygeneration stage, including when ordinary generic dispatch hides
// the trace call from the entry's immediate body. This check does not make legacy intrinsics or
// explicitly stronger native shader profiles portable.
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

RaytracingAccelerationStructure scene;
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

#if defined(TEST_NATIVE_PROFILE)
[shader("ser_raygen")]
#elif !defined(TEST_NO_ATTRIBUTE)
[shader("raygeneration")]
#endif
void RayGeneration()
/*LEGACY:
     ^^^^^^^^^^^^^ unavailable features in entry point
*/
/*NATIVE:
     ^^^^^^^^^^^^^ unavailable features in entry point
*/
{
#if defined(TEST_LEGACY)
    output[0] = DispatchRaysIndex().x;
#elif defined(TEST_DIRECT)
    output[0] = Tracer().trace();
#else
    output[0] = viaInterface(Tracer());
#endif
}
