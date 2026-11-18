//DIAGNOSTIC_TEST:SIMPLE(diag=PAYLOAD): -experimental-feature -target cuda -entry main
//DIAGNOSTIC_TEST:SIMPLE(diag=NO_PAYLOAD): -experimental-feature -target cuda -entry main -DTEST_NO_PAYLOAD

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::InstanceMotion;
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::EmptyHitGroupList;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

rt::AccelerationStructure scene;
rt::TraceProgramDescriptor<Schema, 1> program;

// Both concrete overloads must reach the unsupported CUDA motion branch. In particular,
// the payload-free wrapper must not silently discard the trace.
[shader("raygeneration")]
void main()
{
    rt::RayTraversalDesc desc = {};
    rt::RayTracer<Schema> tracer;
#ifdef TEST_NO_PAYLOAD
    tracer.trace(desc, scene, program);
//NO_PAYLOAD: error E41400
//NO_PAYLOAD: CUDA structural ray-tracing motion is not implemented
#else
    uint payload = 0;
    tracer.trace(desc, scene, program, payload);
//PAYLOAD: error E41400
//PAYLOAD: CUDA structural ray-tracing motion is not implemented
#endif
}
