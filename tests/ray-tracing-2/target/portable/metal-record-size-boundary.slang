//TEST:SIMPLE(filecheck=VALID): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4 -validate-ir
//DIAGNOSTIC_TEST:SIMPLE(filecheck=OVERFLOW): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4 -DELEMENT_COUNT=1073741817

import slang.raytracing.portable;

#ifndef ELEMENT_COUNT
#define ELEMENT_COUNT 1073741816
#endif

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

// This is a compile-only layout test: no buffer is allocated and the shader never reads these
// elements. The default data size is 4294967264 bytes; adding the 16-byte header and rounding
// to 16 bytes gives 4294967280, which fits uint32. One more uint rounds the stride to 2^32.
struct Record { uint values[ELEMENT_COUNT]; }
struct MissContext : rt::IPayloadContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = ::Record;
}
struct Miss : rt::IMissShader
{
    typealias Context = MissContext;
    static void invoke(rt::MissInput<Context> input) { input.payload = 1; }
}
struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::EmptyHitGroupList;
    typealias MissShaders = rt::MissShaderList<Miss>;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}
rt::AccelerationStructure scene;
rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<uint> output;
[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 100;
    ray.instanceMask = 255;
    uint payload = 0;
    rt::RayTracer<Schema> tracer;
    tracer.trace(ray, scene, program, payload);
    output[0] = payload;
}

// The physical offset expression must retain the exact unsigned 32-bit stride.
// VALID-DAG: __slang_portable_rt_0_miss_0
// VALID-DAG: _slang_miss_record = {{.*}}missIndex{{.*}} * 4294967280U;
// OVERFLOW: Metal ray-tracing Record requires a fixed-size inline storage layout supported by the descriptor ABI
