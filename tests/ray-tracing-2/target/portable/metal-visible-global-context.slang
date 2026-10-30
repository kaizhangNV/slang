//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct MissContext : rt::IPayloadContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = void;
}

StructuredBuffer<uint> background;

// Keep the buffer read behind an ordinary helper so native address-space analysis must visit
// the visible adapter even though it is invoked through a table rather than a source call.
[noinline]
uint loadBackground()
{
    return background[0];
}

struct BufferMiss : rt::IMissShader
{
    typealias Context = MissContext;

    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload = loadBackground();
    }
}

struct ConstantMiss : rt::IMissShader
{
    typealias Context = MissContext;

    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload = 17;
    }
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::EmptyHitGroupList;
    typealias MissShaders = rt::MissShaderList<BufferMiss, ConstantMiss>;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

// This helper never directly reads a global. Its indirect call to BufferMiss must nevertheless
// carry the caller's global context, and ConstantMiss must share the same physical VFT signature.
[noinline]
uint traceIndirect(
    rt::RayTraversalDesc desc,
    rt::AccelerationStructure scene,
    rt::TraceProgramDescriptor<Schema> descriptor)
{
    uint payload = 0;
    rt::RayTracer<Schema> tracer;
    tracer.trace(desc, scene, descriptor, payload);
    return payload;
}

rt::AccelerationStructure scene;
rt::TraceProgramDescriptor<Schema> descriptor;
RWStructuredBuffer<uint> output;
uint selectedMiss;

[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTraversalDesc desc = {};
    desc.ray.Direction = float3(0.0, 0.0, 1.0);
    desc.ray.TMax = 10.0;
    desc.instanceMask = 255;
    desc.missIndex = selectedMiss;
    output[0] = traceIndirect(desc, scene, descriptor);
}

// METAL-DAG: loadBackground
// METAL-DAG: traceIndirect
// METAL-DAG: __slang_portable_rt_0_miss_0({{.*}}uchar thread*{{.*}})
// METAL-DAG: __slang_portable_rt_0_miss_1({{.*}}uchar thread*{{.*}})
// METAL-DAG: [_slang_miss_function_index]({{.*}}, (thread uchar*)
