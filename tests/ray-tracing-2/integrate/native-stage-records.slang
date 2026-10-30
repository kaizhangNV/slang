//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry Hit -stage closesthit -entry Filter -stage anyhit -entry Intersect -stage intersection -entry Miss -stage miss -entry Callable -stage callable -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry Hit -stage closesthit -entry Filter -stage anyhit -entry Intersect -stage intersection -entry Miss -stage miss -entry Callable -stage callable -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry Hit -stage closesthit -entry Filter -stage anyhit -entry Intersect -stage intersection -entry Miss -stage miss -entry Callable -stage callable -target cuda
//TEST:SIMPLE(filecheck=DXIL): -experimental-feature -entry Intersect -stage intersection -target dxil-asm -profile lib_6_6
//TEST:SIMPLE(filecheck=PTX): -experimental-feature -entry Callable -stage callable -target ptx -Xnvrtc -I"./external/optix-dev/include/"

import slang.raytracing.portable;

struct Record
{
    uint value;
    uint threshold;
}

struct Attributes
{
    uint value;
}

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Primitive = rt::BoundingBoxPrimitive<Attributes>;
    typealias Record = ::Record;
}

struct MissContext : rt::IPayloadContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = ::Record;
}

struct CallableContext : rt::ICallableContext
{
    typealias TraceContext = ::TraceContext;
    typealias CallableData = uint;
    typealias Record = ::Record;
}

// Each stage reads the current invocation's record. Selecting several stages together must not
// share a single global value accidentally, even though all five records have the same data type.
struct Hit : rt::IClosestHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload = input.record.value + input.attributes.value;
    }
}

struct Filter : rt::IAnyHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::AnyHitInput<Context> input)
    {
        if (input.attributes.value < input.record.threshold)
            input.ignoreHit();
    }
}

struct Intersect : rt::IIntersectionShader
{
    typealias Context = HitContext;
    static void invoke(in rt::IntersectionInput<Context> input)
    {
        Attributes attributes = {input.record.value};
        input.reportHit(float(input.record.threshold), 0, attributes);
    }
}

struct Miss : rt::IMissShader
{
    typealias Context = MissContext;
    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload += input.record.value;
    }
}

struct Callable : rt::ICallableShader
{
    typealias Context = CallableContext;
    static void invoke(in rt::CallableInput<Context> input)
    {
        let local = input;
        local.data += local.record.value;
    }
}

// HLSL: cbuffer {{.*}}register(b0)
// HLSL: [shader("closesthit")]void Hit_closesthit(
// HLSL: cbuffer {{.*}}register(b1)
// HLSL: [shader("anyhit")]void Filter_anyhit(
// HLSL: cbuffer {{.*}}register(b2)
// HLSL: [shader("intersection")]void Intersect_intersection()
// HLSL: cbuffer {{.*}}register(b3)
// HLSL: [shader("miss")]void Miss_miss(
// HLSL: cbuffer {{.*}}register(b4)
// HLSL: [shader("callable")]void Callable_callable(
// SPIRV-COUNT-5: OpVariable {{.*}} ShaderRecordBufferKHR
// CUDA-COUNT-6: optixGetSbtDataPointer()
// DXIL: call {{.*}} @dx.op.cbufferLoadLegacy
// DXIL: call {{.*}} @dx.op.reportHit
// PTX: .visible .func __direct_callable__Callable_callable(
