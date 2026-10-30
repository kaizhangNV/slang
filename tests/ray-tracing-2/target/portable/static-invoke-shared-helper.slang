//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry CallingMiss -stage miss -entry SharedMiss -stage miss -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry CallingMiss -stage miss -entry SharedMiss -stage miss -target spirv-assembly -emit-spirv-directly

import slang.raytracing.portable;

struct Payload
{
    uint value;
}

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct MissContext : rt::IPayloadContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = ::Payload;
    typealias Record = void;
}

struct SharedMiss : rt::IMissShader
{
    typealias Context = MissContext;

    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload.value += 5;
    }
}

[noinline]
void forwardToShared(in rt::MissInput<MissContext> input)
{
    let local = input;
    local.payload.value += 3;
    SharedMiss::invoke(local);
}

struct CallingMiss : rt::IMissShader
{
    typealias Context = MissContext;

    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload.value += 2;
        SharedMiss::invoke(input);
        forwardToShared(input);
    }
}

// SharedMiss is both selected as an entry and called as an ordinary helper. Synthesizing its
// native adapter must not replace the helper with a call to a native entry point or separate
// payload storage: CallingMiss must add 15 to its own payload, while SharedMiss alone adds 5.
// The noinline helper also needs a payload reference rather than a private value copy.
// HLSL-DAG: void forwardToShared{{[a-zA-Z0-9_]*}}(inout
// HLSL-DAG: void CallingMiss_miss(inout {{.*}} : SV_RayPayload)
// HLSL-DAG: void SharedMiss_miss(inout {{.*}} : SV_RayPayload)
// SPIRV-COUNT-2: OpEntryPoint MissKHR
