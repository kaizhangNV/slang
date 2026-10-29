//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry nativeMiss -stage miss -entry PortableMiss -stage miss -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry nativeMiss -stage miss -entry PortableMiss -stage miss -target spirv-assembly -emit-spirv-directly

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

void updatePayload(inout Payload payload)
{
    payload.value += 7;
}

[shader("miss")]
void nativeMiss(inout Payload payload)
{
    updatePayload(payload);
}

struct PortableMiss : rt::IMissShader
{
    typealias Context = MissContext;

    static void invoke(in rt::MissInput<Context> input)
    {
        updatePayload(input.payload);
    }
}

// The two declaration models can coexist and call the same ordinary payload helper. Only the
// portable entry needs an adapter; the native entry must retain its existing parameter ABI.
// HLSL-DAG: void nativeMiss(inout {{.*}})
// HLSL-DAG: void PortableMiss_miss(inout {{.*}} : SV_RayPayload)
// SPIRV-COUNT-2: OpEntryPoint MissKHR
