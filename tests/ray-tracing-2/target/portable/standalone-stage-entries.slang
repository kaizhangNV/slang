//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry Stages.Background -stage miss -target hlsl
//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry Stages.Background.invoke -stage miss -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry Stages.Background -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry Stages.Background.invoke -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry Stages.Background -stage miss -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry Stages.Background.invoke -stage miss -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry Stages.Background -stage miss -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTX): -experimental-feature -entry Stages.Background -stage miss -target ptx -validate-ir -Xnvrtc -I"./external/optix-dev/include/"
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry Stages.Background.invoke -stage miss -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTX): -experimental-feature -entry Stages.Background.invoke -stage miss -target ptx -validate-ir -Xnvrtc -I"./external/optix-dev/include/"

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct MissContext : rt::IPayloadContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = float3;
    typealias Record = void;
}

namespace Stages
{
struct Background : rt::IMissShader
{
    typealias Context = MissContext;

    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload = float3(0.25, 0.5, 1.0);
    }
}
}

// Neither a program schema nor a descriptor is needed to compile this native miss entry.
// Selecting the type or its invoke method must preserve the same typed native payload ABI.
// HLSL: [shader("miss")]void Stages_Background_miss(inout {{[^,]+}} : SV_RayPayload)
// SPIRV: OpEntryPoint MissKHR
// SPIRV: OpTypePointer IncomingRayPayloadKHR {{.*}}
// SPIRV: OpVariable {{.*}} IncomingRayPayloadKHR

// Both selections produce the same visible adapter and concrete float3 payload access.
// METAL: void [[INVOKE:Stages_Background_invoke_[0-9]+]](float3 thread* [[PAYLOAD:[A-Za-z0-9_]+]])
// METAL: *[[PAYLOAD]] = float3(0.25f, 0.5f, 1.0f);
// METAL: visible]] void Stages_Background_miss(void thread* [[ENTRY_PAYLOAD:payload_[0-9]+]],
// METAL: [[INVOKE]]((float3 thread*)((ulong)([[ENTRY_PAYLOAD]])));

// Selecting either declaration emits the same OptiX entry and writes all three payload words.
// CUDA: void [[INVOKE:Stages_Background_invoke_[0-9]+]](float3 {{ *}}* [[PAYLOAD:[A-Za-z0-9_]+]])
// CUDA: *[[PAYLOAD]] = make_float3 (0.25f, 0.5f, 1.0f);
// CUDA: __miss__Stages_Background_miss()
// CUDA: optixSetPayload_0(
// CUDA: optixSetPayload_1(
// CUDA: optixSetPayload_2(
// PTX: .visible .entry __miss__Stages_Background_miss()
// PTX-COUNT-3: call _optix_set_payload,
