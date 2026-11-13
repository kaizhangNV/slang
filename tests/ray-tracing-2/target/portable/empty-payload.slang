//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry EmptyPayloadMiss -stage miss -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry EmptyPayloadMiss -stage miss -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry EmptyPayloadMiss -stage miss -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry EmptyPayloadMiss -stage miss -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTX): -experimental-feature -entry EmptyPayloadMiss -stage miss -target ptx -validate-ir -Xnvrtc -I"./external/optix-dev/include/"

import slang.raytracing.portable;

struct EmptyPayload {}

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct MissContext : rt::IPayloadContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = EmptyPayload;
    typealias Record = void;
}

// Empty application payloads are legal. Reuse the ordinary native ABI's empty-payload handling
// rather than requiring a dummy field in every portable context.
struct EmptyPayloadMiss : rt::IMissShader
{
    typealias Context = MissContext;
    static void invoke(in rt::MissInput<Context> input) {}
}

// HLSL: [shader("miss")]void EmptyPayloadMiss_miss(inout {{.*}} : SV_RayPayload)
// SPIRV: OpEntryPoint MissKHR

// Metal keeps its fixed visible-function ABI even when the payload type is empty.
// METAL: visible]] void EmptyPayloadMiss_miss(void thread* payload_
// METAL-NOT: EmptyPayload_
// METAL: return;

// Empty payloads require no OptiX payload registers.
// CUDA: __miss__EmptyPayloadMiss_miss()
// CUDA-NOT: optixGetPayload_
// CUDA-NOT: optixSetPayload_
// CUDA: return;
// PTX: .visible .entry __miss__EmptyPayloadMiss_miss()
// PTX-NOT: _optix_get_payload
// PTX-NOT: _optix_set_payload
// PTX: ret;
