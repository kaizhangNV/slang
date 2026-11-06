//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry EmptyPayloadMiss -stage miss -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry EmptyPayloadMiss -stage miss -target spirv-assembly -emit-spirv-directly

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
