//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry Stages.Background -stage miss -target hlsl
//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry Stages.Background.invoke -stage miss -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry Stages.Background -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry Stages.Background.invoke -target spirv-assembly -emit-spirv-directly

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
