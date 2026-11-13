//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry StatefulMiss -stage miss -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry StatefulMiss -stage miss -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry StatefulMiss -stage miss -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry StatefulMiss -stage miss -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTX): -experimental-feature -entry StatefulMiss -stage miss -target ptx -validate-ir -Xnvrtc -I"./external/optix-dev/include/"

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

struct StatefulMiss : rt::IMissShader
{
    typealias Context = MissContext;

    // These are ordinary object fields, not shader parameters. A stage adapter invokes the
    // static method without constructing an object or providing an implicit this parameter.
    uint ordinaryState;
    float4 ordinaryVector;

    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload = 11;
    }
}

// HLSL-NOT: ordinaryState
// HLSL-NOT: ordinaryVector
// HLSL: [shader("miss")]void StatefulMiss_miss(inout {{[^,]+}} : SV_RayPayload)
// HLSL-NOT: ordinaryState
// HLSL-NOT: ordinaryVector
// SPIRV: OpEntryPoint MissKHR
// SPIRV: OpVariable {{.*}} IncomingRayPayloadKHR

// METAL-NOT: ordinaryState
// METAL-NOT: ordinaryVector
// METAL: void [[INVOKE:StatefulMiss_invoke_[0-9]+]](uint thread* [[PAYLOAD:[A-Za-z0-9_]+]])
// METAL: *[[PAYLOAD]] = 11U;
// METAL-NOT: ordinaryState
// METAL-NOT: ordinaryVector
// METAL: visible]] void StatefulMiss_miss(void thread* [[ENTRY_PAYLOAD:payload_[0-9]+]],
// METAL: [[INVOKE]]((uint thread*)((ulong)([[ENTRY_PAYLOAD]])));
// METAL-NOT: ordinaryState
// METAL-NOT: ordinaryVector

// CUDA-NOT: ordinaryState
// CUDA-NOT: ordinaryVector
// CUDA: void [[INVOKE:StatefulMiss_invoke_[0-9]+]](uint * [[PAYLOAD:[A-Za-z0-9_]+]])
// CUDA: *[[PAYLOAD]] = 11U;
// CUDA: __miss__StatefulMiss_miss()
// CUDA: optixSetPayload_0(
// CUDA-NOT: ordinaryState
// CUDA-NOT: ordinaryVector
// PTX: .visible .entry __miss__StatefulMiss_miss()
// PTX: mov.u32 [[VALUE:%r[0-9]+]], 11;
// PTX: call _optix_set_payload, ({{%r[0-9]+}}, [[VALUE]]);
