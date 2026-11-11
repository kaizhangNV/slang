//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry CopyingMiss -stage miss -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry CopyingMiss -stage miss -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry CopyingMiss -stage miss -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=DXIL): -experimental-feature -entry CopyingMiss -stage miss -target dxil-asm -profile lib_6_6 -validate-ir
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry CopyingMiss -stage miss -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTX): -experimental-feature -entry CopyingMiss -stage miss -target ptx -validate-ir

import slang.raytracing.portable;

struct Payload
{
    uint value;
    uint observed;
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

// Copying the logical view must copy its reference to the payload, not the payload itself.
// The emitted helper must receive the same payload reference as its caller.
void overwriteThroughCopy(in rt::MissInput<MissContext> input)
{
    rt::MissInput<MissContext> explicitCopy = input;
    let inferredCopy = explicitCopy;
    rt::MissInput<MissContext>(inferredCopy).payload.value = 47;
}

struct CopyingMiss : rt::IMissShader
{
    typealias Context = MissContext;

    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload.value = 3;
        let forwardedCopy = input;
        overwriteThroughCopy(forwardedCopy);
        // This must observe 47, not the value from before the helper modified its copy.
        input.payload.observed = input.payload.value;
    }
}

// HLSL: void [[OVERWRITE:overwriteThroughCopy[_0-9]*]](inout [[PAYLOAD:Payload[_0-9]*]] [[HELPER:payload[_0-9]*]])
// HLSL: [[HELPER]].value{{[_0-9]*}} = 47U;
// HLSL: void [[INVOKE:CopyingMiss_invoke[_0-9]*]](inout [[PAYLOAD]] [[CALLER:payload[_0-9]*]])
// HLSL: [[CALLER]].value{{[_0-9]*}} = 3U;
// HLSL: [[OVERWRITE]]([[CALLER]]);
// HLSL: [[CALLER]].observed{{[_0-9]*}} = [[CALLER]].value{{[_0-9]*}};
// HLSL: [shader("miss")]void CopyingMiss_miss(inout [[PAYLOAD]] [[ENTRY:payload[_0-9]*]] : SV_RayPayload)
// HLSL: [[INVOKE]]([[ENTRY]]);
// METAL: void [[OVERWRITE:overwriteThroughCopy[_0-9]*]]([[PAYLOAD:Payload[_0-9]*]] thread* [[HELPER:invocationValue[_0-9]*]])
// METAL: [[HELPER]]->value{{[_0-9]*}} = 47U;
// METAL: void [[INVOKE:CopyingMiss_invoke[_0-9]*]]([[PAYLOAD]] thread* [[CALLER:invocationValue[_0-9]*]])
// METAL: [[CALLER]]->value{{[_0-9]*}} = 3U;
// METAL: [[OVERWRITE]]([[CALLER]]);
// METAL: [[CALLER]]->observed{{[_0-9]*}} = [[CALLER]]->value{{[_0-9]*}};
// METAL: {{\[\[visible\]\]}} void CopyingMiss_miss(void thread* [[ENTRY:payload[_0-9]*]],
// METAL: [[INVOKE]](([[PAYLOAD]] thread*)((ulong)([[ENTRY]])));
// SPIRV: OpEntryPoint MissKHR
// SPIRV: OpVariable {{.*}} IncomingRayPayloadKHR
// DXIL: define void @

// OptiX passes one payload pointer through the same ordinary calls, then writes back both fields.
// CUDA: static __device__ void [[OVERWRITE:overwriteThroughCopy[_0-9]*]]([[PAYLOAD:Payload[_0-9]*]] * [[HELPER:payload[_0-9]*]])
// CUDA: [[HELPER]]->value{{[_0-9]*}} = 47U;
// CUDA: static __device__ void [[INVOKE:CopyingMiss_invoke[_0-9]*]]([[PAYLOAD]] * [[CALLER:payload[_0-9]*]])
// CUDA: [[CALLER]]->value{{[_0-9]*}} = 3U;
// CUDA: [[OVERWRITE]]([[CALLER]]);
// CUDA: [[CALLER]]->observed{{[_0-9]*}} = [[CALLER]]->value{{[_0-9]*}};
// CUDA: extern "C" __global__ void __miss__CopyingMiss_miss()
// CUDA: optixGetPayload_0();
// CUDA: optixGetPayload_1();
// CUDA: [[PAYLOAD]] [[ENTRY:_S[0-9]+]];
// CUDA: [[INVOKE]](&[[ENTRY]]);
// CUDA: optixSetPayload_0(
// CUDA: optixSetPayload_1(
// NVRTC resolves the helper and all copies: both output payload registers receive 47.
// PTX: .visible .entry __miss__CopyingMiss_miss()
// PTX: mov.u32 [[RESULT:%r[0-9]+]], 47;
// PTX: call _optix_set_payload, ({{%r[0-9]+}}, [[RESULT]]);
// PTX: call _optix_set_payload, ({{%r[0-9]+}}, [[RESULT]]);
