//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry CallingMiss -stage miss -entry SharedMiss -stage miss -target hlsl -whole-program -validate-ir
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry CallingMiss -stage miss -entry SharedMiss -stage miss -target spirv-assembly -emit-spirv-directly

//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry CallingMiss -stage miss -entry SharedMiss -stage miss -target metal -profile metallib_2_4 -whole-program -validate-ir
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry CallingMiss -stage miss -entry SharedMiss -stage miss -target cuda -whole-program -validate-ir
//TEST:SIMPLE(filecheck=PTX): -experimental-feature -entry CallingMiss -stage miss -entry SharedMiss -stage miss -target ptx -whole-program -validate-ir

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
// Check that both direct and forwarded calls use SharedMiss's ordinary implementation
// with the caller's payload reference. The two native wrappers supply their own payloads.
// HLSL: void [[SHARED:SharedMiss_invoke[_0-9]*]](inout [[PAYLOAD:Payload[_0-9]*]] [[SP:payload[_0-9]*]])
// HLSL: [[SP]].value{{[_0-9]*}} = [[SP]].value{{[_0-9]*}} + 5U;
// HLSL: void SharedMiss_miss(inout [[PAYLOAD]] [[SE:payload[_0-9]*]] : SV_RayPayload)
// HLSL: [[SHARED]]([[SE]]);
// HLSL: void [[FORWARD:forwardToShared[_0-9]*]](inout [[PAYLOAD]] [[FP:payload[_0-9]*]])
// HLSL: [[FP]].value{{[_0-9]*}} = [[FP]].value{{[_0-9]*}} + 3U;
// HLSL: [[SHARED]]([[FP]]);
// HLSL: void [[CALLING:CallingMiss_invoke[_0-9]*]](inout [[PAYLOAD]] [[CP:payload[_0-9]*]])
// HLSL: [[CP]].value{{[_0-9]*}} = [[CP]].value{{[_0-9]*}} + 2U;
// HLSL: [[SHARED]]([[CP]]);
// HLSL: [[FORWARD]]([[CP]]);
// HLSL: void CallingMiss_miss(inout [[PAYLOAD]] [[CE:payload[_0-9]*]] : SV_RayPayload)
// HLSL: [[CALLING]]([[CE]]);
// METAL: void [[SHARED:SharedMiss_invoke[_0-9]*]]([[PAYLOAD:Payload[_0-9]*]] thread* [[SP:invocationValue[_0-9]*]])
// METAL: [[SP]]->value{{[_0-9]*}} = [[SP]]->value{{[_0-9]*}} + 5U;
// METAL: void [[FORWARD:forwardToShared[_0-9]*]]([[PAYLOAD]] thread* [[FP:invocationValue[_0-9]*]])
// METAL: [[FP]]->value{{[_0-9]*}} = [[FP]]->value{{[_0-9]*}} + 3U;
// METAL: [[SHARED]]([[FP]]);
// METAL: void [[CALLING:CallingMiss_invoke[_0-9]*]]([[PAYLOAD]] thread* [[CP:invocationValue[_0-9]*]])
// METAL: [[CP]]->value{{[_0-9]*}} = [[CP]]->value{{[_0-9]*}} + 2U;
// METAL: [[SHARED]]([[CP]]);
// METAL: [[FORWARD]]([[CP]]);
// METAL: {{\[\[visible\]\]}} void CallingMiss_miss(void thread* [[CE:payload[_0-9]*]],
// METAL: [[CALLING]](([[PAYLOAD]] thread*)((ulong)([[CE]])));
// METAL: {{\[\[visible\]\]}} void SharedMiss_miss(void thread* [[SE:payload[_0-9]*]],
// METAL: [[SHARED]](([[PAYLOAD]] thread*)((ulong)([[SE]])));
// SPIRV-COUNT-2: OpEntryPoint MissKHR

// OptiX wrappers obtain their own payload; ordinary calls keep forwarding that payload's pointer.
// CUDA: static __device__ void [[SHARED:SharedMiss_invoke[_0-9]*]]([[PAYLOAD:Payload[_0-9]*]] * [[SP:payload[_0-9]*]])
// CUDA: [[SP]]->value{{[_0-9]*}} = [[SP]]->value{{[_0-9]*}} + 5U;
// CUDA: extern "C" __global__ void __miss__SharedMiss_miss()
// CUDA: optixGetPayload_0();
// CUDA: [[PAYLOAD]] [[SE:_S[0-9]+]];
// CUDA: [[SHARED]](&[[SE]]);
// CUDA: optixSetPayload_0([[SE]].value{{[_0-9]*}});
// CUDA: static __device__ void [[FORWARD:forwardToShared[_0-9]*]]([[PAYLOAD]] * [[FP:payload[_0-9]*]])
// CUDA: [[FP]]->value{{[_0-9]*}} = [[FP]]->value{{[_0-9]*}} + 3U;
// CUDA: [[SHARED]]([[FP]]);
// CUDA: static __device__ void [[CALLING:CallingMiss_invoke[_0-9]*]]([[PAYLOAD]] * [[CP:payload[_0-9]*]])
// CUDA: [[CP]]->value{{[_0-9]*}} = [[CP]]->value{{[_0-9]*}} + 2U;
// CUDA: [[SHARED]]([[CP]]);
// CUDA: [[FORWARD]]([[CP]]);
// CUDA: extern "C" __global__ void __miss__CallingMiss_miss()
// CUDA: optixGetPayload_0();
// CUDA: [[PAYLOAD]] [[CE:_S[0-9]+]];
// CUDA: [[CALLING]](&[[CE]]);
// CUDA: optixSetPayload_0([[CE]].value{{[_0-9]*}});
// The compiled PTX adds 5 for the standalone entry, and 2+5+3+5=15 for its caller.
// PTX: .visible .entry __miss__SharedMiss_miss()
// PTX: call ([[SP:%r[0-9]+]]), _optix_get_payload,
// PTX: add.{{[su]}}32 [[SR:%r[0-9]+]], [[SP]], 5;
// PTX: call _optix_set_payload, ({{%r[0-9]+}}, [[SR]]);
// PTX: .visible .entry __miss__CallingMiss_miss()
// PTX: call ([[CP:%r[0-9]+]]), _optix_get_payload,
// PTX: add.{{[su]}}32 [[CR:%r[0-9]+]], [[CP]], 15;
// PTX: call _optix_set_payload, ({{%r[0-9]+}}, [[CR]]);
