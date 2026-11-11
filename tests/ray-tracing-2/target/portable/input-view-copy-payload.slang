//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry CopyingMiss -stage miss -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry CopyingMiss -stage miss -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry CopyingMiss -stage miss -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=DXIL): -experimental-feature -entry CopyingMiss -stage miss -target dxil-asm -profile lib_6_6 -validate-ir

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
