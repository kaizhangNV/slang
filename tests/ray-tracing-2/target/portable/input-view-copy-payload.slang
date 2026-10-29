//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry CopyingMiss -stage miss -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry CopyingMiss -stage miss -target spirv-assembly -emit-spirv-directly
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
// Force inlining makes the final assignment to observed expose that aliasing in native code.
[ForceInline]
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

// HLSL-NOT: struct MissContext
// HLSL-NOT: struct rt_MissInput
// HLSL-DAG: observed{{[_0-9]*}} = 47U;
// HLSL-DAG: [shader("miss")]void CopyingMiss_miss(inout {{.*}} : SV_RayPayload)
// HLSL-NOT: struct MissContext
// HLSL-NOT: struct rt_MissInput
// SPIRV: OpEntryPoint MissKHR
// SPIRV: OpVariable {{.*}} IncomingRayPayloadKHR
// DXIL: define void @
