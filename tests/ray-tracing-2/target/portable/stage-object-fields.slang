//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry StatefulMiss -stage miss -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry StatefulMiss -stage miss -target spirv-assembly -emit-spirv-directly

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
