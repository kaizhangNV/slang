//TEST:SIMPLE(filecheck=HLSLCH): -experimental-feature -entry ClosestHit -stage closesthit -target hlsl
//TEST:SIMPLE(filecheck=HLSLAH): -experimental-feature -entry AnyHit -stage anyhit -target hlsl
//TEST:SIMPLE(filecheck=SPIRVCH): -experimental-feature -entry ClosestHit -stage closesthit -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=SPIRVAH): -experimental-feature -entry AnyHit -stage anyhit -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=DXIL): -experimental-feature -entry ClosestHit -stage closesthit -target dxil-asm -profile lib_6_6 -validate-ir

import slang.raytracing.portable;

struct Payload
{
    float value;
}

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = ::Payload;
    typealias Primitive = rt::TrianglePrimitive;
    typealias Record = void;
}

struct ClosestHit : rt::IClosestHitShader
{
    typealias Context = HitContext;

    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        // TriangleData is a builtin view. Copying it must keep access to both the native
        // barycentrics and front-facing builtin, not transport an unrelated application struct.
        let triangle = input.attributes;
        input.payload.value = triangle.barycentricCoord.x + (triangle.frontFacing ? 1.0 : 0.0);
    }
}

struct AnyHit : rt::IAnyHitShader
{
    typealias Context = HitContext;

    static void invoke(in rt::AnyHitInput<Context> input)
    {
        let triangle = input.attributes;
        input.payload.value = triangle.barycentricCoord.y + (triangle.frontFacing ? 2.0 : 0.0);
    }
}

// HLSLCH-DAG: HitKind()
// HLSLCH-DAG: [shader("closesthit")]void ClosestHit_closesthit(inout {{.*}} : SV_RayPayload, {{.*}} : SV_IntersectionAttributes)
// HLSLAH-DAG: HitKind()
// HLSLAH-DAG: [shader("anyhit")]void AnyHit_anyhit(inout {{.*}} : SV_RayPayload, {{.*}} : SV_IntersectionAttributes)
// SPIRVCH: OpEntryPoint ClosestHitKHR
// SPIRVCH-DAG: OpVariable {{.*}} IncomingRayPayloadKHR
// SPIRVCH-DAG: OpVariable {{.*}} HitAttributeKHR
// SPIRVAH: OpEntryPoint AnyHitKHR
// SPIRVAH-DAG: OpVariable {{.*}} IncomingRayPayloadKHR
// SPIRVAH-DAG: OpVariable {{.*}} HitAttributeKHR
// DXIL: define void @
