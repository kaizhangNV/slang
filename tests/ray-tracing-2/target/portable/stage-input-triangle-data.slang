//TEST:SIMPLE(filecheck=HLSLCH): -experimental-feature -entry ClosestHit -stage closesthit -target hlsl
//TEST:SIMPLE(filecheck=HLSLAH): -experimental-feature -entry AnyHit -stage anyhit -target hlsl
//TEST:SIMPLE(filecheck=SPIRVCH): -experimental-feature -entry ClosestHit -stage closesthit -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=SPIRVAH): -experimental-feature -entry AnyHit -stage anyhit -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=DXIL): -experimental-feature -entry ClosestHit -stage closesthit -target dxil-asm -profile lib_6_6 -validate-ir
//TEST:SIMPLE(filecheck=METALCH): -experimental-feature -entry ClosestHit -stage closesthit -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=METALAH): -experimental-feature -entry AnyHit -stage anyhit -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=CUDACH): -experimental-feature -entry ClosestHit -stage closesthit -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTXCH): -experimental-feature -entry ClosestHit -stage closesthit -target ptx -validate-ir -Xnvrtc -I"./external/optix-dev/include/"
//TEST:SIMPLE(filecheck=CUDAAH): -experimental-feature -entry AnyHit -stage anyhit -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTXAH): -experimental-feature -entry AnyHit -stage anyhit -target ptx -validate-ir -Xnvrtc -I"./external/optix-dev/include/"

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

// The copied triangle view reads Metal's barycentrics and front-facing value from the
// same invocation, using adapter parameters for closest-hit and candidate state for any-hit.
// METALCH: visible]] void ClosestHit_closesthit(
// METALCH-SAME: float2 [[BARY:barycentricCoord_[0-9]+]], bool [[FRONT:frontFacing_[0-9]+]],
// METALCH: ClosestHit_invoke_{{[0-9]+}}({{.*}}, [[BARY]], [[FRONT]]);
// METALAH: visible]] uint AnyHit_anyhit(void thread* candidateState_
// METALAH: AnyHit_invoke_{{[0-9]+}}({{.*}}->payload_{{[0-9]+}})), [[STATE:[A-Za-z0-9_]+]]->barycentrics_{{[0-9]+}}, [[STATE]]->frontFacing_{{[0-9]+}});

// OptiX gets front-facing from HitKind and the copied view's barycentrics from attributes.
// CUDACH: optixGetHitKind()
// CUDACH: __closesthit__ClosestHit_closesthit()
// CUDACH: optixGetAttribute_0()
// CUDACH: optixGetAttribute_1()
// CUDACH: optixSetPayload_0(
// CUDAAH: optixGetHitKind()
// CUDAAH: __anyhit__AnyHit_anyhit()
// CUDAAH: optixGetAttribute_0()
// CUDAAH: optixGetAttribute_1()
// CUDAAH: optixSetPayload_0(
// PTXCH: .visible .entry __closesthit__ClosestHit_closesthit()
// PTXCH: _optix_get_attribute_0
// PTXCH: _optix_get_hit_kind
// PTXCH: _optix_set_payload
// PTXAH: .visible .entry __anyhit__AnyHit_anyhit()
// PTXAH: _optix_get_attribute_1
// PTXAH: _optix_get_hit_kind
// PTXAH: _optix_set_payload
