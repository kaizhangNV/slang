//TEST:SIMPLE(filecheck=HLSLCH): -experimental-feature -entry ProceduralClosestHit -stage closesthit -target hlsl
//TEST:SIMPLE(filecheck=HLSLAH): -experimental-feature -entry ProceduralAnyHit -stage anyhit -target hlsl
//TEST:SIMPLE(filecheck=HLSLIS): -experimental-feature -entry ProceduralIntersection -stage intersection -target hlsl
//TEST:SIMPLE(filecheck=SPIRVCH): -experimental-feature -entry ProceduralClosestHit -stage closesthit -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=SPIRVAH): -experimental-feature -entry ProceduralAnyHit -stage anyhit -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=SPIRVIS): -experimental-feature -entry ProceduralIntersection -stage intersection -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=DXIL): -experimental-feature -entry ProceduralIntersection -stage intersection -target dxil-asm -profile lib_6_6 -validate-ir

import slang.raytracing.portable;

struct Attributes
{
    float2 uv;
}

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Primitive = rt::BoundingBoxPrimitive<Attributes>;
    typealias Record = void;
}

// These independently selected stages know the custom attribute type from their context.
// No schema or hit-group declaration is needed to form any of the three native signatures.
struct ProceduralClosestHit : rt::IClosestHitShader
{
    typealias Context = HitContext;

    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload = asuint(input.attributes.uv.x) + input.primitiveIndex + input.instanceID;
    }
}

struct ProceduralAnyHit : rt::IAnyHitShader
{
    typealias Context = HitContext;

    static void invoke(in rt::AnyHitInput<Context> input)
    {
        if (input.attributes.uv.y < 0)
            input.ignoreHit();
        input.payload += input.hitKind;
    }
}

struct ProceduralIntersection : rt::IIntersectionShader
{
    typealias Context = HitContext;

    static void invoke(in rt::IntersectionInput<Context> input)
    {
        let ray = input.objectSpaceRay;
        Attributes attributes = {ray.Origin.xy};
        // Native ReportHit carries these attributes to any-hit and closest-hit. Intersection
        // does not receive an incoming payload/attribute parameter on D3D or Vulkan.
        input.reportHit(input.minDistance + 1, 7, attributes);
    }
}

// HLSLCH-DAG: PrimitiveIndex()
// HLSLCH-DAG: InstanceID()
// HLSLCH-DAG: [shader("closesthit")]void ProceduralClosestHit_closesthit(inout {{.*}} : SV_RayPayload, {{.*}} : SV_IntersectionAttributes)
// HLSLAH-DAG: IgnoreHit()
// HLSLAH-DAG: HitKind()
// HLSLAH-DAG: [shader("anyhit")]void ProceduralAnyHit_anyhit(inout {{.*}} : SV_RayPayload, {{.*}} : SV_IntersectionAttributes)
// HLSLIS-DAG: ObjectRayOrigin()
// HLSLIS-DAG: ReportHit(
// HLSLIS-DAG: [shader("intersection")]void ProceduralIntersection_intersection()
// SPIRVCH: OpEntryPoint ClosestHitKHR
// SPIRVCH-DAG: OpVariable {{.*}} IncomingRayPayloadKHR
// SPIRVCH-DAG: OpVariable {{.*}} HitAttributeKHR
// SPIRVAH: OpEntryPoint AnyHitKHR
// SPIRVAH-DAG: OpVariable {{.*}} IncomingRayPayloadKHR
// SPIRVAH-DAG: OpVariable {{.*}} HitAttributeKHR
// SPIRVAH: OpIgnoreIntersectionKHR
// SPIRVIS: OpEntryPoint IntersectionKHR
// SPIRVIS-DAG: OpVariable {{.*}} HitAttributeKHR
// SPIRVIS: OpReportIntersectionKHR
// DXIL: define void @
