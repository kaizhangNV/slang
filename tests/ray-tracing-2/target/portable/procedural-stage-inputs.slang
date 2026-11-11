//TEST:SIMPLE(filecheck=HLSLCH): -experimental-feature -entry ProceduralClosestHit -stage closesthit -target hlsl
//TEST:SIMPLE(filecheck=HLSLAH): -experimental-feature -entry ProceduralAnyHit -stage anyhit -target hlsl
//TEST:SIMPLE(filecheck=HLSLIS): -experimental-feature -entry ProceduralIntersection -stage intersection -target hlsl
//TEST:SIMPLE(filecheck=SPIRVCH): -experimental-feature -entry ProceduralClosestHit -stage closesthit -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=SPIRVAH): -experimental-feature -entry ProceduralAnyHit -stage anyhit -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=SPIRVIS): -experimental-feature -entry ProceduralIntersection -stage intersection -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=DXIL): -experimental-feature -entry ProceduralIntersection -stage intersection -target dxil-asm -profile lib_6_6 -validate-ir
//TEST:SIMPLE(filecheck=METALCH): -experimental-feature -entry ProceduralClosestHit -stage closesthit -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=METALAH): -experimental-feature -entry ProceduralAnyHit -stage anyhit -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=METALIS): -experimental-feature -entry ProceduralIntersection -stage intersection -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=CUDACH): -experimental-feature -entry ProceduralClosestHit -stage closesthit -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTXCH): -experimental-feature -entry ProceduralClosestHit -stage closesthit -target ptx -validate-ir -Xnvrtc -I"./external/optix-dev/include/"
//TEST:SIMPLE(filecheck=CUDAAH): -experimental-feature -entry ProceduralAnyHit -stage anyhit -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTXAH): -experimental-feature -entry ProceduralAnyHit -stage anyhit -target ptx -validate-ir -Xnvrtc -I"./external/optix-dev/include/"
//TEST:SIMPLE(filecheck=CUDAIS): -experimental-feature -entry ProceduralIntersection -stage intersection -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTXIS): -experimental-feature -entry ProceduralIntersection -stage intersection -target ptx -validate-ir -Xnvrtc -I"./external/optix-dev/include/"

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

// Metal's closest-hit adapter forwards custom attributes and the invocation's hit IDs.
// METALCH: visible]] void ProceduralClosestHit_closesthit(void thread* [[P:payload_[0-9]+]], void thread* [[A:attributes_[0-9]+]],
// METALCH-SAME: uint [[PRIMITIVE:primitiveIndex_[0-9]+]],
// METALCH-SAME: uint [[INSTANCE:instanceID_[0-9]+]],
// METALCH: *(Attributes_{{[0-9]+}} thread*)((ulong)([[A]]))
// METALCH: ProceduralClosestHit_invoke_{{[0-9]+}}({{.*}}, [[PRIMITIVE]], [[INSTANCE]]);
// Any-hit reads custom attributes from the candidate and reports IgnoreHit via its result.
// METALAH: rt_AnyHitInput_ignoreHit_
// METALAH: {{candidateAttributes_[0-9]+}}))).{{uv_[0-9]+}}.y) < 0.0f
// METALAH: rt_AnyHitInput_ignoreHit_{{[0-9]+}}();
// METALAH: visible]] uint ProceduralAnyHit_anyhit(void thread* candidateState_
// METALAH: ProceduralAnyHit_invoke_{{[0-9]+}}({{.*}}->payload_
// ReportHit writes the candidate attributes and invokes the supplied any-hit table slot.
// METALIS: portableRTReportHit_{{[0-9]+}}({{.*}}visible_function_table<uint(void thread*, uchar thread*)> [[TABLE:[A-Za-z0-9_]+]], uint [[SLOT:[A-Za-z0-9_]+]],
// METALIS: ->candidateAttributes_
// METALIS: [[TABLE]]{{\[}}[[SLOT]]{{\]}}(
// METALIS: ->committedAttributes_
// METALIS: visible]] uint ProceduralIntersection_intersection(void thread* candidateState_
// METALIS-SAME: visible_function_table<uint(void thread*, uchar thread*)>

// OptiX consumes custom attributes as registers and provides builtins through intrinsics.
// CUDACH-DAG: optixGetInstanceId()
// CUDACH-DAG: optixGetPrimitiveIndex()
// CUDACH: __closesthit__ProceduralClosestHit_closesthit()
// CUDACH: optixGetAttribute_0()
// CUDACH: optixGetAttribute_1()
// CUDACH: optixSetPayload_0(
// CUDAAH: __anyhit__ProceduralAnyHit_anyhit()
// CUDAAH: optixGetAttribute_1()
// CUDAAH: optixIgnoreIntersection()
// CUDAAH: optixSetPayload_0(
// CUDAIS: optixReportIntersection(distance_{{[0-9]+}}, hitKind_{{[0-9]+}}, __float_as_uint({{.*}}->uv_{{[0-9]+}})->x)), __float_as_uint({{.*}}->uv_{{[0-9]+}})->y)))
// CUDAIS: __intersection__ProceduralIntersection_intersection()
// PTXCH: .visible .entry __closesthit__ProceduralClosestHit_closesthit()
// PTXCH-DAG: _optix_get_attribute_0
// PTXCH-DAG: _optix_read_primitive_idx
// PTXCH-DAG: _optix_read_instance_id
// PTXCH: _optix_set_payload
// PTXAH: .visible .entry __anyhit__ProceduralAnyHit_anyhit()
// PTXAH: _optix_get_attribute_1
// PTXAH: _optix_ignore_intersection
// PTXAH: _optix_get_hit_kind
// PTXAH: _optix_set_payload
// PTXIS: .visible .entry __intersection__ProceduralIntersection_intersection()
// PTXIS-DAG: _optix_get_object_ray_origin_x
// PTXIS-DAG: _optix_get_object_ray_origin_y
// PTXIS-DAG: _optix_get_ray_tmin
// PTXIS: _optix_report_intersection_2
