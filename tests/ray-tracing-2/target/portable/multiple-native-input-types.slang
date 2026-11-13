//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry SharedHit -stage closesthit -entry CallingHit -stage closesthit -entry OtherHit -stage closesthit -entry TriangleHit -stage closesthit -target hlsl -whole-program -validate-ir
//TEST:SIMPLE(filecheck=IR): -experimental-feature -entry SharedHit -stage closesthit -entry CallingHit -stage closesthit -entry OtherHit -stage closesthit -entry TriangleHit -stage closesthit -target hlsl -whole-program -validate-ir -dump-ir-after lowerPortableRayTracingEntryPoints
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry SharedHit -stage closesthit -entry CallingHit -stage closesthit -entry OtherHit -stage closesthit -entry TriangleHit -stage closesthit -target metal -profile metallib_2_4 -whole-program -validate-ir
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry SharedHit -stage closesthit -entry CallingHit -stage closesthit -entry OtherHit -stage closesthit -entry TriangleHit -stage closesthit -target cuda -whole-program -validate-ir
//TEST:SIMPLE(filecheck=PTX): -experimental-feature -entry SharedHit -stage closesthit -entry CallingHit -stage closesthit -entry OtherHit -stage closesthit -entry TriangleHit -stage closesthit -target ptx -whole-program -validate-ir -Xnvrtc -I"./external/optix-dev/include/"

import slang.raytracing.portable;

// Attributes and Record deliberately use the same type. They must remain separate values.
struct SharedData { float value; }
struct OtherAttributes { float2 uv; }
struct OtherRecord { float bias; }

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct HitContext<P : rt::IIntersectionPrimitive, R> : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = float;
    typealias Primitive = P;
    typealias Record = R;
}

typealias SharedContext = HitContext<rt::BoundingBoxPrimitive<SharedData>, SharedData>;
typealias OtherContext = HitContext<rt::BoundingBoxPrimitive<OtherAttributes>, OtherRecord>;
typealias TriangleContext = HitContext<rt::TrianglePrimitive, OtherRecord>;

struct SharedHit : rt::IClosestHitShader
{
    typealias Context = SharedContext;

    [noinline]
    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload += input.attributes.value + input.record.value * 10.0;
    }
}

struct CallingHit : rt::IClosestHitShader
{
    typealias Context = SharedContext;

    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        // This ordinary call must forward CallingHit's attributes and record to SharedHit.
        SharedHit::invoke(input);
        input.payload += input.record.value;
    }
}

struct OtherHit : rt::IClosestHitShader
{
    typealias Context = OtherContext;

    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload = input.attributes.uv.x + input.record.bias;
    }
}

struct TriangleHit : rt::IClosestHitShader
{
    typealias Context = TriangleContext;

    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload = input.attributes.barycentricCoord.y + input.record.bias;
    }
}

// The two entries using SharedData still receive distinct record resources.
// HLSL: cbuffer [[SHARED_RECORD:record_[0-9]+]] : register(
// HLSL: [[SHARED_TYPE:SharedData_[0-9]+]] [[SHARED_RECORD]];
// HLSL: cbuffer [[CALLING_RECORD:record_[0-9]+]] : register(
// HLSL: [[SHARED_TYPE]] [[CALLING_RECORD]];
// HLSL: cbuffer [[OTHER_RECORD:record_[0-9]+]] : register(
// HLSL: [[OTHER_RECORD_TYPE:OtherRecord_[0-9]+]] [[OTHER_RECORD]];
// HLSL: cbuffer [[TRIANGLE_RECORD:record_[0-9]+]] : register(
// HLSL: [[OTHER_RECORD_TYPE]] [[TRIANGLE_RECORD]];

// SharedData appears twice in the helper signature: attributes and record cannot be merged.
// HLSL: void [[SHARED:SharedHit_invoke_[0-9]+]](inout float [[P:[a-zA-Z0-9_]+]], [[SHARED_TYPE]] [[A:[a-zA-Z0-9_]+]], [[SHARED_TYPE]] [[R:[a-zA-Z0-9_]+]])
// HLSL: [[P]] = [[P]] + ([[A]].{{value_[0-9]+}} + [[R]].{{value_[0-9]+}} * 10.0f);
// HLSL: void SharedHit_closesthit(inout {{[a-zA-Z0-9_]+}} [[SP:[a-zA-Z0-9_]+]] : SV_RayPayload, [[SHARED_TYPE]] [[SA:[a-zA-Z0-9_]+]] : SV_IntersectionAttributes)
// HLSL: [[SHARED_TYPE]] [[SR:[a-zA-Z0-9_]+]] = { [[SHARED_RECORD]].{{value_[0-9]+}} };
// HLSL: [[SHARED]]([[SP]].{{data_[0-9]+}}, [[SA]], [[SR]]);

// CallingHit forwards its own record, including when it calls another selected implementation.
// HLSL: void [[CALLING:CallingHit_invoke_[0-9]+]](inout float [[CP:[a-zA-Z0-9_]+]], [[SHARED_TYPE]] [[CA:[a-zA-Z0-9_]+]], [[SHARED_TYPE]] [[CR:[a-zA-Z0-9_]+]])
// HLSL: [[SHARED]]([[CP]], [[CA]], [[CR]]);
// HLSL: [[CP]] = [[CP]] + [[CR]].{{value_[0-9]+}};
// HLSL: void CallingHit_closesthit(inout {{[a-zA-Z0-9_]+}} [[CEP:[a-zA-Z0-9_]+]] : SV_RayPayload, [[SHARED_TYPE]] [[CEA:[a-zA-Z0-9_]+]] : SV_IntersectionAttributes)
// HLSL: [[SHARED_TYPE]] [[CER:[a-zA-Z0-9_]+]] = { [[CALLING_RECORD]].{{value_[0-9]+}} };
// HLSL: [[CALLING]]([[CEP]].{{data_[0-9]+}}, [[CEA]], [[CER]]);

// A different custom attribute type and record type get their own matching parameters.
// HLSL: void [[OTHER:OtherHit_invoke_[0-9]+]](inout float [[OP:[a-zA-Z0-9_]+]], [[OTHER_ATTR_TYPE:OtherAttributes_[0-9]+]] [[OA:[a-zA-Z0-9_]+]], [[OTHER_RECORD_TYPE]] [[OR:[a-zA-Z0-9_]+]])
// HLSL: [[OP]] = [[OA]].{{uv_[0-9]+}}.x + [[OR]].{{bias_[0-9]+}};
// HLSL: void OtherHit_closesthit(inout {{[a-zA-Z0-9_]+}} [[OEP:[a-zA-Z0-9_]+]] : SV_RayPayload, [[OTHER_ATTR_TYPE]] [[OEA:[a-zA-Z0-9_]+]] : SV_IntersectionAttributes)
// HLSL: [[OTHER_RECORD_TYPE]] [[OER:[a-zA-Z0-9_]+]] = { [[OTHER_RECORD]].{{bias_[0-9]+}} };
// HLSL: [[OTHER]]([[OEP]].{{data_[0-9]+}}, [[OEA]], [[OER]]);

// Triangle attributes use the native barycentrics wrapper alongside both custom types.
// HLSL: void [[TRIANGLE:TriangleHit_invoke_[0-9]+]](inout float [[TP:[a-zA-Z0-9_]+]], float2 [[TA:[a-zA-Z0-9_]+]], [[OTHER_RECORD_TYPE]] [[TR:[a-zA-Z0-9_]+]])
// HLSL: [[TP]] = [[TA]].y + [[TR]].{{bias_[0-9]+}};
// HLSL: void TriangleHit_closesthit(inout {{[a-zA-Z0-9_]+}} [[TEP:[a-zA-Z0-9_]+]] : SV_RayPayload, TriangleHitAttributes_{{[0-9]+}} [[TEA:[a-zA-Z0-9_]+]] : SV_IntersectionAttributes)
// HLSL: [[OTHER_RECORD_TYPE]] [[TER:[a-zA-Z0-9_]+]] = { [[TRIANGLE_RECORD]].{{bias_[0-9]+}} };
// HLSL: [[TRIANGLE]]([[TEP]].{{data_[0-9]+}}, [[TEA]].{{barycentrics_[0-9]+}}, [[TER]]);

// IR: ### AFTER lowerPortableRayTracingEntryPoints:
// IR-NOT: portableRTGetPayload
// IR-NOT: portableRTGetHitAttributes
// IR-NOT: portableRTGetTriangleBarycentricCoord
// IR-NOT: portableRTGetRecord
// IR: ###

// Metal must also keep same-typed attributes and Record as separate arguments.
// METAL: void [[SHARED:SharedHit_invoke_[0-9]+]](float thread* [[P:[A-Za-z0-9_]+]], const [[SHARED_TYPE:SharedData_[0-9]+]] thread* [[A:[A-Za-z0-9_]+]], const [[SHARED_TYPE]] thread* [[R:[A-Za-z0-9_]+]])
// METAL: *[[P]] = *[[P]] + ([[A]]->{{value_[0-9]+}} + [[R]]->{{value_[0-9]+}} * 10.0f);
// METAL: void [[CALLING:CallingHit_invoke_[0-9]+]](float thread* [[CP:[A-Za-z0-9_]+]], const [[SHARED_TYPE]] thread* [[CA:[A-Za-z0-9_]+]], const [[SHARED_TYPE]] thread* [[CR:[A-Za-z0-9_]+]])
// METAL: [[SHARED]]([[CP]], [[CA]], [[CR]]);
// METAL: *[[CP]] = *[[CP]] + [[CR]]->{{value_[0-9]+}};
// METAL: void [[OTHER:OtherHit_invoke_[0-9]+]](float thread* {{[A-Za-z0-9_]+}}, const [[OTHER_ATTR:OtherAttributes_[0-9]+]] thread* {{[A-Za-z0-9_]+}}, const [[OTHER_RECORD:OtherRecord_[0-9]+]] thread*
// METAL: void [[TRIANGLE:TriangleHit_invoke_[0-9]+]](float thread* {{[A-Za-z0-9_]+}}, const [[OTHER_RECORD]] thread* {{[A-Za-z0-9_]+}}, float2
// Each adapter loads its own record data, independently of its attributes pointer.
// METAL: visible]] void SharedHit_closesthit(void thread* payload_{{[0-9]+}}, void thread* [[SA:attributes_[0-9]+]],
// METAL-SAME: uchar device* [[SR:recordData_[0-9]+]],
// METAL: *(SharedData_{{[0-9]+}} device*)((ulong)([[SR]]))
// METAL: *(SharedData_{{[0-9]+}} thread*)((ulong)([[SA]]))
// METAL: [[SHARED]](
// METAL: visible]] void CallingHit_closesthit(void thread* payload_{{[0-9]+}}, void thread* [[CEA:attributes_[0-9]+]],
// METAL-SAME: uchar device* [[CER:recordData_[0-9]+]],
// METAL: *(SharedData_{{[0-9]+}} device*)((ulong)([[CER]]))
// METAL: *(SharedData_{{[0-9]+}} thread*)((ulong)([[CEA]]))
// METAL: [[CALLING]](
// METAL: visible]] void OtherHit_closesthit(void thread* payload_{{[0-9]+}}, void thread* [[OA:attributes_[0-9]+]],
// METAL-SAME: uchar device* [[OR:recordData_[0-9]+]],
// METAL: *([[OTHER_RECORD]] device*)((ulong)([[OR]]))
// METAL: *([[OTHER_ATTR]] thread*)((ulong)([[OA]]))
// METAL: [[OTHER]](
// METAL: visible]] void TriangleHit_closesthit(
// METAL-SAME: float2 [[BARY:barycentricCoord_[0-9]+]],
// METAL-SAME: uchar device* [[TR:recordData_[0-9]+]],
// METAL: *([[OTHER_RECORD]] device*)((ulong)([[TR]]))
// METAL: [[TRIANGLE]]({{.*}}, [[BARY]]);

// OptiX forwards payload and custom attributes through helpers. Record reads use the
// current invocation's SBT pointer, including inside another selected stage's invoke.
// CUDA: void [[SHARED:SharedHit_invoke_[0-9]+]](float * [[P:payload_[0-9]+]], [[SHARED_TYPE:SharedData_[0-9]+]] * [[A:attributes_[0-9]+]])
// CUDA: [[A]]->value_
// CUDA: *[[P]] = *[[P]] + ({{.*}} + (*(([[SHARED_TYPE]]*)optixGetSbtDataPointer())).{{value_[0-9]+}} * 10.0f);
// CUDA: __closesthit__SharedHit_closesthit()
// CUDA: optixGetAttribute_0()
// CUDA: [[SHARED]](
// CUDA: void [[CALLING:CallingHit_invoke_[0-9]+]](float * [[CP:payload_[0-9]+]], [[SHARED_TYPE]] * [[CA:attributes_[0-9]+]])
// CUDA: [[SHARED]]([[CP]], [[CA]]);
// CUDA: *[[CP]] = *[[CP]] + (*(([[SHARED_TYPE]]*)optixGetSbtDataPointer())).value_
// CUDA: __closesthit__CallingHit_closesthit()
// CUDA: [[CALLING]](
// CUDA: void OtherHit_invoke_{{[0-9]+}}(float * {{payload_[0-9]+}}, OtherAttributes_{{[0-9]+}} * attributes_
// CUDA: OtherRecord_{{[0-9]+}}*)optixGetSbtDataPointer()
// CUDA: __closesthit__OtherHit_closesthit()
// CUDA: optixGetAttribute_0()
// CUDA: optixGetAttribute_1()
// CUDA: void TriangleHit_invoke_{{[0-9]+}}(float * {{payload_[0-9]+}}, float2 {{ *}}attributes_
// CUDA: OtherRecord_{{[0-9]+}}*)optixGetSbtDataPointer()
// CUDA: __closesthit__TriangleHit_closesthit()
// PTX: .visible .entry __closesthit__SharedHit_closesthit()
// PTX: _optix_get_attribute_0
// PTX: SharedHit_invoke_
// PTX: _optix_set_payload
// PTX: .visible .entry __closesthit__CallingHit_closesthit()
// PTX: _optix_get_attribute_0
// PTX: SharedHit_invoke_
// PTX: _optix_get_sbt_data_ptr_64
// PTX: _optix_set_payload
// PTX: .visible .entry __closesthit__OtherHit_closesthit()
// PTX: _optix_get_attribute_0
// PTX: _optix_get_sbt_data_ptr_64
// PTX: _optix_set_payload
// PTX: .visible .entry __closesthit__TriangleHit_closesthit()
// PTX: _optix_get_attribute_1
// PTX: _optix_get_sbt_data_ptr_64
// PTX: _optix_set_payload
