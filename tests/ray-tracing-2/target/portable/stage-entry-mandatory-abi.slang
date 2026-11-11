//TEST:SIMPLE(filecheck=HLSLCUSTOM): -experimental-feature -entry EmptyCustomClosestHit -stage closesthit -target hlsl
//TEST:SIMPLE(filecheck=HLSLTRIANGLE): -experimental-feature -entry EmptyTriangleAnyHit -stage anyhit -target hlsl
//TEST:SIMPLE(filecheck=HLSLMISS): -experimental-feature -entry EmptyMiss -stage miss -target hlsl
//TEST:SIMPLE(filecheck=HLSLCALLABLE): -experimental-feature -entry EmptyCallable -stage callable -target hlsl
//TEST:SIMPLE(filecheck=METALCUSTOM): -experimental-feature -entry EmptyCustomClosestHit -stage closesthit -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=METALTRIANGLE): -experimental-feature -entry EmptyTriangleAnyHit -stage anyhit -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=METALMISS): -experimental-feature -entry EmptyMiss -stage miss -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=METALCALLABLE): -experimental-feature -entry EmptyCallable -stage callable -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=CUDACUSTOM): -experimental-feature -entry EmptyCustomClosestHit -stage closesthit -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTXCUSTOM): -experimental-feature -entry EmptyCustomClosestHit -stage closesthit -target ptx -validate-ir -Xnvrtc -I"./external/optix-dev/include/"
//TEST:SIMPLE(filecheck=CUDATRIANGLE): -experimental-feature -entry EmptyTriangleAnyHit -stage anyhit -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTXTRIANGLE): -experimental-feature -entry EmptyTriangleAnyHit -stage anyhit -target ptx -validate-ir -Xnvrtc -I"./external/optix-dev/include/"
//TEST:SIMPLE(filecheck=CUDAMISS): -experimental-feature -entry EmptyMiss -stage miss -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTXMISS): -experimental-feature -entry EmptyMiss -stage miss -target ptx -validate-ir -Xnvrtc -I"./external/optix-dev/include/"
//TEST:SIMPLE(filecheck=CUDACALLABLE): -experimental-feature -entry EmptyCallable -stage callable -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTXCALLABLE): -experimental-feature -entry EmptyCallable -stage callable -target ptx -validate-ir -Xnvrtc -I"./external/optix-dev/include/"

import slang.raytracing.portable;

struct Payload
{
    uint value;
}

struct Attributes
{
    float2 uv;
}

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct CustomHitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = ::Payload;
    typealias Primitive = rt::BoundingBoxPrimitive<Attributes>;
    typealias Record = void;
}

struct TriangleHitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = ::Payload;
    typealias Primitive = rt::TrianglePrimitive;
    typealias Record = void;
}

struct MissContext : rt::IPayloadContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = ::Payload;
    typealias Record = void;
}

struct CallableContext : rt::ICallableContext
{
    typealias TraceContext = ::TraceContext;
    typealias CallableData = uint;
    typealias Record = void;
}

// The empty bodies do not read their logical inputs. Native ABI requirements still apply:
// closest-hit/any-hit have payload and attributes, miss has payload, and callable has data.
struct EmptyCustomClosestHit : rt::IClosestHitShader
{
    typealias Context = CustomHitContext;
    static void invoke(in rt::ClosestHitInput<Context> input) {}
}

struct EmptyTriangleAnyHit : rt::IAnyHitShader
{
    typealias Context = TriangleHitContext;
    static void invoke(in rt::AnyHitInput<Context> input) {}
}

struct EmptyMiss : rt::IMissShader
{
    typealias Context = MissContext;
    static void invoke(in rt::MissInput<Context> input) {}
}

struct EmptyCallable : rt::ICallableShader
{
    typealias Context = CallableContext;
    static void invoke(in rt::CallableInput<Context> input) {}
}

// HLSLCUSTOM: [shader("closesthit")]void EmptyCustomClosestHit_closesthit(inout {{.*}} : SV_RayPayload, {{.*}} : SV_IntersectionAttributes)
// HLSLTRIANGLE: [shader("anyhit")]void EmptyTriangleAnyHit_anyhit(inout {{.*}} : SV_RayPayload, {{.*}} : SV_IntersectionAttributes)
// HLSLMISS: [shader("miss")]void EmptyMiss_miss(inout {{.*}} : SV_RayPayload)
// HLSLCALLABLE: [shader("callable")]void EmptyCallable_callable(inout {{[^,)]+}})

// Metal keeps its fixed visible-function signatures even when no input getter is used.
// METALCUSTOM: visible]] void EmptyCustomClosestHit_closesthit(void thread* payload_{{[0-9]+}}, void thread* attributes_
// METALTRIANGLE: visible]] uint EmptyTriangleAnyHit_anyhit(void thread* candidateState_{{[0-9]+}}, uchar thread* globalResourceContext_
// METALMISS: visible]] void EmptyMiss_miss(void thread* payload_
// METALCALLABLE: visible]] void EmptyCallable_callable(void thread* data_{{[0-9]+}}, uint3 dispatchRaysIndex_

// OptiX wrappers retain payload register round-trips even for an empty body. NVRTC may
// remove unused attribute loads, while direct callable entries retain their data pointer ABI.
// CUDACUSTOM: __closesthit__EmptyCustomClosestHit_closesthit()
// CUDACUSTOM: optixGetPayload_0()
// CUDACUSTOM: optixGetAttribute_0()
// CUDACUSTOM: optixGetAttribute_1()
// CUDACUSTOM: optixSetPayload_0(
// CUDATRIANGLE: __anyhit__EmptyTriangleAnyHit_anyhit()
// CUDATRIANGLE: optixGetPayload_0()
// CUDATRIANGLE: optixGetAttribute_0()
// CUDATRIANGLE: optixGetAttribute_1()
// CUDATRIANGLE: optixSetPayload_0(
// CUDAMISS: __miss__EmptyMiss_miss()
// CUDAMISS: optixGetPayload_0()
// CUDAMISS: optixSetPayload_0(
// CUDACALLABLE: __direct_callable__EmptyCallable_callable(uint * data_
// PTXCUSTOM: .visible .entry __closesthit__EmptyCustomClosestHit_closesthit()
// PTXCUSTOM: call ([[PAYLOAD:%r[0-9]+]]), _optix_get_payload, ([[SLOT:%r[0-9]+]]);
// PTXCUSTOM: call _optix_set_payload, ([[SLOT]], [[PAYLOAD]]);
// PTXTRIANGLE: .visible .entry __anyhit__EmptyTriangleAnyHit_anyhit()
// PTXTRIANGLE: call ([[PAYLOAD:%r[0-9]+]]), _optix_get_payload, ([[SLOT:%r[0-9]+]]);
// PTXTRIANGLE: call _optix_set_payload, ([[SLOT]], [[PAYLOAD]]);
// PTXMISS: .visible .entry __miss__EmptyMiss_miss()
// PTXMISS: call ([[PAYLOAD:%r[0-9]+]]), _optix_get_payload, ([[SLOT:%r[0-9]+]]);
// PTXMISS: call _optix_set_payload, ([[SLOT]], [[PAYLOAD]]);
// PTXCALLABLE: .visible .func __direct_callable__EmptyCallable_callable(
// PTXCALLABLE: .param .b64
// PTXCALLABLE: ret;
