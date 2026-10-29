//TEST:SIMPLE(filecheck=HLSLCUSTOM): -experimental-feature -entry EmptyCustomClosestHit -stage closesthit -target hlsl
//TEST:SIMPLE(filecheck=HLSLTRIANGLE): -experimental-feature -entry EmptyTriangleAnyHit -stage anyhit -target hlsl
//TEST:SIMPLE(filecheck=HLSLMISS): -experimental-feature -entry EmptyMiss -stage miss -target hlsl
//TEST:SIMPLE(filecheck=HLSLCALLABLE): -experimental-feature -entry EmptyCallable -stage callable -target hlsl

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
