//TEST:SIMPLE(filecheck=MISS): -experimental-feature -entry StandaloneMiss -stage miss -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=HIT): -experimental-feature -entry StandaloneHit -stage closesthit -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=CALLABLE): -experimental-feature -entry StandaloneCallable -stage callable -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct Record { uint addend; }
struct MissContext : rt::IPayloadContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = ::Record;
}
struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = float3;
    typealias Record = void;
    typealias Primitive = rt::TrianglePrimitive;
}
struct CallableContext : rt::ICallableContext
{
    typealias TraceContext = ::TraceContext;
    typealias CallableData = uint;
    typealias Record = void;
}

StructuredBuffer<uint> values;

// A standalone selected stage is an ordinary exported helper, not a native Metal stage. Its
// caller supplies the used invocation values; no schema, function table, or SBT is fabricated.
// Keep a helper call so address-space analysis must start from the selected exported wrapper.
[noinline]
void updateMiss(rt::MissInput<MissContext> input)
{
    let copy = input;
    copy.payload += values[0] + copy.record.addend;
}

struct StandaloneMiss : rt::IMissShader
{
    typealias Context = MissContext;
    static void invoke(rt::MissInput<Context> input)
    {
        updateMiss(input);
        input.payload += 7;
    }
}

struct StandaloneHit : rt::IClosestHitShader
{
    typealias Context = HitContext;
    static void invoke(rt::ClosestHitInput<Context> input)
    {
        input.payload = float3(input.attributes.barycentricCoord, input.attributes.frontFacing ? 1 : 0);
    }
}

[noinline]
void updateCallable(rt::CallableInput<CallableContext> input)
{
    let copy = input;
    copy.data += copy.dispatchRaysIndex.x;
}

struct StandaloneCallable : rt::ICallableShader
{
    typealias Context = CallableContext;
    static void invoke(rt::CallableInput<Context> input)
    {
        updateCallable(input);
        input.data += 11;
    }
}

// MISS: updateMiss
// MISS: void StandaloneMiss_miss(
// MISS-NOT: kernel void
// MISS-NOT: visible_function_table
// MISS-NOT: intersection_function_table
// HIT: void StandaloneHit_closesthit(
// HIT-NOT: kernel void
// HIT-NOT: visible_function_table
// CALLABLE: updateCallable
// CALLABLE: void StandaloneCallable_callable(
// CALLABLE-NOT: kernel void
// CALLABLE-NOT: visible_function_table
