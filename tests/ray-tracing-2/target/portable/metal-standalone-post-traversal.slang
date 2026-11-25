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

// A standalone selected stage exports a visible adapter with the fixed stage contract.
// No schema or table resource is needed to compile it. Keep a helper call so address-space
// analysis must start from the adapter and propagate through the ordinary invocation body.
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
// MISS: visible]] void StandaloneMiss_miss(MissInvocationContext_{{[0-9]+}} thread* [[CONTEXT:invocation_[0-9]+]])
// MISS: [[CONTEXT]]->globalResources_
// MISS: [[CONTEXT]]->recordData_
// MISS: StandaloneMiss_invoke_
// MISS-NOT: kernel]]
// MISS-NOT: visible_function_table
// MISS-NOT: intersection_function_table
// HIT: visible]] void StandaloneHit_closesthit(ClosestHitInvocationContext_{{[0-9]+}} thread* [[CONTEXT:invocation_[0-9]+]])
// HIT: StandaloneHit_invoke_{{[0-9]+}}({{.*}}[[CONTEXT]]->barycentricCoord_{{[0-9]+}}, [[CONTEXT]]->frontFacing_{{[0-9]+}});
// HIT-NOT: kernel]]
// HIT-NOT: visible_function_table
// CALLABLE: updateCallable
// CALLABLE: visible]] void StandaloneCallable_callable(CallableInvocationContext_{{[0-9]+}} thread* [[CONTEXT:invocation_[0-9]+]])
// CALLABLE: StandaloneCallable_invoke_{{[0-9]+}}([[CONTEXT]]->dispatchRaysIndex_{{[0-9]+}}, (uint thread*)((ulong)([[CONTEXT]]->data_{{[0-9]+}})));
// CALLABLE-NOT: kernel]]
// CALLABLE-NOT: visible_function_table
