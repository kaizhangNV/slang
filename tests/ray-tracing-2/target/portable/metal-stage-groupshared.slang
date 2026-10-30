//TEST:SIMPLE(filecheck=UNSUPPORTED): -experimental-feature -entry SharedMiss -stage miss -target metal -profile metallib_2_4
//TEST:SIMPLE(filecheck=UNSUPPORTED): -experimental-feature -entry SharedAnyHit -stage anyhit -target metal -profile metallib_2_4
//TEST:SIMPLE(filecheck=UNUSED): -experimental-feature -entry Compute -stage compute -entry ConstantMiss -stage miss -target metal -profile metallib_2_4 -validate-ir

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
struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = void;
    typealias Primitive = rt::TrianglePrimitive;
}

groupshared uint sharedValues[1];

// Writing before the read makes the issue the missing threadgroup allocation, not an
// uninitialized source value. The reference graph must follow this ordinary helper call.
[noinline]
uint useGroupShared()
{
    sharedValues[0] = 7;
    return sharedValues[0];
}

struct SharedMiss : rt::IMissShader
{
    typealias Context = MissContext;
    static void invoke(rt::MissInput<Context> input) { input.payload = useGroupShared(); }
}
struct SharedAnyHit : rt::IAnyHitShader
{
    typealias Context = HitContext;
    static void invoke(rt::AnyHitInput<Context> input) { input.payload = useGroupShared(); }
}
struct ConstantMiss : rt::IMissShader
{
    typealias Context = MissContext;
    static void invoke(rt::MissInput<Context> input) { input.payload = 13; }
}

RWStructuredBuffer<uint> output;
[shader("compute")]
[numthreads(1, 1, 1)]
void Compute()
{
    output[0] = useGroupShared();
}

// UNSUPPORTED: error
// UNSUPPORTED: groupshared storage in a portable ray-tracing stage
// The groupshared declaration remains live in Compute, but ConstantMiss never accesses it.
// UNUSED-DAG: kernel]] void Compute(
// UNUSED-DAG: visible]] void ConstantMiss_miss(
// UNUSED-DAG: threadgroup array<uint, int(1)>
