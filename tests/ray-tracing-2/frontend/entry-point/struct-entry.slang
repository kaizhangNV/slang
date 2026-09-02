//TEST:SIMPLE: -experimental-feature -entry ClosestHit -stage closesthit -no-codegen
//TEST:SIMPLE: -experimental-feature -entry ClosestHit -no-codegen

import slang.raytracing;

RWStructuredBuffer<uint> outputBuffer;

struct Payload {}
struct Record {}

struct TraceContext : rt::ITraceContext
{
    typealias Payload = Payload;
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = TraceContext;
    typealias Primitive = rt::TrianglePrimitive;
    typealias Record = Record;
}

struct ClosestHit : rt::IClosestHitShader<HitContext>
{
    static const uint result = 0x12345678;

    void invoke(rt::ClosestHitInput<HitContext> input)
    {
        outputBuffer[0] = result;
    }
}
