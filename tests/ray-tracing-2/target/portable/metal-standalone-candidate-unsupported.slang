//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -experimental-feature -entry StandaloneAnyHit -stage anyhit -target metal -profile metallib_2_4

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}
struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = void;
    typealias Primitive = rt::TrianglePrimitive;
}
struct StandaloneAnyHit : rt::IAnyHitShader
{
    typealias Context = HitContext;
    static void invoke(rt::AnyHitInput<Context> input) { input.ignoreHit(); }
}

// CHECK: standalone Metal AnyHit/Intersection requires a trace schema
