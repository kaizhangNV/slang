//TEST:SIMPLE: -experimental-feature -target metal -no-codegen
//DIAGNOSTIC_TEST:SIMPLE(diag=CHECK): -experimental-feature -target metal -whole-program

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::MultiLevelAccelerationStructure<3>;
    typealias Motion = rt::NoMotion;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = void;
    typealias Primitive = rt::BoundingBoxPrimitive<float2>;
    typealias Record = void;
}

// These exported helpers have no selected stage/schema to supply an instance path. Keep source
// and module compilation legal, but diagnose native emission instead of inventing placeholder
// IDs or letting an unbound logical getter reach the emitter.
[require(metal, portable_raytracing_closesthit)]
export uint readInstanceCount(rt::ClosestHitInput<HitContext> input)
{
    return input.instanceCount;
//CHECK: operation 'metalRTGetInstanceCount' in an exported helper has no invocation state provider
//CHECK: Metal ray-tracing operation has no stage invocation ABI
}

[require(metal, portable_raytracing_anyhit)]
export uint readInstanceIndex(rt::AnyHitInput<HitContext> input, uint level)
{
    return input.getInstanceIndex(level);
//CHECK: operation 'metalRTGetInstanceIndex' in an exported helper has no invocation state provider
//CHECK: Metal ray-tracing operation has no stage invocation ABI
}

[require(metal, portable_raytracing_intersection)]
export uint readInstanceID(rt::IntersectionInput<HitContext> input, uint level)
{
    return input.getInstanceID(level);
//CHECK: operation 'metalRTGetInstanceID' in an exported helper has no invocation state provider
//CHECK: Metal ray-tracing operation has no stage invocation ABI
}
