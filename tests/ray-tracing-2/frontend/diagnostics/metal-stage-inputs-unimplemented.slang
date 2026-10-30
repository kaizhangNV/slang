//TEST:SIMPLE: -experimental-feature -target metal -no-codegen
//DIAGNOSTIC_TEST:SIMPLE(diag=METAL): -experimental-feature -target metal -whole-program
//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -target hlsl -whole-program
//DIAGNOSTIC_TEST:SIMPLE(diag=CUDA): -experimental-feature -target cuda -whole-program -DTEST_CUDA_PENDING_INPUTS=1

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = void;
    typealias Primitive = rt::TrianglePrimitive;
    typealias Record = void;
}

struct ProceduralContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = void;
    typealias Primitive = rt::BoundingBoxPrimitive<float2>;
    typealias Record = void;
}

// Exported helpers keep these bodies reachable during library code generation without selecting
// an entry point. Their ordinary stage-input parameters are legal, and no structural adapter or
// native Metal entry-point check defines an invocation ABI for them. Source checking and module
// serialization remain valid; native library emission needs a selected stage/schema to provide
// these inputs. The target check runs after reachability, without restricting ordinary calls.
[require(portable_raytracing_closesthit)]
export float readClosestHitState(rt::ClosestHitInput<HitContext> input)
{
    return float(input.rayFlags) + input.objectSpaceRay.Origin.x;
//METAL: operation 'metalRTGetRayFlags' in an exported helper has no invocation state provider
//METAL: Metal ray-tracing operation has no stage invocation ABI
//METAL: operation 'metalRTGetObjectSpaceRay' in an exported helper has no invocation state provider
//METAL: Metal ray-tracing operation has no stage invocation ABI
}

[require(portable_raytracing_anyhit)]
export void handleAnyHit(rt::AnyHitInput<HitContext> input)
{
    if (input.objectSpaceRay.Origin.x < 0.0)
//METAL: operation 'metalRTGetObjectSpaceRay' in an exported helper has no invocation state provider
//METAL: Metal ray-tracing operation has no stage invocation ABI
        input.ignoreHit();
//METAL: operation 'metalRTIgnoreHit' in an exported helper has no invocation state provider
//METAL: Metal ray-tracing operation has no stage invocation ABI
    else
        input.acceptHitAndEndSearch();
//METAL: operation 'metalRTAcceptHitAndEndSearch' in an exported helper has no invocation state provider
//METAL: Metal ray-tracing operation has no stage invocation ABI
}

[require(portable_raytracing_intersection)]
export bool reportIntersection(rt::IntersectionInput<ProceduralContext> input)
{
    return input.reportHit(1.0, float2(0.0));
//METAL: operation 'metalRTReportHit' in an exported helper has no invocation state provider
//METAL: Metal ray-tracing operation has no stage invocation ABI
}

[require(portable_raytracing_intersection)]
export bool reportIntersectionWithKind(rt::IntersectionInput<ProceduralContext> input)
{
    return input.reportHit(2.0, 7, float2(0.0));
//METAL: operation 'metalRTReportHit' in an exported helper has no invocation state provider
//METAL: Metal ray-tracing operation has no stage invocation ABI
}

// An unbound triangle view likewise cannot manufacture the current invocation's front face.
[require(portable_raytracing_anyhit_closesthit)]
export bool readFrontFacing()
{
    rt::TriangleData attributes = {};
    return attributes.frontFacing;
//METAL: operation 'metalRTGetTriangleFrontFacing' in an exported helper has no invocation state provider
//METAL: Metal ray-tracing operation has no stage invocation ABI
}

#ifdef TEST_CUDA_PENDING_INPUTS
struct MotionTraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::InstanceMotion;
}

struct MotionHitContext : rt::IHitContext
{
    typealias TraceContext = MotionTraceContext;
    typealias Payload = void;
    typealias Primitive = rt::TrianglePrimitive;
    typealias Record = void;
}

// CUDA still has a pending motion-time input. Keep its exported use out of the HLSL success case,
// whose traversal API does not provide InstanceMotion.
[require(cuda, portable_raytracing_closesthit)]
export float readRayTime(rt::ClosestHitInput<MotionHitContext> input)
{
    return input.time;
//CUDA: error E41400
//CUDA: CUDA structural ray-tracing time
//CUDA: see call to 'rt.ClosestHitInput.time.get'
}
#endif

// Portable implementations must continue to emit their native intrinsics.
//HLSL-DAG: RayFlags()
//HLSL-DAG: ObjectRayOrigin()
//HLSL-DAG: IgnoreHit()
//HLSL-DAG: AcceptHitAndEndSearch()
//HLSL-DAG: ReportHit(
//HLSL-DAG: HitKind()
