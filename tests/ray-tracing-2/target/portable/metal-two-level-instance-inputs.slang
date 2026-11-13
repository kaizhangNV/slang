//TEST:SIMPLE(filecheck=PROVIDER): -experimental-feature -entry ClosestHit -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=PROVIDER): -experimental-feature -entry ClosestHit -target metal -profile metallib_2_4 -DMULTILEVEL=1 -validate-ir
//TEST:SIMPLE(filecheck=ANYHIT): -experimental-feature -entry AnyHit -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=ANYHIT): -experimental-feature -entry AnyHit -target metal -profile metallib_2_4 -DMULTILEVEL=1 -validate-ir
//TEST:SIMPLE(filecheck=TRACE): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=TRACE): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -DMULTILEVEL=1 -validate-ir

import slang.raytracing.portable;

#ifndef MULTILEVEL
#define MULTILEVEL 0
#endif

struct TraceContext : rt::ITraceContext
{
#if MULTILEVEL
    typealias AccelerationStructure = rt::MultiLevelAccelerationStructure<2>;
#else
    typealias AccelerationStructure = rt::AccelerationStructure;
#endif
    typealias Motion = rt::NoMotion;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = void;
    typealias Primitive = rt::TrianglePrimitive;
}

// Ordinary inputs expose scalar properties; explicit multilevel inputs expose an indexed path.
// Both source contracts must use the same fixed visible-function signature for two-level ASes.
[noinline]
uint readInstance(in rt::ClosestHitInput<HitContext> input)
{
#if MULTILEVEL
    return input.instanceCount + input.getInstanceIndex(0) + input.getInstanceID(0);
#else
    return 1 + input.instanceIndex + input.instanceID;
#endif
}

struct ClosestHit : rt::IClosestHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload = readInstance(input);
    }
}

struct AnyHit : rt::IAnyHitShader
{
    typealias Context = HitContext;
    static void invoke(in rt::AnyHitInput<Context> input)
    {
#if MULTILEVEL
        input.payload += input.instanceCount + input.getInstanceIndex(0) + input.getInstanceID(0);
#else
        input.payload += 1 + input.instanceIndex + input.instanceID;
#endif
    }
}

struct HitGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = ::ClosestHit;
    typealias AnyHit = ::AnyHit;
    typealias Intersection = rt::NoIntersection<Context>;
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::HitGroupList<HitGroup>;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

TraceContext.AccelerationStructure scene;
rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<uint> output;

[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.instanceMask = 255;
    ray.sbtStride = 1;
    uint payload = 0;
    rt::RayTracer<Schema> tracer;
    tracer.trace(ray, scene, program, payload);
    output[0] = payload;
}

// Neither spelling needs Metal 3.1's native multilevel traversal types for a single instance.
// PROVIDER-NOT: metal::array_ref
// PROVIDER-NOT: max_levels
// PROVIDER: visible]] void ClosestHit_closesthit(
// PROVIDER-SAME: uint instanceCount
// PROVIDER-SAME: uint thread* instanceIndices
// PROVIDER-SAME: uint thread* instanceIDs
// PROVIDER-NOT: metal::array_ref
// PROVIDER-NOT: max_levels
// Any-hit dispatch transports the same path through its candidate-state pointer.
// ANYHIT-NOT: metal::array_ref
// ANYHIT-NOT: max_levels
// ANYHIT: struct PortableRTCandidateState
// ANYHIT: uint thread* [[INDICES:instanceIndices[_0-9]*]];
// ANYHIT-NEXT: uint thread* [[IDS:instanceIDs[_0-9]*]];
// ANYHIT-NEXT: uint instanceCount
// ANYHIT: return *(candidateState{{[_0-9]*}}->[[INDICES]] +
// ANYHIT: return *(candidateState{{[_0-9]*}}->[[IDS]] +
// ANYHIT: visible]] uint AnyHit_anyhit(void thread* candidateState
// ANYHIT-NOT: metal::array_ref
// ANYHIT-NOT: max_levels
// TRACE-NOT: metal::array_ref
// TRACE-NOT: max_levels
// TRACE: intersector<metal::raytracing::instancing, metal::raytracing::triangle_data, metal::raytracing::world_space_data>
// The committed hit passes scalar Metal builtins as a one-element path to that same signature.
// TRACE: , 1u, &_slang_result.instance_id, &_slang_result.user_instance_id
// TRACE-NOT: metal::array_ref
// TRACE-NOT: max_levels
// The candidate adapter fills the same one-element path without requiring native array_ref.
// TRACE: ->instanceIndices{{[_0-9]*}} = &
// TRACE: ->instanceIDs{{[_0-9]*}} = &
// TRACE: ->instanceCount{{[_0-9]*}} = 1U;
// TRACE: intersection(triangle,
// TRACE-NOT: metal::array_ref
// TRACE-NOT: max_levels
