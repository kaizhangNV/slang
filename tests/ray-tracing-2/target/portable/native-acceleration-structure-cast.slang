//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry RayGeneration -target hlsl -profile sm_6_5 -O0 -validate-ir
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry RayGeneration -target spirv-assembly -emit-spirv-directly -O0 -validate-ir
//TEST:SIMPLE(filecheck=IR): -experimental-feature -entry RayGeneration -target hlsl -profile sm_6_5 -O0 -validate-ir -dump-ir-after collectMetadata
//TEST:SIMPLE(filecheck=MOTION): -experimental-feature -entry RayGeneration -target spirv-assembly -emit-spirv-directly -capability raytracing_motionblur -DMOTION=1 -O0 -validate-ir
//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry RayGeneration -target hlsl -profile sm_6_6 -DSTORED_HANDLE=1 -O0 -validate-ir
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry RayGeneration -target spirv-assembly -emit-spirv-directly -DSTORED_HANDLE=1 -O0 -validate-ir

import slang.raytracing.portable;

#ifndef MOTION
#define MOTION 0
#endif
#ifndef STORED_HANDLE
#define STORED_HANDLE 0
#endif

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
#if MOTION
    typealias Motion = rt::InstanceMotion;
#else
    typealias Motion = rt::NoMotion;
#endif
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::EmptyHitGroupList;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

#if STORED_HANDLE
StructuredBuffer<rt::AccelerationStructure.Handle> scenes;
#else
rt::AccelerationStructure scene;
#endif
RaytracingAccelerationStructure queryScene;
rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<uint> output;

// Keep a portable AS parameter across a helper boundary. Native tracing needs only a view
// of this same resource value; converting it must not rewrite the parameter's IR type.
[noinline]
void traceSelected(rt::AccelerationStructure selected, rt::RayTraversalDesc ray, inout uint payload)
{
    rt::RayTracer<Schema> tracer;
    tracer.trace(ray, selected, program, payload);
}

[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.instanceMask = 255;
    ray.sbtStride = 1;
    ray.time = 0.5;
    uint payload = 0;
#if STORED_HANDLE
    rt::AccelerationStructure selected = scenes[0];
#else
    rt::AccelerationStructure selected = scene;
#endif
    traceSelected(selected, ray, payload);

    // Core RayQuery continues using the legacy AS type in the same compiled entry point.
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(queryScene, RAY_FLAG_NONE, 255, ray.ray);
    while (query.Proceed()) {}
    output[0] = payload + query.CommittedInstanceIndex();
}

// Immediately before emission, the helper parameter and global still use AS(2). Only the
// operand passed to the native intrinsic has the AS(-1) view; no type-normalization pass ran.
// IR: ### AFTER collectMetadata:
// IR: func %traceSelected
// IR: param [[SELECTED:%[A-Za-z0-9_]+]] : RaytracingAccelerationStructure(2 : Int)
// IR: let [[NATIVE:%[A-Za-z0-9_]+]] : RaytracingAccelerationStructure(-1 : Int) = castToNativeAccelerationStructure([[SELECTED]])
// IR: call {{%[A-Za-z0-9_]+}}([[NATIVE]],
// IR-DAG: let %scene : RaytracingAccelerationStructure(2 : Int) = global_param
// IR-DAG: let %queryScene : RaytracingAccelerationStructure(-1 : Int) = global_param

// Even at -O0 the native resource is passed directly, with no resource bit-cast helper.
// HLSL-NOT: bit_cast
// HLSL: void traceSelected{{[_0-9]*}}(RaytracingAccelerationStructure [[SELECTED:selected[_0-9]*]],
// HLSL-NOT: bit_cast
// HLSL: TraceRay([[SELECTED]],
// HLSL-NOT: bit_cast
// HLSL: TraceRayInline(queryScene
// HLSL-NOT: bit_cast

// Native query and portable trace use the same SPIR-V resource type. The trace consumes the
// helper parameter's existing value ID, including when its caller loaded a stored handle.
// SPIRV: [[AS:%[A-Za-z0-9_]+]] = OpTypeAccelerationStructureKHR
// SPIRV-NOT: OpBitcast
// SPIRV-NOT: OpCopyObject
// SPIRV: [[QUERY_AS:%[A-Za-z0-9_]+]] = OpLoad [[AS]] %queryScene
// SPIRV-NEXT: OpRayQueryInitializeKHR {{%[A-Za-z0-9_]+}} [[QUERY_AS]]
// SPIRV-NOT: OpBitcast
// SPIRV-NOT: OpCopyObject
// SPIRV: %traceSelected = OpFunction
// SPIRV-NEXT: [[SELECTED:%[A-Za-z0-9_]+]] = OpFunctionParameter [[AS]]
// SPIRV-NOT: OpBitcast
// SPIRV-NOT: OpCopyObject
// SPIRV: OpTraceRayKHR [[SELECTED]]
// SPIRV-NOT: OpBitcast
// SPIRV-NOT: OpCopyObject

// The motion overload performs the same conversion before OpTraceRayMotionNV.
// MOTION: [[AS:%[A-Za-z0-9_]+]] = OpTypeAccelerationStructureKHR
// MOTION-NOT: OpBitcast
// MOTION-NOT: OpCopyObject
// MOTION: [[QUERY_AS:%[A-Za-z0-9_]+]] = OpLoad [[AS]] %queryScene
// MOTION-NEXT: OpRayQueryInitializeKHR {{%[A-Za-z0-9_]+}} [[QUERY_AS]]
// MOTION-NOT: OpBitcast
// MOTION-NOT: OpCopyObject
// MOTION: %traceSelected = OpFunction
// MOTION-NEXT: [[SELECTED:%[A-Za-z0-9_]+]] = OpFunctionParameter [[AS]]
// MOTION-NOT: OpBitcast
// MOTION-NOT: OpCopyObject
// MOTION: OpTraceRayMotionNV [[SELECTED]]
// MOTION-NOT: OpBitcast
// MOTION-NOT: OpCopyObject
