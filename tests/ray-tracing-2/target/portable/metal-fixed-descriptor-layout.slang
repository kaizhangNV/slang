//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry RayGeneration -target hlsl -validate-ir
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry RayGeneration -target spirv-assembly -emit-spirv-directly -validate-ir
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry RayGeneration -target cuda -validate-ir
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -target metal -profile metallib_2_4 -validate-ir

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

struct AnyHit : rt::IAnyHitShader
{
    typealias Context = HitContext;
    static void invoke(rt::AnyHitInput<Context> input) { input.payload += 3; }
}

struct ClosestHit : rt::IClosestHitShader
{
    typealias Context = HitContext;
    static void invoke(rt::ClosestHitInput<Context> input) { input.payload += 5; }
}

struct HitGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias AnyHit = ::AnyHit;
    typealias ClosestHit = ::ClosestHit;
    typealias Intersection = rt::NoIntersection<Context>;
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::HitGroupList<HitGroup>;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

rt::AccelerationStructure scene;
rt::TraceProgramDescriptor<Schema, 2> program;
RWStructuredBuffer<uint> output;
uniform uint rayType;

[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTraversalDesc ray = {};
    ray.ray.Direction = float3(0, 0, 1);
    ray.ray.TMax = 10;
    ray.instanceMask = 255;
    ray.rayType = rayType;
    uint payload = 0;
    rt::RayTracer<Schema> tracer;
    tracer.trace(ray, scene, program, payload);
    output[0] = payload;
}

// Native targets still use the fixed geometry multiplier and the runtime ray selector.
// HLSL: TraceRay({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, 2U,
// SPIRV: OpTraceRayKHR
// CUDA: optixTrace({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, 2U,

// The six fields stay present even though this schema has no miss, callable, or procedural stage.
// The final field is an inline resource view, with no intersection-shader VFT or extra pointer.
// METAL-NOT: candidate_triangle
// METAL-DAG: struct [[ENTRY:PortableRTIFTEntry_[A-Za-z0-9_]+]]
// METAL-DAG: metal::raytracing::intersection_function_table<{{.*}}> [[TABLE:table_[0-9]+]];
// METAL-DAG: struct [[VIEW:PortableRTCandidateResources_[A-Za-z0-9_]+]]
// METAL-DAG: metal::visible_function_table<{{.*}}> anyHitFunctions_{{[0-9]+}};
// METAL: struct [[PROGRAM:PortableRTDescriptorResources_[A-Za-z0-9_]+]]
// METAL-NEXT: {
// METAL-NEXT: const [[ENTRY]] device* [[TABLES:tables_[0-9]+]];
// METAL-NEXT: metal::visible_function_table<{{.*}}> missFunctions_{{[0-9]+}};
// METAL-NEXT: metal::visible_function_table<{{.*}}> [[CLOSEST:closestHitFunctions_[0-9]+]];
// METAL-NEXT: metal::visible_function_table<{{.*}}> callableFunctions_{{[0-9]+}};
// METAL-NEXT: const uint device* records_{{[0-9]+}};
// METAL-NEXT: [[VIEW]] candidateResources_{{[0-9]+}};
// METAL-NEXT: };

// The callback's pointer addresses that exact subobject type, not a second layout clone.
// METAL: struct PortableRTRayDataHeader_{{[0-9]+}}
// METAL: [[VIEW]] constant* candidateResources_{{[0-9]+}};

// Select an IFT with the runtime ray type before traversal, then use the hit record's VFT slot.
// A host-assigned slot need not be a schema ordinal; only the absent-provider sentinel filters it.
// METAL-NOT: candidate_triangle
// METAL-NOT: intersectionShaders
// METAL: [[TABLES]] + {{.*}}rayType_{{[0-9]+}})->[[TABLE]]
// METAL: _slang_result.geometry_id * 2U +
// METAL: if (_slang_hit_function_index != 0xffffffffU){{$}}
// METAL: [[CLOSEST]])[_slang_hit_function_index](
// METAL-NOT: candidate_triangle
// METAL-NOT: intersectionShaders
