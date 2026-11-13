//TEST:SIMPLE(filecheck=CHECK): -experimental-feature -entry main -target metal -profile metallib_2_4 -validate-ir -dump-ir-after specializeModule

import slang.raytracing.portable;

rt::AccelerationStructure ordinaryScene;
rt::MultiLevelAccelerationStructure<2> explicitScene;
RaytracingAccelerationStructure nativeScene;
RWStructuredBuffer<rt::AccelerationStructure> ordinarySaved;
RWStructuredBuffer<rt::MultiLevelAccelerationStructure<2>> explicitSaved;
RWStructuredBuffer<uint> queryOutput;

// The portable spellings have different AST contracts but share one two-level IR type. Keep
// both values live by saving them. The native query must retain the separate legacy IR type.
[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    ordinarySaved[0] = ordinaryScene;
    explicitSaved[0] = explicitScene;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    RayDesc ray = {};
    ray.Direction = float3(0, 0, 1);
    ray.TMax = 10;
    query.TraceRayInline(nativeScene, RAY_FLAG_NONE, 255, ray);
    while (query.Proceed()) {}
    queryOutput[0] = query.CommittedInstanceIndex();
}

// CHECK: ### AFTER specializeModule:
// CHECK-DAG: let %ordinaryScene : RaytracingAccelerationStructure(2 : Int) = global_param
// CHECK-DAG: let %explicitScene : RaytracingAccelerationStructure(2 : Int) = global_param
// CHECK-DAG: let %nativeScene : RaytracingAccelerationStructure(-1 : Int) = global_param
