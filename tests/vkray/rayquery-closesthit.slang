// rayquery-closesthit.slang
//TEST:SIMPLE(filecheck=CHECK): -profile glsl_460+GL_EXT_ray_tracing -stage closesthit -entry main -target spirv-assembly
//TEST:SIMPLE(filecheck=CHECK): -stage closesthit -entry main -target spirv-assembly -emit-spirv-directly

struct IntersectionPayload
{
    float3 triangleVertice[3];
};

RaytracingAccelerationStructure accelerationStructure;

void main(
	BuiltInTriangleIntersectionAttributes 	attributes,
	in out IntersectionPayload 				ioPayload)
{
	RayQuery<RAY_FLAG_NONE> rayQuery;

	uint instanceInclusionMask = 0x00;
	RayDesc rayDesc;
	rayQuery.TraceRayInline(accelerationStructure, RAY_FLAG_NONE, instanceInclusionMask, rayDesc);

	rayQuery.CandidateGetIntersectionTriangleVertexPositions(ioPayload.triangleVertice);
	rayQuery.CommittedGetIntersectionTriangleVertexPositions(ioPayload.triangleVertice);
}

// CHECK: OpCapability RayQueryKHR
// CHECK: OpCapability RayTracingPositionFetchKHR
// CHECK: OpCapability RayQueryPositionFetchKHR
// CHECK: OpExtension "SPV_KHR_ray_query"
// CHECK: OpExtension "SPV_KHR_ray_tracing_position_fetch"
// CHECK: OpEntryPoint ClosestHitNV %main "main"
// CHECK: OpRayQueryInitializeKHR %rayQuery{{.*}}
// CHECK: OpRayQueryGetIntersectionTriangleVertexPositionsKHR %_arr_v3float_uint_3 %rayQuery{{.*}} %{{u?}}int_0
// CHECK: OpRayQueryGetIntersectionTriangleVertexPositionsKHR %_arr_v3float_uint_3 %rayQuery{{.*}} %{{u?}}int_1
