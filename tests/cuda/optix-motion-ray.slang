// OptiX has one native trace operation whose ray-time operand selects the motion sample. Exercise
// register, pointer, and empty payload transport because each shape selects a different path in the
// generated C++ prelude.

//TEST:SIMPLE(filecheck=TRACE_CUDA): -target cuda -entry traceMotion -stage raygeneration
//TEST:SIMPLE(filecheck=TRACE_PTX): -target ptx -Xnvrtc -I"./external/optix-dev/include/" -entry traceMotion -stage raygeneration
//TEST:SIMPLE(filecheck=TIME_CUDA): -target cuda -entry readMotionTime -stage closesthit
//TEST:SIMPLE(filecheck=TIME_PTX): -target ptx -Xnvrtc -I"./external/optix-dev/include/" -entry readMotionTime -stage closesthit

uniform RaytracingAccelerationStructure scene;

struct Payload
{
    uint value;
}

struct EmptyPayload {}

struct LargePayload
{
    uint values[33];
}

RayDesc makeRay()
{
    RayDesc ray;
    ray.Origin = float3(0.0, 0.0, 0.0);
    ray.TMin = 0.0;
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMax = 100.0;
    return ray;
}

[shader("raygeneration")]
void traceMotion()
{
    RayDesc ray = makeRay();

    Payload payload = {1};
    TraceMotionRay(scene, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, 0.375, payload);

    LargePayload largePayload = {};
    TraceMotionRay(scene, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, 0.5, largePayload);

    EmptyPayload emptyPayload = {};
    TraceMotionRay(scene, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, 0.625, emptyPayload);
}

// TRACE_CUDA: optixTrace({{.*}}, 0.375{{.*}}, &{{.*}});
// TRACE_CUDA: optixTrace({{.*}}, 0.5{{.*}}, &{{.*}});
// TRACE_CUDA: optixTrace({{.*}}, 0.625{{.*}});
// TRACE_PTX: _optix_trace_typed_32

[shader("closesthit")]
void readMotionTime(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes)
{
    payload.value = uint(RayCurrentTime() * 1000.0);
}

// TIME_CUDA: optixGetRayTime()
// TIME_PTX: _optix_get_ray_time
