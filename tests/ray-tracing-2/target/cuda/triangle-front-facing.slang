//TEST:SIMPLE(filecheck=CHECK): -experimental-feature -target cuda -entry closestHit -stage closesthit

import slang.raytracing;

// Structural entry adapters are covered by a later implementation slice. This legacy entry point
// keeps this test focused on the CUDA implementation of the public TriangleData property.
struct Payload
{
    bool frontFacing;
}

[shader("closesthit")]
void closestHit(inout Payload payload, in BuiltInTriangleIntersectionAttributes attributes)
{
    rt::TriangleData triangle = {};
    payload.frontFacing = triangle.frontFacing;
}

//CHECK: optixGetHitKind()
//CHECK: return {{.*}} == 254U;
