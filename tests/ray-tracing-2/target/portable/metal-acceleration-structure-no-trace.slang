//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry computeMain -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

rt::AccelerationStructure scene;
RWStructuredBuffer<rt::AccelerationStructure> output;

// A compute shader can store a portable AS binding for a later dispatch without tracing it.
// The AS type itself must request lowering: there is no trace instruction, descriptor, or portable
// stage here. Store the whole argument-buffer pointer so the later reader keeps its metadata.
[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{
    output[0] = scene;
}

// METAL: struct [[AS:PortableRTAccelerationStructureResources[A-Za-z0-9_]*]]
// METAL: acceleration_structure<metal::raytracing::instancing> accelerationStructure
// METAL: uint device* instanceContributionMetadata
// METAL: kernel]] void computeMain{{[_0-9]*}}([[AS]] constant* device* [[OUTPUT:output[_0-9]*]]
// METAL-SAME: [[AS]] constant* [[SCENE:scene[_0-9]*]]
// METAL: *([[OUTPUT]]+int(0)) = [[SCENE]];
