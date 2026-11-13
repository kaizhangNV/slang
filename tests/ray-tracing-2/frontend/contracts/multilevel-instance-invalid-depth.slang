//TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target metal -profile metallib_2_4 -DTEST_DEPTH=-1
//TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target metal -profile metallib_2_4 -DTEST_DEPTH=-2
//TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target metal -profile metallib_2_4 -DTEST_DEPTH=0
//TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target metal -profile metallib_2_4 -DTEST_DEPTH=33
//TEST:SIMPLE(filecheck=VALID): -experimental-feature -entry main -target metal -profile metallib_2_4 -DTEST_DEPTH=2 -validate-ir

import slang.raytracing.portable;

// Reject invalid portable depths when producing IR, even when the shader only stores an AS.
// In particular, -1 must not turn a portable binding into the core type's native resource.
rt::MultiLevelAccelerationStructure<TEST_DEPTH> scene;
RWStructuredBuffer<rt::MultiLevelAccelerationStructure<TEST_DEPTH>> saved;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    saved[0] = scene;
}

// INVALID: error[E38096]
// VALID: result code = 0
// VALID: instanceContributionMetadata
// VALID: kernel]] void main
