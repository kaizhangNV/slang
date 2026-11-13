//TEST:COMPILE: tests/ray-tracing-2/frontend/contracts/multilevel-instance-linked-depth.slang -experimental-feature -o tests/ray-tracing-2/frontend/contracts/multilevel-instance-linked-depth.slang-module
//TEST:COMPILE: tests/ray-tracing-2/frontend/contracts/multilevel-instance-linked-depth-provider.slang -DTEST_DEPTH=-1 -o tests/ray-tracing-2/frontend/contracts/multilevel-instance-linked-depth-invalid.slang-module
//TEST:COMPILE: tests/ray-tracing-2/frontend/contracts/multilevel-instance-linked-depth-provider.slang -DTEST_DEPTH=2 -o tests/ray-tracing-2/frontend/contracts/multilevel-instance-linked-depth-valid.slang-module
//TEST:SIMPLE_EX(filecheck=INVALID): -experimental-feature tests/ray-tracing-2/frontend/contracts/multilevel-instance-linked-depth.slang-module -entry main -r tests/ray-tracing-2/frontend/contracts/multilevel-instance-linked-depth-invalid.slang-module -target metal -profile metallib_2_4
//TEST:SIMPLE_EX(filecheck=VALID): -experimental-feature tests/ray-tracing-2/frontend/contracts/multilevel-instance-linked-depth.slang-module -entry main -r tests/ray-tracing-2/frontend/contracts/multilevel-instance-linked-depth-valid.slang-module -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

// This depth is defined by a separately serialized module. The assertion produced for the
// symbolic AS type must survive linking, including when no TraceRay reads its depth.
extern static const int levelCount;
rt::MultiLevelAccelerationStructure<levelCount> scene;
RWStructuredBuffer<rt::MultiLevelAccelerationStructure<levelCount>> saved;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    saved[0] = scene;
}

// INVALID: error[E41400]
// VALID: result code = 0
// VALID: instanceContributionMetadata
// VALID: kernel]] void main
