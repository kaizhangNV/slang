//TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target metal -profile metallib_2_4 -DTEST_DEPTH=-1
//TEST:SIMPLE(filecheck=FIELD_INVALID): -experimental-feature -entry main -target metal -profile metallib_2_4 -DTEST_DEPTH=-1 -DFIELD=1
//TEST:SIMPLE(filecheck=VALID): -experimental-feature -entry main -target metal -profile metallib_2_4 -DTEST_DEPTH=2 -validate-ir
//TEST:SIMPLE(filecheck=VALID): -experimental-feature -entry main -target metal -profile metallib_2_4 -DTEST_DEPTH=2 -DFIELD=1 -validate-ir

import slang.raytracing.portable;

#ifndef FIELD
#define FIELD 0
#endif

RWStructuredBuffer<rt::AccelerationStructure> saved;

#if FIELD
// The generic field's depth must be checked once the Holder argument is known, even though
// this shader only stores the acceleration structure and never traces it.
struct Holder<let N : int>
{
    rt::MultiLevelAccelerationStructure<N> scene;
}
ParameterBlock<Holder<TEST_DEPTH>> input;
#else
rt::AccelerationStructure scene;

// The first use is in dead code. Caching the lowered type must not put its range check only
// inside that branch: the live use below still requires the same depth to be valid.
[noinline]
void save<let N : int>(rt::AccelerationStructure value)
{
    if (false)
    {
        rt::MultiLevelAccelerationStructure<N> unused =
            bit_cast<rt::MultiLevelAccelerationStructure<N>>(value);
    }
    rt::MultiLevelAccelerationStructure<N> selected =
        bit_cast<rt::MultiLevelAccelerationStructure<N>>(value);
    saved[0] = bit_cast<rt::AccelerationStructure>(selected);
}
#endif

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
#if FIELD
    saved[0] = bit_cast<rt::AccelerationStructure>(input.scene);
#else
    save<TEST_DEPTH>(scene);
#endif
}

// INVALID: error[E41400]
// FIELD_INVALID: error[E38096]
// VALID: result code = 0
// VALID: instanceContributionMetadata
// VALID: kernel]] void main
