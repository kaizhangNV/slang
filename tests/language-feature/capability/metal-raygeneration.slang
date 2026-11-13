//TEST:SIMPLE(filecheck=METAL): -entry RayGeneration -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=METAL): -entry RayGeneration -target metal -profile metallib_2_4 -validate-ir -DTEST_SHORT_NAME=1
//TEST:SIMPLE(filecheck=METAL): -entry RayGeneration -target metal -profile metallib_2_4 -validate-ir -DTEST_EMPTY=1
//TEST:SIMPLE(filecheck=METAL): -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4 -validate-ir -DTEST_NO_ATTRIBUTE=1
//DIAGNOSTIC_TEST:SIMPLE(diag=NATIVE,non-exhaustive): -entry RayGeneration -target metal -profile metallib_2_4 -no-codegen -DTEST_NATIVE_PROFILE=1

// Both raygeneration spellings produce Metal kernels without using the portable API.
// Empty bodies and explicit stage selection also need compute-stage lowering.
// A profile that requires shader execution reordering still excludes Metal.
//METAL: kernel]] void RayGeneration(
[require(metal)]
uint metalOnly() { return 1; }

RWStructuredBuffer<uint> output;

#if defined(TEST_NATIVE_PROFILE)
[shader("ser_raygen")]
#elif defined(TEST_SHORT_NAME)
[shader("raygen")]
#elif !defined(TEST_NO_ATTRIBUTE)
[shader("raygeneration")]
#endif
void RayGeneration()
/*NATIVE:
     ^^^^^^^^^^^^^ unavailable features in entry point
*/
{
#if !defined(TEST_EMPTY)
    output[0] = metalOnly();
#endif
}
