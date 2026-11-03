//TEST:SIMPLE: -entry RayGeneration -target metal -profile metallib_2_4 -no-codegen
//TEST:SIMPLE: -entry RayGeneration -target metal -profile metallib_2_4 -no-codegen -DTEST_SHORT_NAME=1
//DIAGNOSTIC_TEST:SIMPLE(diag=NATIVE,non-exhaustive): -entry RayGeneration -target metal -profile metallib_2_4 -no-codegen -DTEST_NATIVE_PROFILE=1

// Both raygeneration spellings permit Metal independently of the experimental portable API.
// A profile that requires shader execution reordering still excludes Metal.
// Frontend checking isolates capability compatibility from Metal entry-point lowering.
[require(metal)]
uint metalOnly() { return 1; }

RWStructuredBuffer<uint> output;

#if defined(TEST_NATIVE_PROFILE)
[shader("ser_raygen")]
#elif defined(TEST_SHORT_NAME)
[shader("raygen")]
#else
[shader("raygeneration")]
#endif
void RayGeneration()
/*NATIVE:
     ^^^^^^^^^^^^^ unavailable features in entry point
*/
{
    output[0] = metalOnly();
}
