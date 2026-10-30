//DIAGNOSTIC_TEST:SIMPLE(diag=CHECK,non-exhaustive): -entry RayGeneration -target metal -profile metallib_2_4 -no-codegen

// Without the experimental feature, a native raygeneration profile still excludes Metal.
// No portable module import is needed to verify that the entry-profile behavior stays gated.
[require(metal)]
uint metalOnly() { return 1; }

RWStructuredBuffer<uint> output;

[shader("raygeneration")]
void RayGeneration()
/*CHECK:
     ^^^^^^^^^^^^^ unavailable features in entry point
*/
{
    output[0] = metalOnly();
}
