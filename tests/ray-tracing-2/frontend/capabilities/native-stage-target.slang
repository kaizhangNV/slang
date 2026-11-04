//TEST:SIMPLE: -entry main -target hlsl -no-codegen
//DIAGNOSTIC_TEST:SIMPLE(diag=METAL,non-exhaustive): -entry main -target metal -no-codegen

// The native miss requirement excludes Metal independently of portable_miss.
[require(miss)]
void missOnlyHelper() {}

[shader("miss")]
void main()
/*METAL:
     ^^^^ unavailable features in entry point
*/
{
    missOnlyHelper();
}
