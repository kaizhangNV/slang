// Metal's synthetic ray-tracing stages need separate capability plumbing before a structural
// trace can be validated as a Metal ray-generation entry point. These compute-stage checks isolate
// each motion type's target and version contract from that later adapter work.

//TEST:SIMPLE: -experimental-feature -target metal -profile metallib_2_4 -entry main -stage compute -no-codegen -DPRIMITIVE_MOTION
//TEST:SIMPLE: -experimental-feature -target metal -profile metallib_2_4 -entry main -stage compute -no-codegen -DINSTANCE_MOTION
//TEST:SIMPLE: -experimental-feature -target metal -profile metallib_2_4 -entry main -stage compute -no-codegen -DCOMBINED_MOTION
//DIAGNOSTIC_TEST:SIMPLE(diag=METAL23,non-exhaustive): -experimental-feature -target metal -profile metallib_2_3 -restrictive-capability-check -entry main -stage compute -no-codegen -DPRIMITIVE_MOTION
//DIAGNOSTIC_TEST:SIMPLE(diag=METAL23,non-exhaustive): -experimental-feature -target metal -profile metallib_2_3 -restrictive-capability-check -entry main -stage compute -no-codegen -DINSTANCE_MOTION
//DIAGNOSTIC_TEST:SIMPLE(diag=METAL23,non-exhaustive): -experimental-feature -target metal -profile metallib_2_3 -restrictive-capability-check -entry main -stage compute -no-codegen -DCOMBINED_MOTION

import slang.raytracing;

#if defined(PRIMITIVE_MOTION)
typealias SelectedMotion = rt::PrimitiveMotion;
#elif defined(INSTANCE_MOTION)
typealias SelectedMotion = rt::InstanceMotion;
#elif defined(COMBINED_MOTION)
typealias SelectedMotion = rt::PrimitiveAndInstanceMotion;
#endif

[numthreads(1, 1, 1)]
void main()
// METAL23: ^^^^ unavailable features in entry point
{
    SelectedMotion motion = {};
}
