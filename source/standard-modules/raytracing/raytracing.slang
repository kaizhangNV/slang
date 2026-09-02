/**
Structural pipeline ray tracing APIs for Slang.

This module lets shader code describe a logical shader binding table with types. Stage structs
implement the closest-hit, any-hit, intersection, miss, or callable contracts; group declarations
assign those stages to explicit logical slots; and an `ITraceProgramLayout` connects the resulting
layout to `RayTracer` and `TraceProgramDescriptor`.

@remarks EXPERIMENTAL: This module is under active design and may change significantly or be
removed in future versions. Do not use it in production.

@category raytracing
*/
[ExperimentalModule]
module raytracing;

__include "ray-types";
__include "contexts";
__include "stage-inputs";
__include "stage-contracts";
__include "program-layout";
__include "descriptor";
__include "trace";
