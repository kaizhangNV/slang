# Structural Ray Tracing Design

This directory defines a structural shader-binding-table API that maps to native D3D/Vulkan ray
tracing and synthesized Metal dispatch.

Start with [PROPOSAL.md](PROPOSAL.md) for the source model and target semantics. Use
[TUTORIAL.md](TUTORIAL.md) for a user-oriented walkthrough and
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for compiler boundaries, sequencing, and tests.
The focused analyses cover [Metal tag inference](METAL_TAG_LIST_ANALYSIS.md),
[payload behavior](RAY_PAYLOAD_MODEL.md), and [callable shaders](CALLABLE_SHADER_CONCERNS.md).

## Design Summary

Shader authors explicitly describe one logical SBT in source:

1. Hit, miss, and callable logic is written as structs implementing the corresponding
   `IClosestHitShader`, `IAnyHitShader`, `IIntersectionShader`, `IMissShader`, and
   `ICallableShader` interfaces.
2. `IHitGroup`, `IMissGroup`, and `ICallableGroup` declarations associate those stages with a
   logical slot and context.
3. Group lists and `ITraceProgramLayout` describe the complete logical SBT. The layout gives Slang
   the finite stage set needed to preserve and synthesize target entry points.
4. `RayTracer<ProgramLayout>` traces through a `TraceProgramDescriptor<ProgramLayout>`. The same
   layout type drives entry synthesis, reflection, and Metal post-trace dispatch.

Stage-input structs contain intrinsic properties rather than stored built-in state. Slang lowers
only the properties reachable from a selected stage, so generated entry-point signatures and Metal
tag lists contain only the required native inputs.

## Target Mapping

| Source contract | D3D/Vulkan | Metal |
| --- | --- | --- |
| `ITraceProgramLayout` | Native pipeline entry points and SBT records | Generated IFT/VFT functions and logical-slot tables |
| `RayTracer.trace` | Existing native trace operation | `intersector::intersect` plus generated post-trace dispatch |
| _ClosestHit_, _Miss_, _Callable_ | Native stages | Visible-function-table dispatch |
| Triangle/curve _AnyHit_ | Native _AnyHit_ where supported | Generated candidate function in the IFT |
| Bounding-box _Intersection_ and _AnyHit_ | Native `ReportHit` and _AnyHit_ control transfer | One generated IFT function that composes both source stages |
| `TraceProgramDescriptor` | No physical shader binding beyond the native pipeline/SBT | Parameter-block-like IFT, VFT, and record-buffer resources |

The compiler infers Metal primitive, topology, motion, level, and optional-data tags from the trace
context, selected capabilities, and reachable stage-input properties. Conflicting requirements are
diagnosed before target emission.

Version one excludes shader execution reordering, `intersection_function_buffer`, and `user_data`.
Those exclusions do not change the logical SBT contract.

## Implementation And Tests

The implementation is organized under:

```text
source/standard-modules/raytracing/        source contracts
source/slang/*structural-ray-tracing*      compiler identity, checks, synthesis, and lowering
tests/ray-tracing-2/                       focused and integration coverage
tools/gfx-unit-test/structural-ray-tracing D3D12/Vulkan runtime host
tools/metal-structural-raytracing-test/    local native Metal runtime host
```

`slang.raytracing` is an explicitly imported experimental standard module. It depends on `core`;
`core` does not depend on it. The compiler performs no structural ray-tracing work when the module
is not imported.

The checked-in [coverage manifest](../../../../tests/ray-tracing-2/coverage-manifest.md) maps the
existing Slang ray-tracing scenarios to focused compiler tests and complete runtime pipelines.
