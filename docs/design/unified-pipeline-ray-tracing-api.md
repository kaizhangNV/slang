# Unified Pipeline Ray Tracing API

The structural ray-tracing API lets shader source declare a logical shader binding table through
`ITraceProgramLayout`. The same layout drives native D3D/Vulkan entry points and records, plus the
function-table and post-trace dispatch synthesized for Metal.

Start with the [design overview](rt-api-workspace/design-sbt-structural-dispatch/README.md), then use:

- [Proposal](rt-api-workspace/design-sbt-structural-dispatch/PROPOSAL.md) for the source contract and
  cross-target semantics.
- [Tutorial](rt-api-workspace/design-sbt-structural-dispatch/TUTORIAL.md) for a shader-author
  walkthrough.
- [Implementation plan](rt-api-workspace/design-sbt-structural-dispatch/IMPLEMENTATION_PLAN.md) for
  compiler boundaries, sequencing, repository structure, and tests.
- [Metal ray-tracing intrinsics](metal-ray-tracing-intrinsics.md) for the target API background.
