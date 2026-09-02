//TEST:SIMPLE(filecheck=CHECK): -I tests/ray-tracing-2/ir/identity/support -entry main -stage compute -dump-ir -no-codegen

import slang.raytracing;

// A user search-path module shadows the packaged standard module. Its declarations must remain
// ordinary source types even when the module and declaration names match.
// CHECK: [nameHint("rt.IClosestHitShader")]
// CHECK-NEXT: {{.*}}= interface(
// CHECK-NOT: closest_hit_stage_interface(
// CHECK: [nameHint("rt.TraceProgramDescriptor")]
// CHECK-NEXT: struct {{.*}}: Type
// CHECK-NOT: TraceProgramDescriptor(

struct UserLayout : rt::ITraceProgramLayout {}

rt::TraceProgramDescriptor<UserLayout> userDescriptor;

struct UserStage : rt::IClosestHitShader<int>
{
    void invoke() {}
}

[numthreads(1, 1, 1)]
void main() {}
