// Test that globallycoherent works.

//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage compute -entry main -emit-spirv-directly
//TEST:SIMPLE(filecheck=VK_MEM_MODEL): -target spirv -stage compute -entry main -emit-spirv-directly -capability vk_mem_model

struct S
{
    globallycoherent int member;
}
globallycoherent RWStructuredBuffer<S> buffer;

RWStructuredBuffer<float> output;
[numthreads(4,1,1)]
void main(int tid : SV_DispatchThreadID)
{
    InterlockedAdd(buffer[0].member, 1);
    AllMemoryBarrier();
    output[tid] = buffer[0].member;
    // CHECK-DAG: OpMemberDecorate {{.*}} 0 Coherent
    // CHECK-DAG: OpDecorate %buffer Coherent
    // VK_MEM_MODEL: OpLoad {{.*}} MakePointerVisible|NonPrivatePointer
}
