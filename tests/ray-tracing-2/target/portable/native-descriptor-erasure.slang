//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry main -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry main -target spirv-assembly -emit-spirv-directly

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::EmptyHitGroupList;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

struct Parameters
{
    rt::TraceProgramDescriptor<Schema> program;
    float scale;
}

ConstantBuffer<Parameters> parameters;
RWStructuredBuffer<float> output;

// Native targets get their SBT from pipeline state. A descriptor in a bound aggregate must
// therefore occupy no storage, even when the aggregate crosses a non-inlined helper boundary.
[noinline]
float readScale(Parameters value)
{
    return value.scale;
}

[shader("raygeneration")]
void main()
{
    output[0] = readScale(parameters);
}

// HLSL: struct Parameters
// HLSL-NEXT: {
// HLSL-NEXT: float scale
// HLSL-NEXT: };
// HLSL: void main()
// SPIRV: OpEntryPoint RayGenerationKHR
// SPIRV: OpMemberDecorate {{.*}} 0 Offset 0
