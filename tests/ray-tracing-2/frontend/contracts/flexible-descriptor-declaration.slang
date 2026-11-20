//TEST:SIMPLE: -experimental-feature -no-codegen
//DIAGNOSTIC_TEST:SIMPLE(filecheck=UNSUPPORTED): -experimental-feature -entry main -target hlsl
//DIAGNOSTIC_TEST:SIMPLE(filecheck=UNSUPPORTED): -experimental-feature -entry main -target spirv -emit-spirv-directly
//DIAGNOSTIC_TEST:SIMPLE(filecheck=UNSUPPORTED): -experimental-feature -entry main -target cuda
//DIAGNOSTIC_TEST:SIMPLE(filecheck=UNSUPPORTED): -experimental-feature -entry main -target metal -profile metallib_2_4

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

// The future descriptor can be named and declared, but cannot generate a resource binding yet.
rt::FlexibleTraceProgramDescriptor<Schema> program;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {}

// UNSUPPORTED: FlexibleTraceProgramDescriptor code generation is not implemented
