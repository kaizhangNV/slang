//DIAGNOSTIC_TEST:SIMPLE(filecheck=INVALID): -experimental-feature -no-codegen -DCOUNT=0
//DIAGNOSTIC_TEST:SIMPLE(filecheck=INVALID): -experimental-feature -no-codegen -DCOUNT=-1
//DIAGNOSTIC_TEST:SIMPLE(filecheck=INVALID): -experimental-feature -no-codegen -DCOUNT=16
//DIAGNOSTIC_TEST:SIMPLE(filecheck=INVALID): -experimental-feature -no-codegen -DCOUNT=17 -DSHARED
//TEST:SIMPLE: -experimental-feature -no-codegen -DCOUNT=15
//TEST:SIMPLE: -experimental-feature -no-codegen -DCOUNT=16 -DSHARED
//DIAGNOSTIC_TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target hlsl -DHOLDER -DCOUNT=0
//DIAGNOSTIC_TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target spirv -emit-spirv-directly -DHOLDER -DCOUNT=0
//DIAGNOSTIC_TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target cuda -DHOLDER -DCOUNT=0
//DIAGNOSTIC_TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target metal -DHOLDER -DCOUNT=0
//DIAGNOSTIC_TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target hlsl -DHOLDER -DCOUNT=16
//DIAGNOSTIC_TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target spirv -emit-spirv-directly -DHOLDER -DCOUNT=16
//DIAGNOSTIC_TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target cuda -DHOLDER -DCOUNT=16
//DIAGNOSTIC_TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target metal -DHOLDER -DCOUNT=16
//DIAGNOSTIC_TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target hlsl -DHOLDER -DCOUNT=17 -DSHARED
//DIAGNOSTIC_TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target spirv -emit-spirv-directly -DHOLDER -DCOUNT=17 -DSHARED
//DIAGNOSTIC_TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target cuda -DHOLDER -DCOUNT=17 -DSHARED
//DIAGNOSTIC_TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target metal -DHOLDER -DCOUNT=17 -DSHARED
//TEST:SIMPLE(filecheck=VALID): -experimental-feature -entry main -target hlsl -DHOLDER -DCOUNT=16 -DSHARED
//TEST:SIMPLE(filecheck=VALID): -experimental-feature -entry main -target spirv -emit-spirv-directly -DHOLDER -DCOUNT=16 -DSHARED
//TEST:SIMPLE(filecheck=VALID): -experimental-feature -entry main -target cuda -DHOLDER -DCOUNT=16 -DSHARED
//TEST:SIMPLE(filecheck=VALID): -experimental-feature -entry main -target metal -DHOLDER -DCOUNT=16 -DSHARED
//DIAGNOSTIC_TEST:SIMPLE(filecheck=INVALID): -experimental-feature -entry main -target metal -DHOLDER -DCOUNT=0 -DUNUSED

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

#ifdef HOLDER
// This field starts with symbolic count and layout. A trace using the specialized containing type
// must validate the descriptor. Metal also retains and validates an unused bound descriptor.
struct Holder<let N : int, let Layout : rt::HitRecordLayout>
{
    rt::TraceProgramDescriptor<Schema, N, Layout> program;
}
#ifdef SHARED
ParameterBlock<Holder<COUNT, rt::HitRecordLayout.SharedAcrossGeometries>> holder;
#else
ParameterBlock<Holder<COUNT, rt::HitRecordLayout.PerGeometry>> holder;
#endif
rt::AccelerationStructure scene;
RWStructuredBuffer<uint> output;
[shader("raygeneration")]
void main()
{
#ifndef UNUSED
    uint payload = 0;
    rt::RayTracer<Schema> tracer;
    tracer.trace({}, scene, holder.program, payload);
    output[0] = payload;
#endif
}
#else
// Going through a generic alias must still diagnose invalid counts once N becomes concrete.
#ifdef SHARED
typealias Program<let N : int> =
    rt::TraceProgramDescriptor<Schema, N, rt::HitRecordLayout.SharedAcrossGeometries>;
#else
typealias Program<let N : int> = rt::TraceProgramDescriptor<Schema, N>;
#endif
Program<COUNT> program;
#endif

// INVALID: TraceProgramDescriptor requires rayTypeCount in [1, 15] for PerGeometry or [1, 16] for SharedAcrossGeometries

// VALID: result code = 0
