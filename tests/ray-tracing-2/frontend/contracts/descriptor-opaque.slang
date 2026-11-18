//DIAGNOSTIC_TEST:SIMPLE(diag=RESOURCES): -experimental-feature -entry main -stage compute -target hlsl -DTEST_RESOURCES
//DIAGNOSTIC_TEST:SIMPLE(diag=CALL): -experimental-feature -entry main -stage compute -target hlsl -DTEST_CALL
//DIAGNOSTIC_TEST:SIMPLE(diag=ARGUMENT): -experimental-feature -entry main -stage compute -target hlsl -DTEST_ARGUMENT
//DIAGNOSTIC_TEST:SIMPLE(diag=INITIALIZER): -experimental-feature -entry main -stage compute -target hlsl -DTEST_INITIALIZER
//DIAGNOSTIC_TEST:SIMPLE(diag=OPTIONAL): -experimental-feature -entry main -stage compute -target hlsl -DTEST_OPTIONAL

import slang.raytracing.portable;

struct TestTraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = TestTraceContext;
    typealias HitGroups = rt::EmptyHitGroupList;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

// A descriptor may be bound directly or as part of a parameter block.
rt::TraceProgramDescriptor<Schema, 1> descriptor;

struct Bindings
{
    rt::TraceProgramDescriptor<Schema, 1> descriptor;
}
ParameterBlock<Bindings> bindings;

#ifdef TEST_OPTIONAL
// Optional<T> needs an ordinary default value for its absent case, which an opaque descriptor does
// not provide.
Optional<rt::TraceProgramDescriptor<Schema, 1>> optionalDescriptor;
/*OPTIONAL:
^^^^^^^^ 'Optional<T>' cannot wrap a resource or opaque type
^^^^^^^^ 'rt.TraceProgramDescriptor<Schema, 1, rt.HitRecordLayout.PerGeometry>' is a resource or opaque type and cannot be used as the value type of 'Optional<T>'.
*/
#endif

[numthreads(1, 1, 1)]
void main()
{
#ifdef TEST_RESOURCES
    descriptor.resources;
/*RESOURCES:
               ^^^^^^^^^ member not found
               ^^^^^^^^^ 'resources' is not a member of 'rt.TraceProgramDescriptor<Schema, 1, rt.HitRecordLayout.PerGeometry>'.
*/
#elif defined(TEST_CALL)
    rt::TraceProgramDescriptor<Schema, 1> fromCall = rt::TraceProgramDescriptor<Schema, 1>();
/*CALL:
                                                                                          ^ cannot use initializer list for type 'rt.TraceProgramDescriptor<Schema, 1, rt.HitRecordLayout.PerGeometry>'
*/
#elif defined(TEST_ARGUMENT)
    rt::TraceProgramDescriptor<Schema, 1> fromArgument = rt::TraceProgramDescriptor<Schema, 1>(0);
/*ARGUMENT:
                                                                                              ^ cannot use initializer list for type 'rt.TraceProgramDescriptor<Schema, 1, rt.HitRecordLayout.PerGeometry>'
                                                                                              ^ too many initializers in initializer list
                                                                                              ^ too many initializers (expected 0, got 1)
*/
#elif defined(TEST_INITIALIZER)
    rt::TraceProgramDescriptor<Schema, 1> fromInitializer = {};
/*INITIALIZER:
                                                            ^ cannot use initializer list for type 'rt.TraceProgramDescriptor<Schema, 1, rt.HitRecordLayout.PerGeometry>'
*/
#endif
}
