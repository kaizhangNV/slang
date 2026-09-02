//DIAGNOSTIC_TEST:SIMPLE(diag=CALL): -experimental-feature -entry main -stage compute -target hlsl -DTEST_CALL
//DIAGNOSTIC_TEST:SIMPLE(diag=ARGUMENT): -experimental-feature -entry main -stage compute -target hlsl -DTEST_ARGUMENT
//DIAGNOSTIC_TEST:SIMPLE(diag=INITIALIZER): -experimental-feature -entry main -stage compute -target hlsl -DTEST_INITIALIZER
//DIAGNOSTIC_TEST:SIMPLE(diag=OPTIONAL): -experimental-feature -entry main -stage compute -target hlsl -DTEST_OPTIONAL

import slang.raytracing;

struct Payload {}

struct TestTraceContext : rt::ITraceContext
{
    typealias Payload = Payload;
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct Layout : rt::ITraceProgramLayout
{
    typealias TraceContext = TestTraceContext;
    typealias HitGroups = rt::NoHitGroups<TestTraceContext>;
    typealias MissGroups = rt::NoMissGroups<TestTraceContext>;
    typealias CallableGroups = rt::NoCallableGroups<TestTraceContext>;
}

// A descriptor may be bound directly or as part of a parameter block.
rt::TraceProgramDescriptor<Layout> descriptor;

struct Bindings
{
    rt::TraceProgramDescriptor<Layout> descriptor;
}
ParameterBlock<Bindings> bindings;

#ifdef TEST_OPTIONAL
// Optional<T> needs an ordinary default value for its absent case, which an opaque descriptor does
// not provide.
Optional<rt::TraceProgramDescriptor<Layout>> optionalDescriptor;
/*OPTIONAL:
^^^^^^^^ 'Optional<T>' cannot wrap a resource or opaque type
^^^^^^^^ 'rt.TraceProgramDescriptor<Layout>' is a resource or opaque type and cannot be used as the value type of 'Optional<T>'.
*/
#endif

[numthreads(1, 1, 1)]
void main()
{
#ifdef TEST_CALL
    rt::TraceProgramDescriptor<Layout> fromCall = rt::TraceProgramDescriptor<Layout>();
//CALL:                                                                             ^ not enough arguments to call
//CALL:                                                                             ^ not enough arguments to call (got 0, expected 1)
//CALL: candidate: private rt::TraceProgramDescriptor<Layout>.init(int)
#elif defined(TEST_ARGUMENT)
    rt::TraceProgramDescriptor<Layout> fromArgument = rt::TraceProgramDescriptor<Layout>(0);
//ARGUMENT:                                                                             ^ declaration not accessible
//ARGUMENT:                                                                             ^ 'TraceProgramDescriptor.init' is not accessible from the current context.
//ARGUMENT:                                                                              ^ type mismatch in expression
//ARGUMENT:                                                                              ^ expected an expression of type 'rt.TraceProgramDescriptor<Layout>', got 'int'
#elif defined(TEST_INITIALIZER)
    rt::TraceProgramDescriptor<Layout> fromInitializer = {};
//INITIALIZER:                                           ^ not enough arguments to call
//INITIALIZER:                                           ^ not enough arguments to call (got 0, expected 1)
//INITIALIZER: candidate: private rt::TraceProgramDescriptor<Layout>.init(int)
#endif
}
