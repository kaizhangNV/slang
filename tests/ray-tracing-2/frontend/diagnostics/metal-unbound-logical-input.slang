//TEST:SIMPLE: -experimental-feature -target metal -no-codegen
//DIAGNOSTIC_TEST:SIMPLE(diag=UNBOUND): -experimental-feature -entry main -stage compute -target metal -profile metallib_2_4 -whole-program -validate-ir
//TEST:SIMPLE(filecheck=BOUND): -experimental-feature -entry Miss -stage miss -entry Callable -stage callable -target metal -profile metallib_2_4 -whole-program -validate-ir
//TEST:SIMPLE(filecheck=DEAD): -experimental-feature -DTEST_UNEXPORTED=1 -entry main -stage compute -target metal -profile metallib_2_4 -whole-program -validate-ir

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct MissContext : rt::IPayloadContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = uint;
}

struct CallableContext : rt::ICallableContext
{
    typealias TraceContext = ::TraceContext;
    typealias CallableData = uint;
    typealias Record = void;
}

// Without a selected stage calling these exports, Metal has no invocation values to supply.
// Diagnose the getters before empty-type legalization removes their logical input operand.
#ifndef TEST_UNEXPORTED
export
#endif
void updatePayload(in rt::MissInput<MissContext> input)
{
    input.payload += 1;
//UNBOUND: operation 'portableRTGetPayload' in an exported helper has no invocation state provider
//UNBOUND: Metal ray-tracing operation has no stage invocation ABI
}

#ifndef TEST_UNEXPORTED
export
#endif
uint readRecord(in rt::MissInput<MissContext> input)
{
    return input.record;
//UNBOUND: operation 'portableRTGetRecord' in an exported helper has no invocation state provider
//UNBOUND: Metal ray-tracing operation has no stage invocation ABI
}

#ifndef TEST_UNEXPORTED
export
#endif
void updateData(in rt::CallableInput<CallableContext> input)
{
    input.data += 3;
//UNBOUND: operation 'portableRTGetCallableData' in an exported helper has no invocation state provider
//UNBOUND: Metal ray-tracing operation has no stage invocation ABI
}

// Selecting these stages makes the same exported helpers valid: their ordinary callers pass
// the values supplied by the generated Metal wrappers.
struct Miss : rt::IMissShader
{
    typealias Context = MissContext;
    static void invoke(in rt::MissInput<Context> input)
    {
        updatePayload(input);
        input.payload += readRecord(input);
    }
}

struct Callable : rt::ICallableShader
{
    typealias Context = CallableContext;
    static void invoke(in rt::CallableInput<Context> input)
    {
        updateData(input);
    }
}

// With exports removed, the unused helpers disappear before the invocation-state check.
[numthreads(1, 1, 1)]
void main() {}

// BOUND-DAG: void updatePayload{{[a-zA-Z0-9_]*}}(uint thread* {{[a-zA-Z0-9_]+}})
// BOUND-DAG: uint readRecord{{[a-zA-Z0-9_]*}}(uint {{[a-zA-Z0-9_]+}})
// BOUND-DAG: void updateData{{[a-zA-Z0-9_]*}}(uint thread* {{[a-zA-Z0-9_]+}})
// BOUND-DAG: visible]] void Miss_miss(
// BOUND-DAG: visible]] void Callable_callable(

// DEAD-NOT: updatePayload
// DEAD-NOT: readRecord
// DEAD-NOT: updateData
// DEAD: kernel]] void main{{[_0-9]*}}(
// DEAD-NOT: updatePayload
// DEAD-NOT: readRecord
// DEAD-NOT: updateData
