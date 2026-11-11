//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry Callable -stage callable -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry Callable -stage callable -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=DXIL): -experimental-feature -entry Callable -stage callable -target dxil-asm -profile lib_6_6 -validate-ir
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry Callable -stage callable -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct CallableContext : rt::ICallableContext
{
    typealias TraceContext = ::TraceContext;
    typealias CallableData = uint;
    typealias Record = void;
}

// Keeping the helper in HLSL proves that data access is threaded through a real call boundary.
// SPIR-V legalization may inline that call. Both copies still refer to the incoming ABI data.
[noinline]
void increment(in rt::CallableInput<CallableContext> input)
{
    let copy = input;
    copy.data += 11;
}

struct Callable : rt::ICallableShader
{
    typealias Context = CallableContext;

    static void invoke(in rt::CallableInput<Context> input)
    {
        increment(input);
        input.data += input.dispatchRaysIndex.x;
    }
}

// HLSL-DAG: void increment{{.*}}(inout uint
// HLSL-DAG: DispatchRaysIndex()
// HLSL-DAG: [shader("callable")]void Callable_callable(inout {{[^,)]+}})
// SPIRV: OpEntryPoint CallableKHR
// SPIRV: [[DATA:%[A-Za-z0-9_]+]] = OpVariable {{.*}} IncomingCallableDataKHR
// SPIRV: OpIAdd
// SPIRV: OpStore [[DATA]]
// DXIL: define void @

// Copies of CallableInput must pass the same data pointer through the helper on Metal.
// METAL: void [[INCREMENT:increment_[0-9]+]](uint thread* [[HELPER_DATA:[A-Za-z0-9_]+]])
// METAL: *[[HELPER_DATA]] = *[[HELPER_DATA]] + 11U;
// METAL: void [[INVOKE:Callable_invoke_[0-9]+]](uint thread* [[DATA:[A-Za-z0-9_]+]], uint3
// METAL: [[INCREMENT]]([[DATA]]);
// METAL: *[[DATA]] = *[[DATA]] +
// METAL: visible]] void Callable_callable(void thread* [[ENTRY_DATA:[A-Za-z0-9_]+]], uint3 [[INDEX:[A-Za-z0-9_]+]],
// METAL: [[INVOKE]]((uint thread*)((ulong)([[ENTRY_DATA]])), [[INDEX]]);
