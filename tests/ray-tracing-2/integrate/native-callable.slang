//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry RayGeneration -stage raygeneration -entry AddThirtyOne -stage callable -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry RayGeneration -stage raygeneration -entry AddThirtyOne -stage callable -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -stage raygeneration -entry AddThirtyOne -stage callable -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry RayGeneration -stage raygeneration -entry AddThirtyOne -stage callable -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTX): -experimental-feature -entry RayGeneration -stage raygeneration -entry AddThirtyOne -stage callable -target ptx -validate-ir

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

struct AddThirtyOne : rt::ICallableShader
{
    typealias Context = CallableContext;

    static void invoke(in rt::CallableInput<Context> input)
    {
        let copy = input;
        copy.data += 31;
    }
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::EmptyHitGroupList;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::CallableShaderList<AddThirtyOne>;
}

rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<uint> resultBuffer;

// The matching RHI test supplies a one-entry callable SBT. This deliberately needs no geometry:
// it isolates actual stage dispatch and the caller-visible mutation of callable data. Native
// targets erase program because their SBT is supplied by the pipeline, not a shader binding.
[shader("raygeneration")]
void RayGeneration()
{
    rt::RayTracer<Schema> tracer;
    uint value = 11;
    tracer.callShader<CallableContext>(0, program, value);
    resultBuffer[0] = value;
}

// HLSL-DAG: CallShader(0U,
// HLSL-DAG: [shader("callable")]void AddThirtyOne_callable(
// SPIRV-DAG: OpEntryPoint RayGenerationKHR
// SPIRV-DAG: OpEntryPoint CallableKHR
// SPIRV-DAG: OpExecuteCallableKHR

// METAL-DAG: kernel]] void RayGeneration(
// METAL-DAG: callableFunctions_{{[0-9]+}}[_slang_callable_function_index](&value_
// METAL-DAG: visible]] void AddThirtyOne_callable(
// METAL-DAG: AddThirtyOne_invoke_{{[0-9]+}}((uint thread*)((ulong)(data_

// OptiX calls the selected callable with the caller's data pointer and returns its mutation.
// CUDA: __raygen__RayGeneration()
// CUDA: optixDirectCall<void>(0U, &[[DATA:value_[0-9]+]]);
// CUDA: = [[DATA]];
// CUDA: void [[INVOKE:AddThirtyOne_invoke_[0-9]+]](uint * [[PAYLOAD:data_[0-9]+]])
// CUDA: *[[PAYLOAD]] = *[[PAYLOAD]] + 31U;
// CUDA: __direct_callable__AddThirtyOne_callable(uint * [[ENTRY:data_[0-9]+]])
// CUDA: [[INVOKE]]([[ENTRY]]);
// PTX: .visible .entry __raygen__RayGeneration()
// PTX: _optix_call_direct_callable
// PTX: .visible .func __direct_callable__AddThirtyOne_callable(
// PTX: add.s32 {{.*}}, 31;
