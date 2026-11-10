//TEST:SIMPLE(filecheck=IR): -experimental-feature -entry SharedMiss -stage miss -entry CallingMiss -stage miss -entry nativeCompute -stage compute -target hlsl -whole-program -validate-ir -validate-uniformity -dump-ir-after specializeModule
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry SharedMiss -stage miss -entry CallingMiss -stage miss -target spirv-assembly -emit-spirv-directly -whole-program -denorm-mode-fp16 preserve -denorm-mode-fp32 ftz -denorm-mode-fp64 preserve -validate-ir
//TEST:SIMPLE(filecheck=NATIVE_MODES): -experimental-feature -entry SharedMiss -stage miss -entry CallingMiss -stage miss -entry nativeCompute -stage compute -target hlsl -whole-program -denorm-mode-fp16 preserve -denorm-mode-fp32 ftz -denorm-mode-fp64 preserve -validate-ir -dump-ir-after lowerPortableRayTracingEntryPoints
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry SharedMiss -stage miss -entry CallingMiss -stage miss -entry nativeCompute -stage compute -target metal -profile metallib_2_4 -whole-program -denorm-mode-fp16 preserve -denorm-mode-fp32 ftz -denorm-mode-fp64 preserve -validate-ir -dump-ir-after lowerPortableRayTracingEntryPoints
//DIAGNOSTIC_TEST:SIMPLE(filecheck=UNIFORMITY): -experimental-feature -entry SharedMiss -stage miss -entry CallingMiss -stage miss -entry nativeCompute -stage compute -target hlsl -whole-program -validate-uniformity -DTEST_UNIFORMITY

// Specialization must still see the original portable entry signatures and ordinary call.
// The native compute entry in the same module must still move its uniform parameter to a
// global resource. Native RT signatures are built after this checkpoint.
// IR: result code = 0
// IR: ### AFTER specializeModule:
// IR: [nameHint("rt.MissInput")]
// IR-NEXT: struct [[INPUT:%[A-Za-z0-9_]+]] : Type;
// IR: [entryPoint({{[0-9]+}} : Int, "SharedMiss_miss",
// IR-NOT: {{^func }}
// IR: [nameHint("SharedMiss.invoke")]
// IR-NOT: {{^func }}
// IR: func [[SHARED:%[A-Za-z0-9_]+]] : Func(Void, [[INPUT]])
// IR: param [[SHARED_INPUT:%[A-Za-z0-9_]+]] : [[INPUT]]):
// IR: portableRTGetRecord([[SHARED_INPUT]])
// IR: [entryPoint({{[0-9]+}} : Int, "CallingMiss_miss",
// IR-NOT: {{^func }}
// IR: [nameHint("CallingMiss.invoke")]
// IR-NOT: {{^func }}
// IR: func {{%[A-Za-z0-9_]+}} : Func(Void, [[INPUT]])
// IR: param [[CALLER_INPUT:%[A-Za-z0-9_]+]] : [[INPUT]]):
// IR-NEXT: call [[SHARED]]([[CALLER_INPUT]])
// IR: [nameHint("EntryPointParams")]
// IR-NEXT: struct [[UNIFORMS:%[A-Za-z0-9_]+]] : Type
// IR: field([[SCALE:%[A-Za-z0-9_]+]], Float)
// IR: [entryPointParam([[NATIVE:%[A-Za-z0-9_]+]])]
// IR: let [[PARAMS:%[A-Za-z0-9_]+]] : ConstantBuffer([[UNIFORMS]], DefaultLayout) = global_param
// IR: [entryPoint({{[0-9]+}} : Int, "nativeCompute",
// IR: func [[NATIVE]] : Func(Void, BorrowInParam(Vec(UInt, 3 : Int), {{.*}}))
// IR: get_field_addr([[PARAMS]], [[SCALE]])
// IR: ###

// All requested execution modes belong to each final native entry, including the entry
// whose implementation is also called directly by CallingMiss.
// SPIRV: OpEntryPoint MissKHR [[SHARED_ENTRY:%[A-Za-z0-9_]+]] "SharedMiss_miss"
// SPIRV: OpEntryPoint MissKHR [[CALLING_ENTRY:%[A-Za-z0-9_]+]] "CallingMiss_miss"
// SPIRV-DAG: OpExecutionMode [[SHARED_ENTRY]] DenormPreserve 16
// SPIRV-DAG: OpExecutionMode [[SHARED_ENTRY]] DenormFlushToZero 32
// SPIRV-DAG: OpExecutionMode [[SHARED_ENTRY]] DenormPreserve 64
// SPIRV-DAG: OpExecutionMode [[CALLING_ENTRY]] DenormPreserve 16
// SPIRV-DAG: OpExecutionMode [[CALLING_ENTRY]] DenormFlushToZero 32
// SPIRV-DAG: OpExecutionMode [[CALLING_ENTRY]] DenormPreserve 64

// Transfer the annotations when constructing native wrappers. Moving them leaves the
// original callable implementations without entry execution modes.
// NATIVE_MODES: ### AFTER lowerPortableRayTracingEntryPoints:
// NATIVE_MODES: [entryPoint({{[0-9]+}} : Int, "SharedMiss_miss",
// NATIVE_MODES-DAG: [fpDenormalPreserve(16 : UInt)]
// NATIVE_MODES-DAG: [fpDenormalFlushToZero(32 : UInt)]
// NATIVE_MODES-DAG: [fpDenormalPreserve(64 : UInt)]
// NATIVE_MODES: func {{%[A-Za-z0-9_]+}}
// NATIVE_MODES: call [[SHARED_IMPL:%[A-Za-z0-9_]+]](
// NATIVE_MODES: }
// NATIVE_MODES-NOT: [fpDenormal
// NATIVE_MODES: [nameHint("SharedMiss.invoke")]
// NATIVE_MODES-NOT: [fpDenormal
// NATIVE_MODES: func [[SHARED_IMPL]]
// NATIVE_MODES: [entryPoint({{[0-9]+}} : Int, "CallingMiss_miss",
// NATIVE_MODES-DAG: [fpDenormalPreserve(16 : UInt)]
// NATIVE_MODES-DAG: [fpDenormalFlushToZero(32 : UInt)]
// NATIVE_MODES-DAG: [fpDenormalPreserve(64 : UInt)]
// NATIVE_MODES: func {{%[A-Za-z0-9_]+}}
// NATIVE_MODES: call [[CALLING_IMPL:%[A-Za-z0-9_]+]](
// NATIVE_MODES: }
// NATIVE_MODES-NOT: [fpDenormal
// NATIVE_MODES: [nameHint("CallingMiss.invoke")]
// NATIVE_MODES-NOT: [fpDenormal
// NATIVE_MODES: func [[CALLING_IMPL]]

// Metal stage exports are visible functions, not native shader entries. Their logical
// implementations and generated exports must have no entry execution modes. The ordinary
// compute entry in this same module must keep all three requested modes.
// METAL: ### AFTER lowerPortableRayTracingEntryPoints:
// METAL-NOT: [fpDenormal
// METAL: [nameHint("SharedMiss.invoke")]
// METAL-NOT: [fpDenormal
// METAL: func [[SHARED_IMPL:%[A-Za-z0-9_]+]]
// METAL-NOT: [fpDenormal
// METAL: [nameHint("CallingMiss.invoke")]
// METAL-NOT: [fpDenormal
// METAL: func [[CALLING_IMPL:%[A-Za-z0-9_]+]]
// METAL-NOT: [fpDenormal
// METAL: [fpDenormalPreserve(64 : UInt)]
// METAL-NEXT: [fpDenormalFlushToZero(32 : UInt)]
// METAL-NEXT: [fpDenormalPreserve(16 : UInt)]
// METAL-NEXT: [entryPoint({{[0-9]+}} : Int, "nativeCompute",
// METAL-NOT: [fpDenormal
// METAL: func {{%[A-Za-z0-9_]+}}
// METAL-NOT: [fpDenormal
// METAL: [metalVisibleFunction(
// METAL-NOT: [fpDenormal
// METAL: [nameHint("SharedMiss_miss")]
// METAL-NOT: [fpDenormal
// METAL: call [[SHARED_IMPL]](
// METAL-NOT: [fpDenormal
// METAL: [metalVisibleFunction(
// METAL-NOT: [fpDenormal
// METAL: [nameHint("CallingMiss_miss")]
// METAL-NOT: [fpDenormal
// METAL: call [[CALLING_IMPL]](
// METAL-NOT: [fpDenormal
// METAL: ###

// A logical input has no varying-parameter layout. Accept it without disabling ordinary
// uniformity validation for the native compute entry in the same module.
// UNIFORMITY: warning[E412
// UNIFORMITY: expectUniform(float(tid.x));
// UNIFORMITY-NOT: warning[E412

import slang.raytracing.portable;

struct Payload { float value; }
struct Record { float scale; }

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct MissContext : rt::IPayloadContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = ::Payload;
    typealias Record = ::Record;
}

RWStructuredBuffer<float> output;

#ifdef TEST_UNIFORMITY
[noinline]
void expectUniform(dynamic_uniform float value)
{
    output[0] = value;
}
#endif

struct SharedMiss : rt::IMissShader
{
    typealias Context = MissContext;

    [noinline]
    static void invoke(rt::MissInput<Context> input)
    {
        input.payload.value *= input.record.scale;
    }
}

struct CallingMiss : rt::IMissShader
{
    typealias Context = MissContext;

    static void invoke(rt::MissInput<Context> input)
    {
        SharedMiss::invoke(input);
        input.payload.value += input.record.scale;
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void nativeCompute(uniform float scale, uint3 tid : SV_DispatchThreadID)
{
    output[tid.x] = scale;
#ifdef TEST_UNIFORMITY
    expectUniform(float(tid.x));
#endif
}
