//TEST:SIMPLE(filecheck=IR): -experimental-feature -entry SharedMiss -stage miss -entry CallingMiss -stage miss -entry nativeCompute -stage compute -target hlsl -whole-program -validate-ir -validate-uniformity -dump-ir-after specializeModule
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry CallingMiss -stage miss -target spirv-assembly -emit-spirv-directly -denorm-mode-fp32 preserve -validate-ir
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

// Apply the requested execution mode to the final native entry after its signature is built.
// SPIRV: OpEntryPoint MissKHR [[ENTRY:%[A-Za-z0-9_]+]]
// SPIRV: OpExecutionMode [[ENTRY]] DenormPreserve 32

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
