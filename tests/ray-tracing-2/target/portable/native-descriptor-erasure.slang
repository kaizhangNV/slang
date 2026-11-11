//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry main -target hlsl
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry main -target spirv-assembly -emit-spirv-directly
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry main -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry main -target cuda -validate-ir
//TEST:SIMPLE(filecheck=PTX): -experimental-feature -entry main -target ptx -validate-ir -Xnvrtc -I"./external/optix-dev/include/"

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

// Metal materializes the descriptor as a separate resource binding. The ordinary scale
// field still crosses the helper boundary in the constant-buffer portion of Parameters.
// METAL: struct [[PARAMS:Parameters_[0-9]+]]
// METAL-NEXT: {
// METAL-NEXT: float [[SCALE:scale_[0-9]+]];
// METAL-NEXT: };
// METAL: struct [[RESOURCES:PortableRTDescriptorResources_[A-Za-z0-9_]+]]
// METAL: intersection_function_table
// METAL: visible_function_table
// METAL: float [[READ:readScale_[0-9]+]](const [[PARAMS]] constant* [[VALUE:[A-Za-z0-9_]+]])
// METAL: return [[VALUE]]->[[SCALE]];
// METAL: kernel]] void main_{{[0-9]+}}(
// METAL-SAME: [[PARAMS]] constant* [[BOUND_PARAMS:parameters_[0-9]+]]
// METAL-SAME: [[RESOURCES]] constant* parameters_program_
// METAL: [[READ]]([[BOUND_PARAMS]]);

// OptiX also erases the descriptor field before passing Parameters to the helper.
// CUDA: struct [[PARAMS:Parameters_[0-9]+]]
// CUDA-NEXT: {
// CUDA-NEXT: float [[SCALE:scale_[0-9]+]];
// CUDA-NEXT: };
// CUDA: float [[READ:readScale_[0-9]+]]([[PARAMS]] * [[VALUE:[A-Za-z0-9_]+]])
// CUDA: return [[VALUE]]->[[SCALE]];
// CUDA: __raygen__main_{{[0-9]+}}()
// CUDA: [[READ]](
// PTX: .visible .entry __raygen__main_{{[0-9]+}}()
// PTX: call.uni
// PTX: readScale_
// PTX: st.global.f32
