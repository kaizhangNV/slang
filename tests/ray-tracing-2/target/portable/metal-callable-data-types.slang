//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=UINT): -experimental-feature -entry UintCallable -stage callable -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=FLOAT): -experimental-feature -entry FloatCallable -stage callable -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=EMPTY): -experimental-feature -entry EmptyCallable -stage callable -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=VOID): -experimental-feature -entry VoidCallable -stage callable -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct Context<Data> : rt::ICallableContext
{
    typealias TraceContext = ::TraceContext;
    typealias CallableData = Data;
    typealias Record = void;
}

struct EmptyData {}
RWStructuredBuffer<uint> output;

struct UintCallable : rt::ICallableShader
{
    typealias Context = ::Context<uint>;
    static void invoke(in rt::CallableInput<Context> input)
    {
        input.data += input.dispatchRaysIndex.x;
    }
}

struct FloatCallable : rt::ICallableShader
{
    typealias Context = ::Context<float>;
    static void invoke(in rt::CallableInput<Context> input)
    {
        input.data *= 2.0;
    }
}

struct EmptyCallable : rt::ICallableShader
{
    typealias Context = ::Context<EmptyData>;
    static void invoke(in rt::CallableInput<Context> input)
    {
        let copy = input;
        copy.data = EmptyData();
        output[2] = copy.dispatchRaysDimensions.x;
    }
}

// Keep the nominal reference live through specialization. A void callable has no data storage,
// even when a generic helper still receives its data property before empty-type legalization.
[noinline]
void preserveData<Data>(inout Data value)
{
    Data copy = value;
    value = copy;
    output[3] += 1;
}

struct VoidCallable : rt::ICallableShader
{
    typealias Context = ::Context<void>;
    static void invoke(in rt::CallableInput<Context> input)
    {
        preserveData(input.data);
    }
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::EmptyHitGroupList;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders =
        rt::CallableShaderList<UintCallable, FloatCallable, EmptyCallable, VoidCallable>;
}

rt::TraceProgramDescriptor<Schema> program;

interface IDispatch
{
    static void apply(inout uint data);
}

struct Dispatcher : IDispatch
{
    static void apply(inout uint data)
    {
        rt::RayTracer<Schema> tracer;
        tracer.callShader<Context<uint>>(0, program, data);
    }
}

[noinline]
void forward<Dispatcher : IDispatch>(inout uint data)
{
    Dispatcher.apply(data);
}

// A single table signature erases the data pointer; only the selected adapter restores its
// concrete type. Empty data needs a valid ABI pointer but has no source bits to transport.
[shader("raygeneration")]
void RayGeneration()
{
    uint integerData = 1;
    forward<Dispatcher>(integerData);
    float floatData = 3.0;
    EmptyData emptyData = {};
    rt::RayTracer<Schema> tracer;
    tracer.callShader<Context<float>>(1, program, floatData);
    tracer.callShader<Context<EmptyData>>(2, program, emptyData);
    output[0] = integerData;
    output[1] = uint(floatData);
}

// METAL-NOT: visible]]
// METAL-DAG: visible_function_table<void(void thread*, uint3, uint3,
// METAL-DAG: thread_position_in_grid
// METAL-DAG: threads_per_grid
// METAL-DAG: forward
// METAL-NOT: visible]]
// METAL-NOT: metalRTCallShader

// UINT: rt_CallableInput_dispatchRaysIndex_get
// UINT: visible]] void UintCallable_callable(void thread* data
// UINT: UintCallable_invoke{{.*}}((uint thread*){{.*}}data{{.*}}, dispatchRaysIndex
// FLOAT: *{{.*}} = *{{.*}} * 2.0f;
// FLOAT: visible]] void FloatCallable_callable(void thread* data
// FLOAT: FloatCallable_invoke{{.*}}((float thread*){{.*}}data
// EMPTY: rt_CallableInput_dispatchRaysDimensions_get
// EMPTY: visible]] void EmptyCallable_callable(void thread* data
// EMPTY: EmptyCallable_invoke{{.*}}(dispatchRaysDimensions
// VOID: void preserveData{{.*}}(void thread*
// VOID: + 1U;
// VOID: visible]] void VoidCallable_callable(void thread* data
// VOID: VoidCallable_invoke{{.*}}(nullptr,
