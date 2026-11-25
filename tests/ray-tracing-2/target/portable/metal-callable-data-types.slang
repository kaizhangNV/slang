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

rt::TraceProgramDescriptor<Schema, 1> program;

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
// METAL-DAG: visible_function_table<void(CallableInvocationContext_{{[0-9]+}} thread*)>
// METAL-DAG: thread_position_in_grid
// METAL-DAG: threads_per_grid
// METAL-DAG: forward
// METAL-NOT: visible]]
// METAL-NOT: metalRTCallShader

// UINT: rt_CallableInput_dispatchRaysIndex_get
// UINT: visible]] void UintCallable_callable(CallableInvocationContext_{{[0-9]+}} thread* [[CONTEXT:invocation_[0-9]+]])
// UINT: UintCallable_invoke{{[_0-9]*}}([[CONTEXT]]->dispatchRaysIndex_{{[0-9]+}}, (uint thread*)((ulong)([[CONTEXT]]->data_{{[0-9]+}})));
// FLOAT: *{{.*}} = *{{.*}} * 2.0f;
// The receiver's fixed context retains the other fields even though this body only reads data.
// FLOAT: struct [[CONTEXT_TYPE:CallableInvocationContext_[0-9]+]]
// FLOAT: void thread* data_{{[0-9]+}};
// FLOAT: uint3 dispatchRaysIndex_{{[0-9]+}};
// FLOAT: uint3 dispatchRaysDimensions_{{[0-9]+}};
// FLOAT: uchar device* recordData_{{[0-9]+}};
// FLOAT: uint constant* descriptorResources_{{[0-9]+}};
// FLOAT: const uint device* records_{{[0-9]+}};
// FLOAT: uchar thread* globalResources_{{[0-9]+}};
// FLOAT: visible]] void FloatCallable_callable([[CONTEXT_TYPE]] thread* [[CONTEXT:invocation_[0-9]+]])
// FLOAT: FloatCallable_invoke{{.*}}((float thread*)((ulong)([[CONTEXT]]->data_{{[0-9]+}})));
// FLOAT-NOT: [[CONTEXT]]->
// EMPTY: rt_CallableInput_dispatchRaysDimensions_get
// EMPTY: visible]] void EmptyCallable_callable(CallableInvocationContext_{{[0-9]+}} thread* [[CONTEXT:invocation_[0-9]+]])
// EMPTY: EmptyCallable_invoke{{.*}}([[CONTEXT]]->dispatchRaysDimensions_
// VOID: void preserveData{{.*}}(void thread*
// VOID: + 1U;
// VOID: visible]] void VoidCallable_callable(CallableInvocationContext_{{[0-9]+}} thread* invocation_
// VOID: VoidCallable_invoke{{.*}}(nullptr,
