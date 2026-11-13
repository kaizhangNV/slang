//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry RayGeneration -stage raygeneration -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=CALLABLE): -experimental-feature -entry Callable -stage callable -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct Record
{
    uint addend;
}

struct CallableContext : rt::ICallableContext
{
    typealias TraceContext = ::TraceContext;
    typealias CallableData = uint;
    typealias Record = ::Record;
}

[noinline]
void update(in rt::CallableInput<CallableContext> input)
{
    let copy = input;
    copy.data += copy.record.addend + copy.dispatchRaysIndex.x + copy.dispatchRaysDimensions.x;
}

struct Callable : rt::ICallableShader
{
    typealias Context = CallableContext;
    static void invoke(in rt::CallableInput<Context> input)
    {
        update(input);
    }
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::EmptyHitGroupList;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::CallableShaderList<Callable>;
}

rt::TraceProgramDescriptor<Schema> program;
RWStructuredBuffer<uint> output;
uniform uint callableRecordIndex;

// A descriptor can be used solely for callable dispatch; no trace or acceleration structure is
// required. The runtime record index selects a record, whose function index selects the VFT arm.
[shader("raygeneration")]
void RayGeneration()
{
    uint data = 1;
    rt::RayTracer<Schema> tracer;
    tracer.callShader<CallableContext>(callableRecordIndex, program, data);
    output[0] = data;
}

// METAL-NOT: visible]]
// METAL-DAG: visible_function_table<void(void thread*, uint3, uint3, uchar device*, uint constant*, uint device*, uchar thread*)>
// METAL-DAG: _slang_callable_record
// METAL-DAG: _slang_callable_function_index
// METAL-NOT: visible]]
// METAL-NOT: metalRTCallShader

// CALLABLE: void [[UPDATE:update[_0-9]*]](uint3 [[UPDATE_INDEX:[A-Za-z0-9_]+]], uint3 [[UPDATE_DIMENSIONS:[A-Za-z0-9_]+]], uint thread* [[UPDATE_DATA:[A-Za-z0-9_]+]],
// CALLABLE: *[[UPDATE_DATA]] = *[[UPDATE_DATA]] + {{.*}}->addend{{.*}}rt_CallableInput_dispatchRaysIndex_get{{[_0-9]*}}([[UPDATE_INDEX]]).x + rt_CallableInput_dispatchRaysDimensions_get{{[_0-9]*}}([[UPDATE_DIMENSIONS]]).x
// CALLABLE: void [[INVOKE:Callable_invoke[_0-9]*]](uint3 [[INVOKE_INDEX:[A-Za-z0-9_]+]], uint3 [[INVOKE_DIMENSIONS:[A-Za-z0-9_]+]], uint thread* [[INVOKE_DATA:[A-Za-z0-9_]+]],
// CALLABLE: [[UPDATE]]([[INVOKE_INDEX]], [[INVOKE_DIMENSIONS]], [[INVOKE_DATA]],
// CALLABLE: visible]] void Callable_callable(void thread* [[ENTRY_DATA:data[_0-9]*]],
// CALLABLE-SAME: uint3 [[ENTRY_INDEX:dispatchRaysIndex[_0-9]*]],
// CALLABLE-SAME: uint3 [[ENTRY_DIMENSIONS:dispatchRaysDimensions[_0-9]*]],
// CALLABLE-SAME: uchar device* recordData
// CALLABLE-SAME: uint constant* descriptorResources
// CALLABLE-SAME: uint device* records
// CALLABLE-SAME: uchar thread* globalResources{{[_0-9]*}})
// CALLABLE: uint thread* [[DATA_PTR:[A-Za-z0-9_]+]] = (uint thread*)((ulong)([[ENTRY_DATA]]));
// CALLABLE: Record{{.*}} device*{{.*}}recordData
// CALLABLE: [[INVOKE]]([[ENTRY_INDEX]], [[ENTRY_DIMENSIONS]], [[DATA_PTR]],
