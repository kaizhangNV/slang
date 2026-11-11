//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry FirstMiss -stage miss -entry SharedMiss -stage miss -entry OtherMiss -stage miss -entry UnusedMiss -stage miss -entry Callable -stage callable -target hlsl -whole-program -validate-ir
//TEST:SIMPLE(filecheck=IR): -experimental-feature -entry FirstMiss -stage miss -entry SharedMiss -stage miss -entry OtherMiss -stage miss -entry UnusedMiss -stage miss -entry Callable -stage callable -target hlsl -whole-program -validate-ir -dump-ir-after lowerPortableRayTracingEntryPoints
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry FirstMiss -stage miss -entry SharedMiss -stage miss -entry OtherMiss -stage miss -entry UnusedMiss -stage miss -entry Callable -stage callable -entry LibraryMiss -stage miss -entry LibraryCallable -stage callable -target metal -profile metallib_2_4 -whole-program -validate-ir
//DIAGNOSTIC_TEST:SIMPLE(filecheck=UNBOUND): -experimental-feature -entry FirstMiss -stage miss -entry SharedMiss -stage miss -entry OtherMiss -stage miss -entry UnusedMiss -stage miss -entry Callable -stage callable -target metal -profile metallib_2_4 -whole-program -validate-ir

import slang.raytracing.portable;

struct SharedPayload { uint sharedValue; }
struct OtherPayload { float otherValue; }
struct UnusedPayload { uint unusedValue; }
struct LibraryPayload { uint libraryValue; }

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct MissContext<T> : rt::IPayloadContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = T;
    typealias Record = void;
}

struct CallableContext<T> : rt::ICallableContext
{
    typealias TraceContext = ::TraceContext;
    typealias CallableData = T;
    typealias Record = void;
}

// Both selected entries must forward their own incoming payload to the same helper.
struct SharedMiss : rt::IMissShader
{
    typealias Context = MissContext<SharedPayload>;

    [noinline]
    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload.sharedValue += 5;
    }
}

struct FirstMiss : rt::IMissShader
{
    typealias Context = MissContext<SharedPayload>;

    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload.sharedValue += 2;
        SharedMiss::invoke(input);
    }
}

struct OtherMiss : rt::IMissShader
{
    typealias Context = MissContext<OtherPayload>;

    static void invoke(in rt::MissInput<Context> input)
    {
        input.payload.otherValue += 3.0;
    }
}

// This type has no getter anywhere, but its selected entry still needs the payload ABI.
struct UnusedMiss : rt::IMissShader
{
    typealias Context = MissContext<UnusedPayload>;
    static void invoke(in rt::MissInput<Context> input) {}
}

// Callable data and ray payloads deliberately share a type, but have distinct native ABIs.
struct Callable : rt::ICallableShader
{
    typealias Context = CallableContext<SharedPayload>;

    static void invoke(in rt::CallableInput<Context> input)
    {
        input.data.sharedValue += 11;
    }
}

// The native runs leave LibraryMiss and LibraryCallable unselected. Their exported helpers
// test getter-only type discovery: no selected native stage names LibraryPayload.
export void updateLibraryPayload(in rt::MissInput<MissContext<LibraryPayload>> input)
{
    input.payload.libraryValue += 23;
}

export void updateLibraryData(in rt::CallableInput<CallableContext<LibraryPayload>> input)
{
    input.data.libraryValue += 29;
}

// Metal needs a selected stage to supply invocation state to each exported helper. The
// successful Metal run selects these stages; the diagnostic run leaves them unselected.
struct LibraryMiss : rt::IMissShader
{
    typealias Context = MissContext<LibraryPayload>;
    static void invoke(in rt::MissInput<Context> input) { updateLibraryPayload(input); }
}

struct LibraryCallable : rt::ICallableShader
{
    typealias Context = CallableContext<LibraryPayload>;
    static void invoke(in rt::CallableInput<Context> input) { updateLibraryData(input); }
}

// HLSL: void [[SHARED:SharedMiss_invoke[a-zA-Z0-9_]*]](inout [[SHARED_TYPE:SharedPayload[a-zA-Z0-9_]*]] [[SHARED_PARAM:[a-zA-Z0-9_]+]])
// HLSL: [[SHARED_PARAM]].{{sharedValue[a-zA-Z0-9_]*}} = [[SHARED_PARAM]].{{sharedValue[a-zA-Z0-9_]*}} + 5U;
// HLSL: void SharedMiss_miss(inout [[SHARED_TYPE]] [[SHARED_ENTRY_PARAM:[a-zA-Z0-9_]+]] : SV_RayPayload)
// HLSL: [[SHARED]]([[SHARED_ENTRY_PARAM]]);
// HLSL: void [[FIRST:FirstMiss_invoke[a-zA-Z0-9_]*]](inout [[SHARED_TYPE]] [[FIRST_PARAM:[a-zA-Z0-9_]+]])
// HLSL: [[FIRST_PARAM]].{{sharedValue[a-zA-Z0-9_]*}} = [[FIRST_PARAM]].{{sharedValue[a-zA-Z0-9_]*}} + 2U;
// HLSL: [[SHARED]]([[FIRST_PARAM]]);
// HLSL: void FirstMiss_miss(inout [[SHARED_TYPE]] [[FIRST_ENTRY_PARAM:[a-zA-Z0-9_]+]] : SV_RayPayload)
// HLSL: [[FIRST]]([[FIRST_ENTRY_PARAM]]);
// HLSL: void OtherMiss_miss(inout OtherPayload{{[a-zA-Z0-9_]*}} {{[a-zA-Z0-9_]+}} : SV_RayPayload)
// HLSL: void UnusedMiss_miss(inout UnusedPayload{{[a-zA-Z0-9_]*}} {{[a-zA-Z0-9_]+}} : SV_RayPayload)
// HLSL: void [[CALLABLE:Callable_invoke[a-zA-Z0-9_]*]](inout [[SHARED_TYPE]] [[DATA:[a-zA-Z0-9_]+]])
// HLSL: [[DATA]].{{sharedValue[a-zA-Z0-9_]*}} = [[DATA]].{{sharedValue[a-zA-Z0-9_]*}} + 11U;
// HLSL: void Callable_callable(inout [[SHARED_TYPE]] [[ENTRY_DATA:[a-zA-Z0-9_]+]])
// HLSL: [[CALLABLE]]([[ENTRY_DATA]]);
// HLSL: void updateLibraryPayload{{[a-zA-Z0-9_]*}}(inout [[LIBRARY_TYPE:LibraryPayload[a-zA-Z0-9_]*]] [[LIBRARY_PAYLOAD:[a-zA-Z0-9_]+]])
// HLSL: [[LIBRARY_PAYLOAD]].{{libraryValue[a-zA-Z0-9_]*}} = [[LIBRARY_PAYLOAD]].{{libraryValue[a-zA-Z0-9_]*}} + 23U;
// HLSL: void updateLibraryData{{[a-zA-Z0-9_]*}}(inout [[LIBRARY_TYPE]] [[LIBRARY_DATA:[a-zA-Z0-9_]+]])
// HLSL: [[LIBRARY_DATA]].{{libraryValue[a-zA-Z0-9_]*}} = [[LIBRARY_DATA]].{{libraryValue[a-zA-Z0-9_]*}} + 29U;

// IR: ### AFTER lowerPortableRayTracingEntryPoints:
// IR-NOT: portableRTGetPayload
// IR-NOT: portableRTGetCallableData
// IR: ###

// Metal threads the selected entries' concrete payload/data pointers through ordinary calls.
// METAL: void [[SHARED:SharedMiss_invoke_[0-9]+]]([[SHARED_TYPE:SharedPayload_[0-9]+]] thread* [[SP:[A-Za-z0-9_]+]])
// METAL: [[SP]]->{{sharedValue_[0-9]+}} = [[SP]]->{{sharedValue_[0-9]+}} + 5U;
// METAL: void [[FIRST:FirstMiss_invoke_[0-9]+]]([[SHARED_TYPE]] thread* [[FP:[A-Za-z0-9_]+]])
// METAL: [[FP]]->{{sharedValue_[0-9]+}} = [[FP]]->{{sharedValue_[0-9]+}} + 2U;
// METAL: [[SHARED]]([[FP]]);
// METAL: void [[OTHER:OtherMiss_invoke_[0-9]+]]([[OTHER_TYPE:OtherPayload_[0-9]+]] thread* [[OP:[A-Za-z0-9_]+]])
// METAL: [[OP]]->{{otherValue_[0-9]+}} = [[OP]]->{{otherValue_[0-9]+}} + 3.0f;
// METAL: void [[CALLABLE:Callable_invoke_[0-9]+]]([[SHARED_TYPE]] thread* [[CP:[A-Za-z0-9_]+]])
// METAL: [[CP]]->{{sharedValue_[0-9]+}} = [[CP]]->{{sharedValue_[0-9]+}} + 11U;
// METAL: void [[LIB_PAYLOAD:updateLibraryPayload_[0-9]+]]([[LIB_TYPE:LibraryPayload_[0-9]+]] thread* [[LP:[A-Za-z0-9_]+]])
// METAL: [[LP]]->{{libraryValue_[0-9]+}} = [[LP]]->{{libraryValue_[0-9]+}} + 23U;
// METAL: void [[LIB_MISS:LibraryMiss_invoke_[0-9]+]]([[LIB_TYPE]] thread* [[LMP:[A-Za-z0-9_]+]])
// METAL: [[LIB_PAYLOAD]]([[LMP]]);
// METAL: void [[LIB_DATA:updateLibraryData_[0-9]+]]([[LIB_TYPE]] thread* [[LD:[A-Za-z0-9_]+]])
// METAL: [[LD]]->{{libraryValue_[0-9]+}} = [[LD]]->{{libraryValue_[0-9]+}} + 29U;
// METAL: void [[LIB_CALLABLE:LibraryCallable_invoke_[0-9]+]]([[LIB_TYPE]] thread* [[LCD:[A-Za-z0-9_]+]])
// METAL: [[LIB_DATA]]([[LCD]]);
// METAL: visible]] void FirstMiss_miss(void thread* [[FEP:payload_[0-9]+]],
// METAL: [[FIRST]](([[SHARED_TYPE]] thread*)((ulong)([[FEP]])));
// METAL: visible]] void SharedMiss_miss(void thread* [[SEP:payload_[0-9]+]],
// METAL: [[SHARED]](([[SHARED_TYPE]] thread*)((ulong)([[SEP]])));
// METAL: visible]] void OtherMiss_miss(void thread* [[OEP:payload_[0-9]+]],
// METAL: [[OTHER]](([[OTHER_TYPE]] thread*)((ulong)([[OEP]])));
// METAL: visible]] void UnusedMiss_miss(void thread* payload_
// METAL: visible]] void Callable_callable(void thread* [[CD:data_[0-9]+]],
// METAL: [[CALLABLE]](([[SHARED_TYPE]] thread*)((ulong)([[CD]])));
// METAL: visible]] void LibraryMiss_miss(void thread* [[LMEP:payload_[0-9]+]],
// METAL: [[LIB_MISS]](([[LIB_TYPE]] thread*)((ulong)([[LMEP]])));
// METAL: visible]] void LibraryCallable_callable(void thread* [[LCED:data_[0-9]+]],
// METAL: [[LIB_CALLABLE]](([[LIB_TYPE]] thread*)((ulong)([[LCED]])));

// Retaining an exported helper alone does not provide a Metal invocation ABI.
// UNBOUND-NOT: error[E99997]
// UNBOUND: error[E38111]: Metal ray-tracing operation has no stage invocation ABI
// UNBOUND: operation 'portableRTGetPayload' in an exported helper has no invocation state provider
// UNBOUND: error[E38111]: Metal ray-tracing operation has no stage invocation ABI
// UNBOUND: operation 'portableRTGetCallableData' in an exported helper has no invocation state provider
// UNBOUND-NOT: error[E99997]
