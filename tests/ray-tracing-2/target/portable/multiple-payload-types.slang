//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry FirstMiss -stage miss -entry SharedMiss -stage miss -entry OtherMiss -stage miss -entry UnusedMiss -stage miss -entry Callable -stage callable -target hlsl -whole-program -validate-ir
//TEST:SIMPLE(filecheck=IR): -experimental-feature -entry FirstMiss -stage miss -entry SharedMiss -stage miss -entry OtherMiss -stage miss -entry UnusedMiss -stage miss -entry Callable -stage callable -target hlsl -whole-program -validate-ir -dump-ir-after lowerPortableRayTracingEntryPoints

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

// These getters are the only uses of LibraryPayload: no selected stage names that type.
export void updateLibraryPayload(in rt::MissInput<MissContext<LibraryPayload>> input)
{
    input.payload.libraryValue += 23;
}

export void updateLibraryData(in rt::CallableInput<CallableContext<LibraryPayload>> input)
{
    input.data.libraryValue += 29;
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
