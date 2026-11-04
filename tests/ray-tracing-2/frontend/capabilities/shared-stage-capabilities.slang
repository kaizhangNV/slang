//TEST:SIMPLE: -experimental-feature -entry PortableMiss -target hlsl -no-codegen
//TEST:SIMPLE: -experimental-feature -entry PortableMiss -target metal -no-codegen
//TEST:SIMPLE: -experimental-feature -entry NativeRayOriginMiss -target hlsl -no-codegen
//DIAGNOSTIC_TEST:SIMPLE(diag=NATIVE_ORIGIN,non-exhaustive): -experimental-feature -entry NativeRayOriginMiss -target metal -no-codegen
//TEST:SIMPLE: -experimental-feature -entry NativeCallableMiss -target hlsl -no-codegen
//DIAGNOSTIC_TEST:SIMPLE(diag=NATIVE_CALL,non-exhaustive): -experimental-feature -entry NativeCallableMiss -target metal -no-codegen
//TEST:SIMPLE: -experimental-feature -entry NativeRequirementMiss -target hlsl -no-codegen
//DIAGNOSTIC_TEST:SIMPLE(diag=NATIVE_REQUIREMENT,non-exhaustive): -experimental-feature -entry NativeRequirementMiss -target metal -no-codegen

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct MissContext : rt::IPayloadContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = void;
}

// A portable stage requirement supports both native targets and Metal dispatch.
[require(portable_miss)]
void missOnlyHelper() {}

struct PortableMiss : rt::IMissShader
{
    typealias Context = MissContext;

    static void invoke(rt::MissInput<Context> input)
    {
        missOnlyHelper();
    }
}

// Portable stage requirements must not make native ray-tracing intrinsics available on Metal.
struct NativeRayOriginMiss : rt::IMissShader
{
    typealias Context = MissContext;

    static void invoke(rt::MissInput<Context> input)
/*NATIVE_ORIGIN:
                ^^^^^^ unavailable features in entry point
*/
    {
        float3 origin = WorldRayOrigin();
    }
}

struct NativeCallableMiss : rt::IMissShader
{
    typealias Context = MissContext;

    static void invoke(rt::MissInput<Context> input)
/*NATIVE_CALL:
                ^^^^^^ unavailable features in entry point
*/
    {
        uint payload = 0;
        CallShader(0, payload);
    }
}

// Requiring the native alias inside a portable stage still excludes Metal.
[require(miss)]
void nativeMissOnlyHelper() {}

struct NativeRequirementMiss : rt::IMissShader
{
    typealias Context = MissContext;

    static void invoke(rt::MissInput<Context> input)
/*NATIVE_REQUIREMENT:
                ^^^^^^ unavailable features in entry point
*/
    {
        nativeMissOnlyHelper();
    }
}
