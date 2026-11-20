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

// This entry's portable_miss profile permits Metal, but WorldRayOrigin requires native ray
// tracing on CUDA, GLSL, HLSL, or SPIR-V. Its Metal compilation must therefore fail.
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

// The portable_miss profile permits Metal, but CallShader requires native ray tracing on
// CUDA, GLSL, HLSL, or SPIR-V. Calling it still makes this entry unavailable on Metal.
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

// The native miss alias includes raytracing, which excludes Metal. Calling this helper must
// therefore fail on Metal even though the entry's portable_miss profile permits that target.
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
