//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry GeometryClosestHit -stage closesthit -target hlsl -validate-ir
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry GeometryClosestHit -stage closesthit -target spirv-assembly -emit-spirv-directly -validate-ir
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry GeometryClosestHit -stage closesthit -target cuda -validate-ir

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = void;
    typealias Primitive = rt::TrianglePrimitive;
}

struct GeometryClosestHit : rt::IClosestHitShader
{
    typealias Context = HitContext;

    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload = input.geometryIndex;
    }
}

// The property uses the existing core GeometryIndex contract on all native targets. In OptiX,
// the host uses one SBT record per GAS build input for this index to identify the input geometry.
// HLSL: GeometryIndex()
// SPIRV: BuiltIn RayGeometryIndexKHR
// CUDA: optixGetSbtGASIndex()
