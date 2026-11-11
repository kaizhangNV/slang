//TEST:SIMPLE(filecheck=HLSL): -experimental-feature -entry GeometryClosestHit -stage closesthit -target hlsl -validate-ir
//TEST:SIMPLE(filecheck=SPIRV): -experimental-feature -entry GeometryClosestHit -stage closesthit -target spirv-assembly -emit-spirv-directly -validate-ir
//TEST:SIMPLE(filecheck=CUDA): -experimental-feature -entry GeometryClosestHit -stage closesthit -target cuda -validate-ir
//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry GeometryClosestHit -stage closesthit -target metal -profile metallib_2_4 -validate-ir

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

// Metal forwards the geometry index supplied to this invocation's visible adapter.
// METAL: uint [[GET:geometryIndex_get_[0-9]+]](uint [[INDEX:[A-Za-z0-9_]+]])
// METAL: return [[INDEX]];
// METAL: *{{[A-Za-z0-9_]+}} = [[GET]](
// METAL: visible]] void GeometryClosestHit_closesthit(
// METAL-SAME: uint [[ENTRY_INDEX:geometryIndex_[0-9]+]],
// METAL: GeometryClosestHit_invoke_{{[0-9]+}}({{.*}}, [[ENTRY_INDEX]]);
