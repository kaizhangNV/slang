//TEST:SIMPLE(filecheck=CHECK): -experimental-feature -entry LevelOneHit -stage closesthit -target metal -profile metallib_2_3 -validate-ir

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::MultiLevelAccelerationStructure<1>;
    typealias Motion = rt::NoMotion;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = void;
    typealias Primitive = rt::TrianglePrimitive;
}

// Direct primitive-AS traversal has a multilevel source contract with an empty instance path.
// Its independently selected closest-hit adapter must not require Metal 3.1's array_ref type.
struct LevelOneHit : rt::IClosestHitShader
{
    typealias Context = HitContext;

    static void invoke(in rt::ClosestHitInput<Context> input)
    {
        input.payload = input.instanceCount;
        if (input.instanceCount != 0)
            input.payload += input.getInstanceIndex(0) + input.getInstanceID(0);
    }
}

// CHECK-NOT: metal::array_ref
// CHECK: visible]] void LevelOneHit_closesthit(
// CHECK-SAME: uint instanceCount
// CHECK-SAME: uint thread* instanceIndices
// CHECK-SAME: uint thread* instanceIDs
// CHECK-NOT: metal::array_ref
