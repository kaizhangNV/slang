//TEST:SIMPLE(filecheck=METAL): -experimental-feature -entry UIntIntersection -stage intersection -entry BoolIntersection -stage intersection -target metal -profile metallib_2_4 -whole-program -validate-ir

import slang.raytracing.portable;

struct UIntAttributes { uint bits; }
struct BoolAttributes { bool flag; }

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct HitContext<A> : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = void;
    typealias Primitive = rt::BoundingBoxPrimitive<A>;
}

struct UIntIntersection : rt::IIntersectionShader
{
    typealias Context = HitContext<UIntAttributes>;
    static void invoke(rt::IntersectionInput<Context> input)
    {
        input.reportHit(2.0, 0, UIntAttributes(0x12345678));
    }
}

struct BoolIntersection : rt::IIntersectionShader
{
    typealias Context = HitContext<BoolAttributes>;
    static void invoke(rt::IntersectionInput<Context> input)
    {
        input.reportHit(1.0, 0, BoolAttributes(true));
    }
}

// These providers may appear in the same traversal. If the uint hit is accepted and the bool
// candidate is rejected by AnyHit, the uint bytes must remain 0x12345678. Unpacking the old bytes
// as bool and repacking them would change them to 1, even though no new hit was committed.
// METAL: intersection(bounding_box,{{.*}} UIntIntersection_intersection(
// METAL-NOT: unpackAnyValue
// METAL: AnyValue4 ray_data* [[UINT_ADDRESS:_S[0-9]+]] =
// METAL-NOT: unpackAnyValue
// METAL: *&rayData_{{[0-9]+}} =
// METAL: if(({{.*}})->accepted)
// METAL-NEXT: {
// METAL: AnyValue4 [[UINT_PACKED:_S[0-9]+]] = packAnyValue4_{{[0-9]+}}(
// METAL: *[[UINT_ADDRESS]] = [[UINT_PACKED]];
// METAL: }
// METAL: return
// METAL: intersection(bounding_box,{{.*}} BoolIntersection_intersection(
// METAL-NOT: unpackAnyValue
// METAL: AnyValue4 ray_data* [[BOOL_ADDRESS:_S[0-9]+]] =
// METAL-NOT: unpackAnyValue
// METAL: *&rayData_{{[0-9]+}} =
// METAL: if(({{.*}})->accepted)
// METAL-NEXT: {
// METAL: AnyValue4 [[BOOL_PACKED:_S[0-9]+]] = packAnyValue4_{{[0-9]+}}(
// METAL: *[[BOOL_ADDRESS]] = [[BOOL_PACKED]];
// METAL: }
// METAL: return
