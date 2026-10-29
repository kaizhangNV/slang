//TEST:SIMPLE(filecheck=CHECK): -experimental-feature -dump-ir -no-codegen

import slang.raytracing.portable;

// Schema discovery uses the same associated-type keys and conformance witnesses as ordinary
// interface calls. No executable source call is needed to retain an entry in a shader list.
// CHECK-DAG: PortableRTAssociatedTypeWitnessDecoration(%rtx5FIHitGroupx5FContext)
// CHECK-DAG: PortableRTAssociatedTypeWitnessDecoration(%rtx5FIHitGroupx5FIntersection)
// CHECK-DAG: PortableRTAssociatedTypeWitnessDecoration(%rtx5FITraceProgramSchemax5FTraceContext)
// CHECK-DAG: PortableRTShaderListDecoration(
// CHECK-DAG: PortableRTStageContractDecoration(8 : Int, %rtx5FIIntersectionShaderx5Finvoke)
// CHECK-DAG: PortableRTIntersectionStageDecoration(
// CHECK-DAG: PortableRTStageInputDecoration(8 : Int,

struct Payload { uint value; }
struct Attributes { float distance; }

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct HitContext : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = ::Payload;
    typealias Record = void;
    typealias Primitive = rt::BoundingBoxPrimitive<Attributes>;
}

struct Intersection : rt::IIntersectionShader
{
    typealias Context = HitContext;
    static void invoke(rt::IntersectionInput<Context> input) {}
}

struct HitGroup : rt::IHitGroup
{
    typealias Context = HitContext;
    typealias ClosestHit = rt::NoClosestHit<Context>;
    typealias AnyHit = rt::NoAnyHit<Context>;
    typealias Intersection = ::Intersection;
}

struct Schema : rt::ITraceProgramSchema
{
    typealias TraceContext = ::TraceContext;
    typealias HitGroups = rt::HitGroupList<HitGroup>;
    typealias MissShaders = rt::EmptyMissShaderList;
    typealias CallableShaders = rt::EmptyCallableShaderList;
}

rt::TraceProgramDescriptor<Schema> program;
