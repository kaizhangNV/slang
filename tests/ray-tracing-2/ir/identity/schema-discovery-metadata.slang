//TEST:SIMPLE(filecheck=CHECK): -experimental-feature -dump-ir -no-codegen

import slang.raytracing.portable;

// Schema discovery records ordinary associated-type projections before linking. Specialization
// resolves these dependencies without retaining the shader implementations.
// CHECK-DAG: Expand
// CHECK-DAG: PortableRTShaderListDecoration(
// CHECK-DAG: PortableRTStageInputDecoration(8 : Int,

// The compact dumper folds descriptor types into their uses and omits their decorations.
// Static schema tests inspect the program decoration and its concrete projections directly.
// CHECK-DAG: TraceProgramDescriptor(%Schema,

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
