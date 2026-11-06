//TEST:SIMPLE(filecheck=CHECK): -experimental-feature -dump-ir -no-codegen

import slang.raytracing.portable;

// Schema discovery records symbolic associated-type lookups before linking. Specialization
// resolves these dependencies without retaining the shader implementations.
// CHECK-DAG: PortableRTShaderListDecoration(
// CHECK-DAG: PortableRTStageInputDecoration(8 : Int,

// Check the imported MissShaderList generic, which has one expansion for Context.Record.
// Follow its witness lookup to the yielded type, then check that the list decoration uses
// this expansion and the original type pack. The six unused operands must be empty packs.
// Static schema tests check the populated hit-group packs after specialization.
// CHECK: [TypeConstraintDecoration(%rtx5FIMissShader)]
// CHECK: param [[TYPES:%[A-Za-z0-9_]+]] : TypeParameterPack,
// CHECK: param [[WITNESSES:%[A-Za-z0-9_]+]] : witness_table_t(%rtx5FIMissShader)):
// CHECK: Expand [[RECORDS:%[A-Za-z0-9_]+]] : Type
// CHECK-NEXT: {
// CHECK-NEXT: block {{%[A-Za-z0-9_]+}}(
// CHECK-NEXT: param [[INDEX:%[A-Za-z0-9_]+]] : Int):
// CHECK: let [[SHADER_WITNESS:%[A-Za-z0-9_]+]] : witness_table_t(%rtx5FIMissShader) = getTupleElement([[WITNESSES]], [[INDEX]])
// CHECK: let [[CONTEXT_WITNESS:%[A-Za-z0-9_]+]] : witness_table_t(%rtx5FIPayloadContext) = lookupWitness([[SHADER_WITNESS]], {{%[A-Za-z0-9_]+}})
// CHECK: let [[STAGE_WITNESS:%[A-Za-z0-9_]+]] : witness_table_t(%rtx5FIStageContext) = lookupWitness([[CONTEXT_WITNESS]], %rtx5FIPayloadContextx5Fx24inheritance)
// CHECK: let [[RECORD:%[A-Za-z0-9_]+]] : Type = lookupWitness([[STAGE_WITNESS]], %rtx5FIStageContextx5FRecord)
// CHECK-NEXT: yield([[RECORD]])
// CHECK-NEXT: }
// CHECK-NEXT: [PortableRTShaderListDecoration([[TYPES]], [[RECORDS]], TypePack, TypePack, TypePack, TypePack, TypePack, TypePack)]
// CHECK-NEXT: [KnownBuiltinDecoration({{[0-9]+}} : Int)]
// CHECK-NEXT: [nameHint("rt.MissShaderList")]
// CHECK-NEXT: struct {{%[A-Za-z0-9_]+}} : Type;

// The compact dumper folds descriptor types into their uses and omits their decorations.
// Static schema tests inspect the program decoration and its concrete types directly.
// CHECK: TraceProgramDescriptor(%Schema,

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
