#include "slang-ir-portable-ray-tracing-schema.h"

#include "slang-ir-util.h"

namespace Slang
{
namespace PortableRT
{

KnownBuiltinDeclName getBuiltinRole(IRInst* value)
{
    auto decoration =
        value ? getResolvedInstForDecorations(value)->findDecoration<IRKnownBuiltinDecoration>()
              : nullptr;
    return decoration ? decoration->getName() : KnownBuiltinDeclName::COUNT;
}

struct Requirement
{
    IRWitnessTable* witness = nullptr;
    IRInst* key = nullptr;
};

// Looks only through declared base-interface edges. For example, IHitContext's Payload belongs
// to its IPayloadContext base, while Primitive's conformance is an associated-type witness and
// is not an inheritance edge. The producer records that distinction on the canonical keys, so
// this reader neither guesses entry positions nor walks arbitrary witness operands.
static Requirement _findRequirement(IRWitnessTable* witness, KnownBuiltinDeclName role)
{
    for (auto entry : witness->getEntries())
    {
        if (getBuiltinRole(entry->getRequirementKey()) == role)
            return {witness, entry->getRequirementKey()};
    }
    for (auto entry : witness->getEntries())
    {
        if (!entry->getRequirementKey()->findDecoration<IRPortableRTBaseInterfaceDecoration>())
            continue;
        auto inheritedWitness = as<IRWitnessTable>(entry->getSatisfyingVal());
        SLANG_RELEASE_ASSERT(inheritedWitness);
        auto requirement = _findRequirement(inheritedWitness, role);
        if (requirement.key)
            return requirement;
    }
    return {};
}

static IRType* _getAssociatedType(IRWitnessTable* witness, KnownBuiltinDeclName role)
{
    auto requirement = _findRequirement(witness, role);
    SLANG_RELEASE_ASSERT(requirement.key);
    auto type = as<IRType>(findWitnessTableEntry(requirement.witness, requirement.key));
    SLANG_RELEASE_ASSERT(type);
    return type;
}

// Retrieves the proof attached to a particular associated-type requirement. Context and
// Context : IHitContext are separate entries; their explicit key relationship survives entry
// reordering, serialization, and a specialized implementation supplied by another module.
static IRWitnessTable* _getAssociatedWitness(IRWitnessTable* witness, KnownBuiltinDeclName role)
{
    auto requirement = _findRequirement(witness, role);
    SLANG_RELEASE_ASSERT(requirement.key);
    for (auto entry : requirement.witness->getEntries())
    {
        auto bound = entry->getRequirementKey()
                         ->findDecoration<IRPortableRTAssociatedTypeWitnessDecoration>();
        if (!bound || bound->getTypeKey() != requirement.key)
            continue;
        auto result = as<IRWitnessTable>(entry->getSatisfyingVal());
        SLANG_RELEASE_ASSERT(result);
        return result;
    }
    SLANG_UNEXPECTED("portable ray-tracing associated type has no conformance witness");
}

static bool _isPlaceholder(IRType* stageType, Stage stage)
{
    switch (getBuiltinRole(stageType))
    {
    case KnownBuiltinDeclName::RayTracingNoClosestHit:
        SLANG_RELEASE_ASSERT(stage == Stage::ClosestHit);
        return true;
    case KnownBuiltinDeclName::RayTracingNoAnyHit:
        SLANG_RELEASE_ASSERT(stage == Stage::AnyHit);
        return true;
    case KnownBuiltinDeclName::RayTracingNoIntersection:
        SLANG_RELEASE_ASSERT(stage == Stage::Intersection);
        return true;
    default:
        return false;
    }
}

// A stage's Context is an associated-type requirement, independent of its invoke signature.
// For example, IMissShader.Context fixes Payload and Record even when another compilation
// supplies Miss.invoke. Read those canonical witnesses before asking for any executable code.
static SchemaStage _readStageContract(IRType* type, IRWitnessTable* witness, Stage stage)
{
    SchemaStage result;
    result.stage = stage;
    result.type = type;
    result.witness = witness;
    result.contextType = _getAssociatedType(witness, KnownBuiltinDeclName::RayTracingStageContext);
    result.contextWitness =
        _getAssociatedWitness(witness, KnownBuiltinDeclName::RayTracingStageContext);
    auto payloadRole = stage == Stage::Callable
                           ? KnownBuiltinDeclName::RayTracingContextCallableData
                           : KnownBuiltinDeclName::RayTracingContextPayload;
    result.payloadType = _getAssociatedType(result.contextWitness, payloadRole);
    result.recordType =
        _getAssociatedType(result.contextWitness, KnownBuiltinDeclName::RayTracingContextRecord);
    if (stage == Stage::ClosestHit || stage == Stage::AnyHit || stage == Stage::Intersection)
    {
        result.primitiveType = _getAssociatedType(
            result.contextWitness,
            KnownBuiltinDeclName::RayTracingContextPrimitive);
        result.attributesType = _getAssociatedType(
            _getAssociatedWitness(
                result.contextWitness,
                KnownBuiltinDeclName::RayTracingContextPrimitive),
            KnownBuiltinDeclName::RayTracingPrimitiveAttributes);
    }
    result.isPresent = !_isPlaceholder(type, stage);
    return result;
}

IRRaytracingAccelerationStructureType* getStageAccelerationStructure(
    IRPortableRTStageInputDecoration* input)
{
    auto traceWitness = _getAssociatedWitness(
        cast<IRWitnessTable>(input->getContextWitness()),
        KnownBuiltinDeclName::RayTracingContextTraceContext);
    return cast<IRRaytracingAccelerationStructureType>(_getAssociatedType(
        traceWitness,
        KnownBuiltinDeclName::RayTracingTraceAccelerationStructure));
}

static SchemaStage _readGroupStageContract(
    IRWitnessTable* groupWitness,
    KnownBuiltinDeclName role,
    Stage stage)
{
    return _readStageContract(
        _getAssociatedType(groupWitness, role),
        _getAssociatedWitness(groupWitness, role),
        stage);
}

struct ShaderList
{
    IRTypePack* types = nullptr;
    IRMakeWitnessPack* witnesses = nullptr;
};

static ShaderList _readList(IRType* type)
{
    auto list = type->findDecoration<IRPortableRTShaderListDecoration>();
    SLANG_RELEASE_ASSERT(list);
    ShaderList result;
    result.types = as<IRTypePack>(list->getEntryTypes());
    result.witnesses = as<IRMakeWitnessPack>(list->getEntryWitnesses());
    SLANG_RELEASE_ASSERT(result.types && result.witnesses);
    SLANG_RELEASE_ASSERT(result.types->getOperandCount() == result.witnesses->getOperandCount());
    return result;
}

bool readSchemaContract(IRTraceProgramDescriptorType* descriptorType, Schema& outSchema)
{
    auto witness = as<IRWitnessTable>(descriptorType->getSchemaWitness());
    if (!witness)
        return false;

    Schema result;
    result.type = descriptorType->getSchema();
    result.traceContext =
        _getAssociatedType(witness, KnownBuiltinDeclName::RayTracingProgramTraceContext);
    auto traceWitness =
        _getAssociatedWitness(witness, KnownBuiltinDeclName::RayTracingProgramTraceContext);
    result.accelerationStructure = _getAssociatedType(
        traceWitness,
        KnownBuiltinDeclName::RayTracingTraceAccelerationStructure);
    result.motion = _getAssociatedType(traceWitness, KnownBuiltinDeclName::RayTracingTraceMotion);
    auto hitList =
        _readList(_getAssociatedType(witness, KnownBuiltinDeclName::RayTracingProgramHitGroups));
    for (UInt i = 0; i < hitList.types->getOperandCount(); ++i)
    {
        SchemaHitGroup group;
        group.type = cast<IRType>(hitList.types->getOperand(i));
        auto groupWitness = cast<IRWitnessTable>(hitList.witnesses->getOperand(i));
        group.contextType =
            _getAssociatedType(groupWitness, KnownBuiltinDeclName::RayTracingHitGroupContext);
        auto contextWitness =
            _getAssociatedWitness(groupWitness, KnownBuiltinDeclName::RayTracingHitGroupContext);
        group.payloadType =
            _getAssociatedType(contextWitness, KnownBuiltinDeclName::RayTracingContextPayload);
        group.recordType =
            _getAssociatedType(contextWitness, KnownBuiltinDeclName::RayTracingContextRecord);
        group.primitiveType =
            _getAssociatedType(contextWitness, KnownBuiltinDeclName::RayTracingContextPrimitive);
        group.attributesType = _getAssociatedType(
            _getAssociatedWitness(contextWitness, KnownBuiltinDeclName::RayTracingContextPrimitive),
            KnownBuiltinDeclName::RayTracingPrimitiveAttributes);
        group.closestHit = _readGroupStageContract(
            groupWitness,
            KnownBuiltinDeclName::RayTracingHitGroupClosestHit,
            Stage::ClosestHit);
        group.anyHit = _readGroupStageContract(
            groupWitness,
            KnownBuiltinDeclName::RayTracingHitGroupAnyHit,
            Stage::AnyHit);
        group.intersection = _readGroupStageContract(
            groupWitness,
            KnownBuiltinDeclName::RayTracingHitGroupIntersection,
            Stage::Intersection);
        result.hitGroups.add(group);
    }
    auto missList =
        _readList(_getAssociatedType(witness, KnownBuiltinDeclName::RayTracingProgramMissShaders));
    for (UInt i = 0; i < missList.types->getOperandCount(); ++i)
    {
        result.missShaders.add(_readStageContract(
            cast<IRType>(missList.types->getOperand(i)),
            cast<IRWitnessTable>(missList.witnesses->getOperand(i)),
            Stage::Miss));
    }
    auto callableList = _readList(
        _getAssociatedType(witness, KnownBuiltinDeclName::RayTracingProgramCallableShaders));
    for (UInt i = 0; i < callableList.types->getOperandCount(); ++i)
    {
        result.callableShaders.add(_readStageContract(
            cast<IRType>(callableList.types->getOperand(i)),
            cast<IRWitnessTable>(callableList.witnesses->getOperand(i)),
            Stage::Callable));
    }
    outSchema = _Move(result);
    return true;
}

} // namespace PortableRT
} // namespace Slang
