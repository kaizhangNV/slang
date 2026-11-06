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

// Each projection expands the same entry pack. Specialization preserves its order and length,
// including the empty pack. Check that invariant once before reading corresponding elements.
static IRTypePack* _getProjectedTypes(IRInst* projection, IRTypePack* entryTypes)
{
    auto types = cast<IRTypePack>(projection);
    SLANG_RELEASE_ASSERT(types->getOperandCount() == entryTypes->getOperandCount());
    return types;
}

// The list declaration owns these projections even though its runtime struct has no fields.
// By the time this reader runs, normal specialization has resolved each expansion to a type pack.
static IRPortableRTShaderListDecoration* _getList(IRType* type)
{
    auto list = type->findDecoration<IRPortableRTShaderListDecoration>();
    SLANG_RELEASE_ASSERT(list);
    return list;
}

// A hit-group slot names either a shader type or the library's checked No* placeholder. The
// selected type is enough to decide whether a function-table slot is needed; no invoke is read.
static SchemaStage _readStageSelection(IRInst* type, Stage stage)
{
    SchemaStage result;
    result.stage = stage;
    result.isPresent = !_isPlaceholder(cast<IRType>(type), stage);
    return result;
}

// Miss and callable sections need record layouts and ordered slot counts. Their implementations
// and payload accessors are compiled independently, so reading these lists never selects a body.
static void _readRecordList(IRType* type, Stage stage, List<SchemaStage>& result)
{
    auto list = _getList(type);
    auto entryTypes = cast<IRTypePack>(list->getTypepack());
    auto recordTypes = _getProjectedTypes(list->getRecordTypes(), entryTypes);
    for (UInt i = 0; i < entryTypes->getOperandCount(); ++i)
    {
        SchemaStage entry;
        entry.stage = stage;
        entry.isPresent = true;
        entry.recordType = cast<IRType>(recordTypes->getOperand(i));
        result.add(entry);
    }
}

void readSchemaContract(IRTraceProgramDescriptorType* descriptorType, Schema& outSchema)
{
    // For example, HitGroupList<GroupA, GroupB> carries one expansion for Primitive, another for
    // Record, and so on. Their lookup instructions request the required conformance entries during
    // linking. Here the operands are already the resulting canonical types, not tables to inspect.
    auto program = descriptorType->findDecoration<IRPortableRTProgramDecoration>();
    SLANG_RELEASE_ASSERT(program);
    Schema result;
    result.accelerationStructure = program->getAccelerationStructureType();
    result.motion = program->getMotionType();

    auto hitList = _getList(program->getHitGroupsType());
    auto entryTypes = cast<IRTypePack>(hitList->getTypepack());
    auto recordTypes = _getProjectedTypes(hitList->getRecordTypes(), entryTypes);
    auto payloadTypes = _getProjectedTypes(hitList->getPayloadTypes(), entryTypes);
    auto primitiveTypes = _getProjectedTypes(hitList->getPrimitiveTypes(), entryTypes);
    auto attributesTypes = _getProjectedTypes(hitList->getAttributesTypes(), entryTypes);
    auto closestHitTypes = _getProjectedTypes(hitList->getClosestHitTypes(), entryTypes);
    auto anyHitTypes = _getProjectedTypes(hitList->getAnyHitTypes(), entryTypes);
    auto intersectionTypes = _getProjectedTypes(hitList->getIntersectionTypes(), entryTypes);
    for (UInt i = 0; i < entryTypes->getOperandCount(); ++i)
    {
        SchemaHitGroup group;
        group.payloadType = cast<IRType>(payloadTypes->getOperand(i));
        group.recordType = cast<IRType>(recordTypes->getOperand(i));
        group.primitiveType = cast<IRType>(primitiveTypes->getOperand(i));
        group.attributesType = cast<IRType>(attributesTypes->getOperand(i));
        group.closestHit = _readStageSelection(closestHitTypes->getOperand(i), Stage::ClosestHit);
        group.anyHit = _readStageSelection(anyHitTypes->getOperand(i), Stage::AnyHit);
        group.intersection =
            _readStageSelection(intersectionTypes->getOperand(i), Stage::Intersection);
        result.hitGroups.add(group);
    }
    _readRecordList(program->getMissShadersType(), Stage::Miss, result.missShaders);
    _readRecordList(program->getCallableShadersType(), Stage::Callable, result.callableShaders);
    outSchema = _Move(result);
}

} // namespace PortableRT
} // namespace Slang
