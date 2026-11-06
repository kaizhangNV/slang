#pragma once

#include "slang-ir-insts.h"
#include "slang-profile.h"

namespace Slang
{
namespace PortableRT
{

/// The stage selection and record type needed by target synthesis. Selected entry points also
/// supply their callable implementation and input contract; schema lists never select a body.
struct SchemaStage
{
    Stage stage = Stage::Unknown;
    // Miss and callable lists supply this type. Hit-stage records belong to their group.
    IRType* recordType = nullptr;
    bool isPresent = false;

    // Only explicit entry-point selection populates executable information.
    IRFunc* invoke = nullptr;
    IRPortableRTStageInputDecoration* input = nullptr;
};

/// A schema entry fixes stage types, not a physical SBT record. Its array position is the dense
/// function index; the host may put that function index in arbitrarily many runtime records.
struct SchemaHitGroup
{
    // These belong to the group's Context even when all three stages are No* placeholders.
    IRType* payloadType = nullptr;
    IRType* recordType = nullptr;
    IRType* primitiveType = nullptr;
    IRType* attributesType = nullptr;
    SchemaStage closestHit;
    SchemaStage anyHit;
    SchemaStage intersection;
};

/// The concrete program types used by target synthesis after ordinary specialization resolves
/// the descriptor and list decorations. These fields reference the existing canonical IR types.
struct Schema
{
    IRType* accelerationStructure = nullptr;
    IRType* motion = nullptr;
    List<SchemaHitGroup> hitGroups;
    List<SchemaStage> missShaders;
    List<SchemaStage> callableShaders;
};

/// Returns a declaration's checked semantic role, or COUNT when it has no builtin role.
KnownBuiltinDeclName getBuiltinRole(IRInst* value);

/// Reads a specialized descriptor's type contract without resolving stage implementations.
/// The producer records ordinary associated-type projections before linking. This reader runs
/// after specialization, so an unresolved pack or a missing decoration is an IR invariant failure.
void readSchemaContract(IRTraceProgramDescriptorType* descriptorType, Schema& outSchema);

} // namespace PortableRT
} // namespace Slang
