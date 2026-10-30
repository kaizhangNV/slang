#pragma once

#include "slang-ir-insts.h"
#include "slang-profile.h"

namespace Slang
{
namespace PortableRT
{

/// A view of one specialized source stage. The canonical IR type/witness/function remain the
/// owners of identity; this transient record only groups the values needed by target synthesis.
struct SchemaStage
{
    Stage stage = Stage::Unknown;
    IRType* type = nullptr;
    IRFunc* invoke = nullptr;
    IRPortableRTStageInputDecoration* input = nullptr;
    bool isPresent = false;
};

/// A schema entry describes code, not a physical SBT record. Its array position is the dense
/// function index; the host may put that function index in arbitrarily many runtime records.
struct SchemaHitGroup
{
    IRType* type = nullptr;
    IRType* contextType = nullptr;
    // These belong to the group's Context even when all three stages are No* placeholders.
    IRType* payloadType = nullptr;
    IRType* recordType = nullptr;
    IRType* primitiveType = nullptr;
    IRType* attributesType = nullptr;
    SchemaStage closestHit;
    SchemaStage anyHit;
    SchemaStage intersection;
};

/// Reads the one canonical descriptor/schema witness into a target-local view after ordinary
/// specialization. No independent IR manifest or alternate identity is introduced.
struct Schema
{
    IRType* type = nullptr;
    IRType* traceContext = nullptr;
    IRType* accelerationStructure = nullptr;
    IRType* motion = nullptr;
    List<SchemaHitGroup> hitGroups;
    List<SchemaStage> missShaders;
    List<SchemaStage> callableShaders;
};

/// Returns a declaration's checked semantic role, or COUNT when it has no builtin role.
KnownBuiltinDeclName getBuiltinRole(IRInst* value);

/// Reads the checked trace context's acceleration-structure type for a standalone stage.
IRRaytracingAccelerationStructureType* getStageAccelerationStructure(
    IRPortableRTStageInputDecoration* input);

/// Reads a fully specialized descriptor schema. Returns false only while its witness is still
/// unresolved; a malformed concrete contract is an IR invariant failure, not an empty schema.
bool readSchema(IRTraceProgramDescriptorType* descriptorType, Schema& outSchema);

} // namespace PortableRT
} // namespace Slang
