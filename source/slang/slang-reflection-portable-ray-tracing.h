#pragma once

#include "core/slang-basic.h"
#include "slang-profile.h"

#include <mutex>

namespace Slang
{
class ProgramLayout;
class Type;
class TypeLayout;

namespace PortableRT
{

/// Describes a source stage independently of whether this compilation selects its entry point.
struct StageReflection : RefObject
{
    Type* type = nullptr;
    Stage stage = Stage::Unknown;
    String defaultEntryPointName;
};

/// Describes the shared application contract of one schema entry. The function index is the
/// position in its source section, not a runtime SBT record index or a resource binding.
struct EntryReflection : RefObject
{
    SlangUInt functionIndex = 0;
    Type* type = nullptr;
    Type* contextType = nullptr;
    Type* recordType = nullptr;
    TypeLayout* recordTypeLayout = nullptr;
};

struct HitGroupReflection : EntryReflection
{
    Type* payloadType = nullptr;
    Type* primitiveType = nullptr;
    Type* attributesType = nullptr;
    RefPtr<StageReflection> closestHit;
    RefPtr<StageReflection> anyHit;
    RefPtr<StageReflection> intersection;
};

struct MissShaderReflection : EntryReflection
{
    Type* payloadType = nullptr;
    RefPtr<StageReflection> miss;
};

struct CallableShaderReflection : EntryReflection
{
    Type* callableDataType = nullptr;
    RefPtr<StageReflection> callable;
};

/// A stable source view of the checked schema. Target-generated functions, physical table sizes,
/// and record-buffer strides belong to compiled-artifact metadata, not this pre-codegen object.
struct SchemaReflection : RefObject
{
    Type* type = nullptr;
    Type* traceContextType = nullptr;
    List<RefPtr<HitGroupReflection>> hitGroups;
    List<RefPtr<MissShaderReflection>> missShaders;
    List<RefPtr<CallableShaderReflection>> callableShaders;
};

/// ProgramLayout owns these views for the lifetime promised by ordinary reflection. Canonical
/// checked Type identities make an alias query return the same object as its underlying schema.
struct SchemaReflectionCache
{
    std::mutex mutex;
    Dictionary<Type*, RefPtr<SchemaReflection>> schemas;
};

SchemaReflection* findSchemaReflection(ProgramLayout* programLayout, const char* name);

} // namespace PortableRT
} // namespace Slang
