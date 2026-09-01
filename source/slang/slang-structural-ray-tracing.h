#pragma once

#include "compiler-core/slang-source-loc.h"
#include "core/slang-dictionary.h"
#include "slang-compiler-fwd.h"

namespace Slang
{

class InterfaceDecl;
class FunctionDeclBase;
class AggTypeDecl;
class Decl;
class ModuleDecl;

enum class StructuralRayTracingStageKind
{
    ClosestHit,
    AnyHit,
    Intersection,
    Miss,
    Callable,
    Count,
};

enum class StructuralRayTracingMetadataKind
{
    ShaderGroupSlot,
    HitGroup,
    MissGroup,
    CallableGroup,
    HitGroupList,
    MissGroupList,
    CallableGroupList,
    TraceProgramLayout,
    Count,
};

enum class RayTracingAPIFamily
{
    Structural,
    Legacy,
};

struct RayTracingAPIUsage
{
    Decl* structuralDecl = nullptr;
    Decl* legacyDecl = nullptr;
    bool diagnosed = false;
};

class StructuralRayTracingDeclRegistry
{
public:
    bool registerTrustedModule(
        Module* module,
        StructuralRayTracingStageKind* outMissingStage = nullptr);
    bool isInitialized() const { return m_stageInterfaces[0] != nullptr; }
    bool isTrustedModule(Module* module) const;

    InterfaceDecl* getStageInterface(StructuralRayTracingStageKind kind) const;
    StructuralRayTracingStageKind getStageKind(InterfaceDecl* interfaceDecl) const;
    AggTypeDecl* getStageInputType(StructuralRayTracingStageKind kind) const;
    StructuralRayTracingStageKind getStageInputKind(AggTypeDecl* typeDecl) const;
    StructuralRayTracingMetadataKind getMetadataKind(InterfaceDecl* interfaceDecl) const;
    bool isTraceMethod(FunctionDeclBase* functionDecl) const;
    bool isCallShaderMethod(FunctionDeclBase* functionDecl) const;

    FunctionDeclBase* getStageInvokeRequirement(StructuralRayTracingStageKind kind) const;
    void registerStageImplementation(
        FunctionDeclBase* implementation,
        StructuralRayTracingStageKind kind);
    StructuralRayTracingStageKind getStageKind(FunctionDeclBase* implementation) const;
    bool registerAPIUse(
        Module* module,
        RayTracingAPIFamily family,
        Decl* decl,
        Decl** outOtherDecl);
    void registerFunctionCall(
        FunctionDeclBase* caller,
        FunctionDeclBase* callee,
        SourceLoc callLoc);
    bool findReachableCallShader(FunctionDeclBase* function, SourceLoc& outCallLoc) const;

private:
    InterfaceDecl* m_stageInterfaces[int(StructuralRayTracingStageKind::Count)] = {};
    InterfaceDecl* m_intersectionStageInterface = nullptr;
    AggTypeDecl* m_stageInputTypes[int(StructuralRayTracingStageKind::Count)] = {};
    FunctionDeclBase* m_stageInvokeRequirements[int(StructuralRayTracingStageKind::Count)] = {};
    InterfaceDecl* m_metadataInterfaces[int(StructuralRayTracingMetadataKind::Count)] = {};
    ModuleDecl* m_trustedModuleDecl = nullptr;
    AggTypeDecl* m_rayTracerType = nullptr;
    Dictionary<FunctionDeclBase*, StructuralRayTracingStageKind> m_stageImplementations;
    Dictionary<Module*, RayTracingAPIUsage> m_apiUsage;
    Dictionary<FunctionDeclBase*, HashSet<FunctionDeclBase*>> m_functionCallees;
    Dictionary<FunctionDeclBase*, SourceLoc> m_callShaderCallers;
};

const char* getStructuralRayTracingStageInterfaceName(StructuralRayTracingStageKind kind);

} // namespace Slang
