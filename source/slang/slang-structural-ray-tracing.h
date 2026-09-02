#pragma once

#include "compiler-core/slang-source-loc.h"
#include "core/slang-dictionary.h"
#include "slang-compiler-fwd.h"

// This file defines the shared vocabulary and per-linkage registry used to recognize structural
// ray-tracing source declarations. The registry is the nominal boundary between ordinary Slang
// interfaces in the packaged standard module and the compiler semantics implemented by AST and IR
// consumers.

namespace Slang
{

class InterfaceDecl;
class FunctionDeclBase;
class AggTypeDecl;
class Decl;
class ModuleDecl;

/// Identifies a shader stage declared through one of the canonical `slang.raytracing` stage
/// interfaces. `Count` is both the number of recognized stages and the sentinel returned for a
/// declaration that has no structural stage identity.
enum class StructuralRayTracingStageKind
{
    ClosestHit,
    AnyHit,
    Intersection,
    Miss,
    Callable,
    Count,
};

/// Identifies a compile-time-only interface that describes the logical shader-binding-table
/// layout. These interfaces are not shader stages, but they must obey the same restriction against
/// becoming runtime values. `Count` is also the unrecognized-interface sentinel.
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

/// Distinguishes the structural pipeline API from the pre-existing entry-point and `TraceRay` API.
/// A source module may use either family, but not both, because the two families require different
/// entry-point discovery and shader-binding-table models.
enum class RayTracingAPIFamily
{
    Structural,
    Legacy,
};

/// Records enough provenance to diagnose the first mixed-API conflict in one source module.
/// Keeping the first declaration from each family lets the diagnostic point to both sides of the
/// conflict; `diagnosed` prevents later uses from producing duplicate diagnostics.
struct RayTracingAPIUsage
{
    Decl* structuralDecl = nullptr;
    Decl* legacyDecl = nullptr;
    bool diagnosed = false;
};

/// Owns the compiler's nominal identities and transient semantic facts for structural ray tracing.
///
/// The structural ray-tracing API starts as ordinary declarations in the separately loaded
/// `slang.raytracing` module. When that packaged module is loaded, `registerTrustedModule` resolves
/// its canonical declarations once. All later recognition compares declaration pointers from this
/// registry instead of names, so a user declaration with a familiar name such as
/// `IClosestHitShader` cannot acquire compiler semantics.
///
/// The registry has `Linkage` lifetime. In addition to the canonical declarations, it records facts
/// discovered at different points in semantic checking: which functions implement stage `invoke`
/// requirements, which ray-tracing API family each source module uses, and the call edges needed to
/// decide whether a stage can reach `RayTracer.callShader`.
class StructuralRayTracingDeclRegistry
{
public:
    /// Resolve the canonical stage interfaces, stage-input types, and `invoke` requirements from
    /// the packaged module. Returns false without publishing partial stage arrays if a required
    /// declaration is absent; `outMissingStage`, when present, identifies the first incomplete
    /// stage contract.
    bool registerTrustedModule(
        Module* module,
        StructuralRayTracingStageKind* outMissingStage = nullptr);

    /// Return whether all canonical stage declarations have been registered successfully.
    bool isInitialized() const { return m_stageInterfaces[0] != nullptr; }

    /// Return whether `module` is the exact packaged module used to initialize this registry.
    bool isTrustedModule(Module* module) const;

    /// Return the canonical public interface for `kind`, or null for `Count` or an invalid value.
    InterfaceDecl* getStageInterface(StructuralRayTracingStageKind kind) const;

    /// Return the structural stage represented by `interfaceDecl`, or `Count` when it is not a
    /// compiler-owned stage interface. The internal intersection-stage interface maps to
    /// `Intersection` as well as the public intersection interface.
    StructuralRayTracingStageKind getStageKind(InterfaceDecl* interfaceDecl) const;

    /// Return the canonical zero-storage input-view type for `kind`, or null for an invalid kind.
    AggTypeDecl* getStageInputType(StructuralRayTracingStageKind kind) const;

    /// Return the stage whose canonical input view is `typeDecl`, or `Count` if none matches.
    StructuralRayTracingStageKind getStageInputKind(AggTypeDecl* typeDecl) const;

    /// Return the compile-time metadata role of `interfaceDecl`, or `Count` if none matches.
    StructuralRayTracingMetadataKind getMetadataKind(InterfaceDecl* interfaceDecl) const;

    /// Return whether this is the canonical `RayTracer.trace` method from the trusted module.
    bool isTraceMethod(FunctionDeclBase* functionDecl) const;

    /// Return whether this is the canonical `RayTracer.callShader` method from the trusted module.
    bool isCallShaderMethod(FunctionDeclBase* functionDecl) const;

    /// Return the canonical `invoke` requirement used to identify implementations of `kind`.
    FunctionDeclBase* getStageInvokeRequirement(StructuralRayTracingStageKind kind) const;

    /// Record that `implementation` satisfies the canonical `invoke` requirement for `kind`.
    void registerStageImplementation(
        FunctionDeclBase* implementation,
        StructuralRayTracingStageKind kind);

    /// Return the stage implemented by `implementation`, or `Count` if it is an ordinary function.
    /// This recognizes both interface requirements and concrete witnesses registered while checking
    /// conformances.
    StructuralRayTracingStageKind getStageKind(FunctionDeclBase* implementation) const;

    /// Record one use of `family` in `module`. Returns true exactly once when the module first
    /// contains both API families, and writes the previously recorded conflicting declaration to
    /// `outOtherDecl`; otherwise returns false and writes null.
    bool registerAPIUse(
        Module* module,
        RayTracingAPIFamily family,
        Decl* decl,
        Decl** outOtherDecl);

    /// Record a checked direct call edge and, for a canonical `callShader` call, its source
    /// location. The graph is intentionally source-level: later capability checks ask whether a
    /// stage implementation can reach that operation through ordinary helper functions.
    void registerFunctionCall(
        FunctionDeclBase* caller,
        FunctionDeclBase* callee,
        SourceLoc callLoc);

    /// Return whether `function` transitively reaches a recorded `RayTracer.callShader` call. On
    /// success, `outCallLoc` receives one such call site for a stage-specific diagnostic.
    bool findReachableCallShader(FunctionDeclBase* function, SourceLoc& outCallLoc) const;

private:
    // These pointers are the nominal source-of-truth used by all AST checks. Names are consulted
    // only while the trusted module is registered.
    InterfaceDecl* m_stageInterfaces[int(StructuralRayTracingStageKind::Count)] = {};
    InterfaceDecl* m_intersectionStageInterface = nullptr;
    AggTypeDecl* m_stageInputTypes[int(StructuralRayTracingStageKind::Count)] = {};
    FunctionDeclBase* m_stageInvokeRequirements[int(StructuralRayTracingStageKind::Count)] = {};
    InterfaceDecl* m_metadataInterfaces[int(StructuralRayTracingMetadataKind::Count)] = {};
    ModuleDecl* m_trustedModuleDecl = nullptr;
    AggTypeDecl* m_rayTracerType = nullptr;

    // These tables accumulate facts produced by semantic checking and consumed by entry-point,
    // API-family, and stage-capability validation later in the same linkage.
    Dictionary<FunctionDeclBase*, StructuralRayTracingStageKind> m_stageImplementations;
    Dictionary<Module*, RayTracingAPIUsage> m_apiUsage;
    Dictionary<FunctionDeclBase*, HashSet<FunctionDeclBase*>> m_functionCallees;
    Dictionary<FunctionDeclBase*, SourceLoc> m_callShaderCallers;
};

/// Return the canonical source spelling of the public stage interface for `kind`, or null for an
/// invalid kind. These names are used only to bootstrap the trusted declaration registry.
const char* getStructuralRayTracingStageInterfaceName(StructuralRayTracingStageKind kind);

} // namespace Slang
