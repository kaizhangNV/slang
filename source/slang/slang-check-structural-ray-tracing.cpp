#include "slang-check-impl.h"
#include "slang-lookup.h"
#include "slang-session.h"
#include "slang-syntax.h"

// The structural ray-tracing API uses ordinary Slang interfaces as a source-facing vocabulary,
// but several of its types describe compiler structure rather than runtime data. This file adds the
// semantic rules that cannot be expressed by interface constraints alone:
//
// * a struct's conformance to a canonical stage interface identifies its `invoke` implementation;
// * a source module cannot mix the structural pipeline model with legacy stage entry points or
//   top-level `TraceRay` intrinsics;
// * a stage-input view is valid only as a read-only, by-value parameter of compatible stage code,
//   while stage implementations and layout metadata never become runtime values; and
// * an entry-point request may name a stage struct, with the requested native stage selecting its
//   `invoke` witness.
//
// All classification begins with declaration pointers from `StructuralRayTracingDeclRegistry`.
// Names and structural similarity are deliberately insufficient, so user-defined lookalike types
// retain ordinary language semantics.

namespace Slang
{

// ## Shared stage and witness queries

static Stage _getNativeStage(StructuralRayTracingStageKind kind);
static StructuralRayTracingStageKind _getStructuralStage(Stage stage);
static StructuralRayTracingStageKind _getDirectStageInputKind(
    const StructuralRayTracingDeclRegistry& registry,
    Type* type);

// Return the concrete function that a checked conformance uses for the canonical stage `invoke`
// requirement. Witness tables are key-to-value maps, so look up the exact requirement declaration
// rather than depending on requirement order. A generic function is stored through its outer
// `GenericDecl`; callers need the inner function declaration that actually owns the body.
static FunctionDeclBase* _getStageImplementation(
    const StructuralRayTracingDeclRegistry& registry,
    StructuralRayTracingStageKind stageKind,
    WitnessTable* witnessTable)
{
    auto invokeRequirement = registry.getStageInvokeRequirement(stageKind);
    RequirementWitness invokeWitness;
    if (!invokeRequirement || !witnessTable ||
        !witnessTable->getRequirementDictionary().tryGetValue(invokeRequirement, invokeWitness) ||
        invokeWitness.getFlavor() != RequirementWitness::Flavor::declRef)
    {
        return nullptr;
    }

    Decl* implementation = invokeWitness.getDeclRef().getDecl();
    while (auto genericDecl = as<GenericDecl>(implementation))
        implementation = genericDecl->inner;
    return as<FunctionDeclBase>(implementation);
}

// ## Structural-versus-legacy API validation

// Record a use of one ray-tracing API family and diagnose the first conflict with the other family
// in the same source module. The registry retains the two originating declarations so the
// diagnostic can explain both sides of the conflict.
static void _registerRayTracingAPIUse(
    Linkage* linkage,
    Module* module,
    RayTracingAPIFamily family,
    Decl* decl,
    DiagnosticSink* sink)
{
    auto& registry = linkage->getStructuralRayTracingDeclRegistry();
    if (!registry.isInitialized())
        return;

    Decl* otherDecl = nullptr;
    if (!registry.registerAPIUse(module, family, decl, &otherDecl))
        return;

    auto currentAPI = family == RayTracingAPIFamily::Structural ? "structural" : "legacy";
    auto otherAPI = family == RayTracingAPIFamily::Structural ? "legacy" : "structural";
    sink->diagnose(Diagnostics::MixedRayTracingApis{
        .currentAPI = currentAPI,
        .otherAPI = otherAPI,
        .currentDecl = decl,
        .otherDecl = otherDecl});
}

// Return whether `functionDecl` is one of the legacy, top-level core-module trace intrinsics.
// Inline ray queries and hit-object methods also contain `TraceRay` in their API, but they do not
// select the legacy shader-entry-point model and are therefore intentionally excluded.
static bool _isCoreLegacyTraceMethod(FunctionDeclBase* functionDecl)
{
    if (!functionDecl || !functionDecl->getName())
        return false;

    auto name = functionDecl->getName()->text.getUnownedSlice();
    if (name != "TraceRay" && name != "TraceMotionRay")
        return false;

    // Only the top-level core intrinsics define the legacy pipeline API. Methods such as
    // RayQuery.TraceRayInline and HitObject.TraceRay are separate APIs that may coexist with a
    // structural pipeline.
    for (auto parent = functionDecl->parentDecl; parent; parent = parent->parentDecl)
    {
        if (as<AggTypeDecl>(parent))
            return false;
        if (auto moduleDecl = as<ModuleDecl>(parent))
            return moduleDecl->hasModifier<FromCoreModuleModifier>();
    }
    return false;
}

// Record a checked source call for later reachability analysis and classify calls that select a
// ray-tracing API family. Call edges are recorded before API classification because even an
// ordinary helper call may connect a stage implementation to `RayTracer.callShader` transitively.
void registerRayTracingAPICall(
    Linkage* linkage,
    FunctionDeclBase* caller,
    FunctionDeclBase* callee,
    SourceLoc callLoc,
    DiagnosticSink* sink)
{
    auto& registry = linkage->getStructuralRayTracingDeclRegistry();
    registry.registerFunctionCall(caller, callee, callLoc);
    if (!registry.isInitialized() || !caller || !callee)
        return;

    auto callerModule = getModule(caller);
    if (!callerModule || registry.isTrustedModule(callerModule))
        return;

    if (registry.isTraceMethod(callee) || registry.isCallShaderMethod(callee))
    {
        _registerRayTracingAPIUse(
            linkage,
            callerModule,
            RayTracingAPIFamily::Structural,
            caller,
            sink);
    }
    else if (_isCoreLegacyTraceMethod(callee))
    {
        _registerRayTracingAPIUse(linkage, callerModule, RayTracingAPIFamily::Legacy, caller, sink);
    }
}

// Record compiler-owned semantics when conformance checking encounters a canonical structural
// interface. Any stage or layout-metadata conformance marks the module as structural; stage
// conformances additionally map the exact `invoke` witness back to its native stage.
void SemanticsVisitor::registerStructuralRayTracingStageConformance(
    DeclRef<InterfaceDecl> superInterfaceDeclRef,
    WitnessTable* witnessTable)
{
    auto& registry = getLinkage()->getStructuralRayTracingDeclRegistry();
    auto stageKind = registry.getStageKind(superInterfaceDeclRef.getDecl());
    auto metadataKind = registry.getMetadataKind(superInterfaceDeclRef.getDecl());
    if ((stageKind == StructuralRayTracingStageKind::Count &&
         metadataKind == StructuralRayTracingMetadataKind::Count) ||
        !witnessTable)
        return;

    auto witnessedType = as<DeclRefType>(witnessTable->witnessedType);
    auto witnessedDecl = witnessedType ? witnessedType->getDeclRef().getDecl() : nullptr;
    if (witnessedDecl)
    {
        _registerRayTracingAPIUse(
            getLinkage(),
            getModule(witnessedDecl),
            RayTracingAPIFamily::Structural,
            witnessedDecl,
            getSink());
    }

    if (stageKind == StructuralRayTracingStageKind::Count)
        return;

    registry.registerStageImplementation(
        _getStageImplementation(registry, stageKind, witnessTable),
        stageKind);
}

// Return whether `stage` participates in the legacy ray-tracing pipeline entry-point model.
static bool _isLegacyRayTracingStage(Stage stage)
{
    switch (stage)
    {
    case Stage::ClosestHit:
    case Stage::AnyHit:
    case Stage::Intersection:
    case Stage::Miss:
    case Stage::Callable:
        return true;
    default:
        return false;
    }
}

// Classify a validated ray-tracing entry point as structural when it carries a separately retained
// `invoke` method, and as legacy otherwise. This catches mixed use introduced through explicit
// command-line or API entry-point requests rather than source attributes or calls.
void diagnoseMixedRayTracingAPIUse(EntryPoint* entryPoint, DiagnosticSink* sink)
{
    if (!_isLegacyRayTracingStage(entryPoint->getStage()))
        return;

    auto entryPointDecl = entryPoint->getFuncDecl();
    auto family = entryPoint->getStructuralRayTracingInvokeMethod()
                      ? RayTracingAPIFamily::Structural
                      : RayTracingAPIFamily::Legacy;
    _registerRayTracingAPIUse(
        entryPoint->getLinkage(),
        getModule(entryPointDecl),
        family,
        entryPointDecl,
        sink);
}

// Recursively register legacy ray-tracing entry points declared with source-level `[shader]`
// attributes. These declarations need no entry-point request to establish which API family their
// module uses.
static void _registerAttributedLegacyEntryPoints(
    Linkage* linkage,
    Module* module,
    ContainerDecl* containerDecl,
    DiagnosticSink* sink)
{
    for (auto member : containerDecl->getDirectMemberDecls())
    {
        auto innerMember = member;
        if (auto genericDecl = as<GenericDecl>(innerMember))
            innerMember = genericDecl->inner;

        if (auto functionDecl = as<FuncDecl>(innerMember))
        {
            if (auto entryPointAttr = functionDecl->findModifier<EntryPointAttribute>())
            {
                auto stage =
                    getStageFromAtom(CapabilitySet{entryPointAttr->capabilitySet}.getTargetStage());
                if (_isLegacyRayTracingStage(stage))
                {
                    _registerRayTracingAPIUse(
                        linkage,
                        module,
                        RayTracingAPIFamily::Legacy,
                        functionDecl,
                        sink);
                }
            }
        }

        if (auto childContainer = as<ContainerDecl>(innerMember))
            _registerAttributedLegacyEntryPoints(linkage, module, childContainer, sink);
    }
}

// Diagnose an any-hit or intersection implementation that can transitively call
// `RayTracer.callShader`. The structural stage contracts deliberately permit callable dispatch from
// other stages, so only these two prohibited native stages need this focused reachability check.
static void _diagnoseInvalidCallableDispatchStages(
    StructuralRayTracingDeclRegistry& registry,
    ContainerDecl* containerDecl,
    DiagnosticSink* sink)
{
    for (auto member : containerDecl->getDirectMemberDecls())
    {
        auto innerMember = member;
        if (auto genericDecl = as<GenericDecl>(innerMember))
            innerMember = genericDecl->inner;

        if (auto functionDecl = as<FunctionDeclBase>(innerMember))
        {
            auto stageKind = registry.getStageKind(functionDecl);
            if (stageKind == StructuralRayTracingStageKind::AnyHit ||
                stageKind == StructuralRayTracingStageKind::Intersection)
            {
                SourceLoc callLoc;
                if (registry.findReachableCallShader(functionDecl, callLoc))
                {
                    auto stageName = stageKind == StructuralRayTracingStageKind::AnyHit
                                         ? "any-hit"
                                         : "intersection";
                    sink->diagnose(Diagnostics::StructuralRayTracingCallableStageMismatch{
                        .stage = stageName,
                        .location = callLoc});
                }
            }
        }

        if (auto childContainer = as<ContainerDecl>(innerMember))
            _diagnoseInvalidCallableDispatchStages(registry, childContainer, sink);
    }
}

// Diagnose inferred requirements that cannot execute in the native stage represented by a
// structural `invoke` method. A reachable `callShader` from any-hit or intersection code already
// receives the more precise diagnostic above, so suppress the generic capability diagnostic for
// that same violation.
static void _diagnoseInvalidStructuralStageCapabilities(
    StructuralRayTracingDeclRegistry& registry,
    ContainerDecl* containerDecl,
    DiagnosticSink* sink)
{
    for (auto member : containerDecl->getDirectMemberDecls())
    {
        auto innerMember = member;
        if (auto genericDecl = as<GenericDecl>(innerMember))
            innerMember = genericDecl->inner;

        if (auto functionDecl = as<FunctionDeclBase>(innerMember))
        {
            auto stageKind = registry.getStageKind(functionDecl);
            auto stage = _getNativeStage(stageKind);
            auto capabilities = functionDecl->inferredCapabilityRequirements;
            SourceLoc callShaderLoc;
            auto hasSpecificCallableDiagnostic =
                (stageKind == StructuralRayTracingStageKind::AnyHit ||
                 stageKind == StructuralRayTracingStageKind::Intersection) &&
                registry.findReachableCallShader(functionDecl, callShaderLoc);
            if (!hasSpecificCallableDiagnostic && stage != Stage::Unknown && capabilities &&
                capabilities->isIncompatibleWith(getAtomFromStage(stage)))
            {
                sink->diagnose(Diagnostics::DeclHasDependenciesNotCompatibleOnStage{
                    .stage = getStageName(stage),
                    .decl = functionDecl});
            }
        }

        if (auto childContainer = as<ContainerDecl>(innerMember))
        {
            _diagnoseInvalidStructuralStageCapabilities(registry, childContainer, sink);
        }
    }
}

// Return the stage in which `functionDecl` is required to execute. A canonical `invoke`
// implementation has an intrinsic structural stage. For an ordinary helper, lexical `[require]`
// attributes may provide one uniquely implied stage; no unique stage returns `Count`.
static StructuralRayTracingStageKind _getRequiredStructuralStage(
    StructuralRayTracingDeclRegistry& registry,
    FunctionDeclBase* functionDecl)
{
    auto stageKind = registry.getStageKind(functionDecl);
    if (stageKind != StructuralRayTracingStageKind::Count)
        return stageKind;

    CapabilitySet declaredCapabilities;
    for (auto decl = static_cast<Decl*>(functionDecl); decl; decl = decl->parentDecl)
    {
        for (auto requirement : decl->getModifiersOfType<RequireCapabilityAttribute>())
            declaredCapabilities.unionWith(requirement->capabilitySet);
        if (as<ModuleDecl>(decl))
            break;
    }

    auto stageAtom = declaredCapabilities.getUniquelyImpliedStageAtom();
    if (stageAtom == CapabilityAtom::Invalid)
        return StructuralRayTracingStageKind::Count;
    return _getStructuralStage(getStageFromAtom(stageAtom));
}

// Diagnose direct stage-input parameters that disagree with the function's execution stage.
//
// Consider an unannotated helper whose first parameter is `ClosestHitInput<C>`. That input itself
// restricts the helper to closest-hit code. A later `AnyHitInput<C>` parameter therefore conflicts,
// just as it would if the helper had explicitly declared `[require(closesthit)]`.
static void _diagnoseInvalidStructuralStageInputParameters(
    StructuralRayTracingDeclRegistry& registry,
    ContainerDecl* containerDecl,
    DiagnosticSink* sink)
{
    for (auto member : containerDecl->getDirectMemberDecls())
    {
        auto innerMember = member;
        if (auto genericDecl = as<GenericDecl>(innerMember))
            innerMember = genericDecl->inner;

        if (auto functionDecl = as<FunctionDeclBase>(innerMember))
        {
            auto functionStage = _getRequiredStructuralStage(registry, functionDecl);
            for (auto parameter : functionDecl->getParameters())
            {
                auto inputStage = _getDirectStageInputKind(registry, parameter->type.type);
                if (inputStage == StructuralRayTracingStageKind::Count)
                    continue;
                if (functionStage == StructuralRayTracingStageKind::Count)
                {
                    // A stage-input parameter implicitly restricts an otherwise-unannotated
                    // helper. Additional stage-input parameters must agree with that stage.
                    functionStage = inputStage;
                    continue;
                }
                if (inputStage == functionStage)
                    continue;

                auto location = parameter->type.exp ? parameter->type.exp->loc : parameter->loc;
                sink->diagnose(Diagnostics::StructuralRayTracingInputStageMismatch{
                    .type = parameter->type.type,
                    .stage = getStageName(_getNativeStage(inputStage)),
                    .function = functionDecl,
                    .location = location});
            }
        }

        if (auto childContainer = as<ContainerDecl>(innerMember))
            _diagnoseInvalidStructuralStageInputParameters(registry, childContainer, sink);
    }
}

// Run validations that require all declarations and call edges in `module` to be available. The
// individual expression and conformance hooks populate the registry during checking; this function
// consumes those records once per module.
void diagnoseMixedRayTracingAPIsInModule(Linkage* linkage, Module* module, DiagnosticSink* sink)
{
    auto& registry = linkage->getStructuralRayTracingDeclRegistry();
    if (!registry.isInitialized())
        return;
    _registerAttributedLegacyEntryPoints(linkage, module, module->getModuleDecl(), sink);
    _diagnoseInvalidCallableDispatchStages(registry, module->getModuleDecl(), sink);
    _diagnoseInvalidStructuralStageCapabilities(registry, module->getModuleDecl(), sink);
    _diagnoseInvalidStructuralStageInputParameters(registry, module->getModuleDecl(), sink);
}

// Identify an `invoke` implementation from its enclosing conformance when the implementation body
// is checked before that conformance has been registered globally.
//
// Consider this source:
//
//     struct Hit : rt::IClosestHitShader<Context>
//     {
//         void invoke(rt::ClosestHitInput<Context> input) { invoke(input); }
//     }
//
// While checking the recursive call, `registerStructuralRayTracingStageConformance` may not have
// run yet. The checked inheritance declaration already owns the canonical witness table, so use
// that table's exact `invoke` requirement and cache the result for subsequent calls.
static StructuralRayTracingStageKind _findStageImplementationFromParentConformance(
    StructuralRayTracingDeclRegistry& registry,
    FunctionDeclBase* functionDecl)
{
    Decl* parent = functionDecl->parentDecl;
    while (auto genericDecl = as<GenericDecl>(parent))
        parent = genericDecl->parentDecl;
    auto container = as<ContainerDecl>(parent);
    if (!container)
        return StructuralRayTracingStageKind::Count;

    for (auto inheritanceDecl : container->getDirectMemberDeclsOfType<InheritanceDecl>())
    {
        auto interfaceType = as<DeclRefType>(inheritanceDecl->base.type);
        auto interfaceDeclRef = interfaceType ? interfaceType->getDeclRef().as<InterfaceDecl>()
                                              : DeclRef<InterfaceDecl>();
        auto stageKind = registry.getStageKind(interfaceDeclRef.getDecl());
        if (stageKind == StructuralRayTracingStageKind::Count)
            continue;

        auto implementation =
            _getStageImplementation(registry, stageKind, inheritanceDecl->witnessTable);
        if (implementation == functionDecl)
        {
            registry.registerStageImplementation(functionDecl, stageKind);
            return stageKind;
        }
    }
    return StructuralRayTracingStageKind::Count;
}

// Return the concrete `invoke` implementation reached through a subtype witness. Entry-point
// discovery uses inheritance facets, whose subtype witness may compose declared and transitive
// conformances; `tryLookUpRequirementWitness` is the canonical operation for resolving the exact
// interface requirement through that composition.
static FunctionDeclBase* _getStageImplementationFromSubtypeWitness(
    ASTBuilder* astBuilder,
    const StructuralRayTracingDeclRegistry& registry,
    StructuralRayTracingStageKind stageKind,
    SubtypeWitness* witness)
{
    witness = witness ? as<SubtypeWitness>(witness->resolve()) : nullptr;
    auto invokeRequirement = registry.getStageInvokeRequirement(stageKind);
    if (!invokeRequirement || !witness)
        return nullptr;
    auto invokeWitness = tryLookUpRequirementWitness(astBuilder, witness, invokeRequirement);
    if (invokeWitness.getFlavor() != RequirementWitness::Flavor::declRef)
        return nullptr;
    return as<FunctionDeclBase>(invokeWitness.getDeclRef().getDecl());
}

// ## Struct-named entry-point discovery

// Map the structural stage vocabulary to the compiler's established entry-point stage vocabulary.
static Stage _getNativeStage(StructuralRayTracingStageKind kind)
{
    switch (kind)
    {
    case StructuralRayTracingStageKind::ClosestHit:
        return Stage::ClosestHit;
    case StructuralRayTracingStageKind::AnyHit:
        return Stage::AnyHit;
    case StructuralRayTracingStageKind::Intersection:
        return Stage::Intersection;
    case StructuralRayTracingStageKind::Miss:
        return Stage::Miss;
    case StructuralRayTracingStageKind::Callable:
        return Stage::Callable;
    default:
        return Stage::Unknown;
    }
}

// Map a requested native entry-point stage back to a structural stage, returning `Count` for stages
// that cannot be implemented by a structural ray-tracing stage struct.
static StructuralRayTracingStageKind _getStructuralStage(Stage stage)
{
    switch (stage)
    {
    case Stage::ClosestHit:
        return StructuralRayTracingStageKind::ClosestHit;
    case Stage::AnyHit:
        return StructuralRayTracingStageKind::AnyHit;
    case Stage::Intersection:
        return StructuralRayTracingStageKind::Intersection;
    case Stage::Miss:
        return StructuralRayTracingStageKind::Miss;
    case Stage::Callable:
        return StructuralRayTracingStageKind::Callable;
    default:
        return StructuralRayTracingStageKind::Count;
    }
}

// Create the function-shaped adapter declaration required by the existing `EntryPoint` machinery.
//
// The user names a type such as `ClosestHit` as the entry point, while the rest of the front end
// expects a `FuncDecl`. The adapter therefore inherits the type's source name and location and has
// an intentionally empty, already-checked body. `findStructuralRayTracingEntryPointByName` returns
// the real `invoke` declaration separately, and `EntryPoint` retains it as the semantic source of
// the stage implementation. Later structural entry-point synthesis can derive target parameters
// from that method without treating the empty adapter body as user code.
static FuncDecl* _createStructuralEntryPointDecl(
    Linkage* linkage,
    Module* module,
    AggTypeDecl* stageType)
{
    auto astBuilder = linkage->getASTBuilder();
    auto moduleDecl = module->getModuleDecl();

    auto funcDecl = astBuilder->create<FuncDecl>();
    funcDecl->nameAndLoc = stageType->nameAndLoc;
    funcDecl->loc = stageType->loc;
    funcDecl->closingSourceLoc = stageType->closingSourceLoc;
    funcDecl->parentDecl = moduleDecl;
    funcDecl->returnType.type = astBuilder->getVoidType();
    funcDecl->ownedScope = astBuilder->create<Scope>();
    funcDecl->ownedScope->containerDecl = funcDecl;
    funcDecl->ownedScope->parent = moduleDecl->ownedScope;

    auto body = astBuilder->create<BlockStmt>();
    body->scopeDecl = astBuilder->create<ScopeDecl>();
    body->scopeDecl->ownedScope = astBuilder->create<Scope>();
    body->scopeDecl->ownedScope->parent = funcDecl->ownedScope;
    body->scopeDecl->parentDecl = funcDecl;
    body->body = astBuilder->create<SeqStmt>();
    body->loc = stageType->loc;
    body->closingSourceLoc = stageType->closingSourceLoc;
    funcDecl->body = body;
    funcDecl->setCheckState(DeclCheckState::CapabilityChecked);

    return funcDecl;
}

// Resolve a source-module type named by an entry-point request and select one canonical stage
// conformance. `outFoundStruct` distinguishes "the name is not a local struct" (so ordinary
// function lookup should continue) from "the struct is not a valid requested stage" (already
// diagnosed). On success, the returned adapter uses the struct's name, `outInvokeMethod` receives
// the selected witness, and `ioProfile` is inferred when the struct implements exactly one stage.
//
// Consider this source and request:
//
//     struct Shading : rt::IClosestHitShader<HitContext>, rt::IAnyHitShader<HitContext> { ... }
//     // Command line: -entry Shading -stage closesthit
//
// Inheritance facets provide the canonical subtype witnesses for both interfaces. The requested
// profile selects the closest-hit witness. Omitting `-stage` is diagnosed as ambiguous, whereas a
// struct with only one stage conformance can infer the profile.
DeclRef<FuncDecl> findStructuralRayTracingEntryPointByName(
    Linkage* linkage,
    Module* module,
    Name* name,
    Profile& ioProfile,
    DiagnosticSink* sink,
    bool* outFoundStruct,
    FuncDecl** outInvokeMethod)
{
    *outFoundStruct = false;
    *outInvokeMethod = nullptr;
    auto& registry = linkage->getStructuralRayTracingDeclRegistry();
    if (!registry.isInitialized())
        return DeclRef<FuncDecl>();

    auto expr = module->findDeclFromString(getText(name), sink);
    auto declRefExpr = as<DeclRefExpr>(expr);
    auto stageTypeDeclRef =
        declRefExpr ? declRefExpr->declRef.as<AggTypeDecl>() : DeclRef<AggTypeDecl>();
    // Imported types are not entry points of this translation unit. Restricting discovery to the
    // requested module also matches ordinary function entry-point lookup.
    if (!stageTypeDeclRef || getModule(stageTypeDeclRef.getDecl()) != module)
        return DeclRef<FuncDecl>();

    *outFoundStruct = true;

    // Build a semantic context equivalent to the translation unit's normal checker context so
    // conformance facets include witnesses supplied by imported modules.
    SharedSemanticsContext sharedContext(linkage, module, sink);
    for (auto dependency : module->getModuleDependencies())
    {
        auto moduleDecl = dependency->getModuleDecl();
        if (sharedContext.importedModulesSet.add(moduleDecl))
            sharedContext.importedModulesList.add(moduleDecl);
    }
    SemanticsVisitor visitor(&sharedContext);
    visitor.ensureDecl(stageTypeDeclRef, DeclCheckState::ReadyForConformances);

    FunctionDeclBase* stageImplementations[int(StructuralRayTracingStageKind::Count)] = {};
    auto stageType = DeclRefType::create(linkage->getASTBuilder(), stageTypeDeclRef);
    // Inheritance facets, rather than direct base syntax, are the source of truth because a stage
    // conformance can be inherited or composed through another interface.
    for (auto facet : visitor.getShared()->getInheritanceInfo(stageType).facets)
    {
        auto interfaceDeclRef = facet->origin.declRef.as<InterfaceDecl>();
        auto kind = registry.getStageKind(interfaceDeclRef.getDecl());
        if (kind != StructuralRayTracingStageKind::Count)
        {
            auto implementation = _getStageImplementationFromSubtypeWitness(
                linkage->getASTBuilder(),
                registry,
                kind,
                facet->subtypeWitness);
            if (implementation)
                stageImplementations[int(kind)] = implementation;
        }
    }

    Count implementedStageCount = 0;
    StructuralRayTracingStageKind onlyImplementedStage = StructuralRayTracingStageKind::Count;
    for (int i = 0; i < int(StructuralRayTracingStageKind::Count); ++i)
    {
        if (stageImplementations[i])
        {
            ++implementedStageCount;
            onlyImplementedStage = StructuralRayTracingStageKind(i);
        }
    }

    if (implementedStageCount == 0)
    {
        sink->diagnose(Diagnostics::StructuralRayTracingEntryPointNotStage{
            .stageType = stageTypeDeclRef.getDecl()});
        return DeclRef<FuncDecl>();
    }

    auto requestedStage = ioProfile.getStage();
    auto selectedStage = _getStructuralStage(requestedStage);
    // An explicit stage must name one of the struct's implemented stage contracts. With no stage,
    // inference is sound only when there is exactly one possible implementation.
    if (requestedStage != Stage::Unknown)
    {
        if (selectedStage == StructuralRayTracingStageKind::Count ||
            !stageImplementations[int(selectedStage)])
        {
            sink->diagnose(Diagnostics::StructuralRayTracingEntryPointStageMismatch{
                .stage = getStageName(requestedStage),
                .stageType = stageTypeDeclRef.getDecl()});
            return DeclRef<FuncDecl>();
        }
    }
    else if (implementedStageCount == 1)
    {
        selectedStage = onlyImplementedStage;
        ioProfile = Profile(_getNativeStage(selectedStage));
    }
    else
    {
        sink->diagnose(Diagnostics::StructuralRayTracingEntryPointAmbiguousStage{
            .stageType = stageTypeDeclRef.getDecl()});
        return DeclRef<FuncDecl>();
    }

    bool hasInstanceField = false;
    // Structural stages are invoked from their type and are never constructed as runtime objects.
    // Reject instance state now because there would be no instance from which synthesized entry
    // code could load it; effectively-static declarations remain valid configuration data.
    for (auto field : stageTypeDeclRef.getDecl()->getFields())
    {
        if (!isEffectivelyStatic(field))
        {
            sink->diagnose(Diagnostics::StructuralRayTracingStageInstanceField{.field = field});
            hasInstanceField = true;
        }
    }
    if (hasInstanceField)
        return DeclRef<FuncDecl>();

    auto invokeMethod = as<FuncDecl>(stageImplementations[int(selectedStage)]);
    if (!invokeMethod)
    {
        // Stage interfaces require ordinary methods. Reaching another callable declaration kind
        // means the checked witness-table invariant was violated, rather than a user error.
        sink->diagnose(Diagnostics::InternalCompilerError{.location = stageTypeDeclRef.getLoc()});
        return DeclRef<FuncDecl>();
    }

    *outInvokeMethod = invokeMethod;
    auto funcDecl = _createStructuralEntryPointDecl(linkage, module, stageTypeDeclRef.getDecl());
    return makeDeclRef(funcDecl);
}

// ## Compile-time-only structural types

// Classifies why a type cannot be represented as ordinary shader data. Stage structs name entry
// points, stage inputs are zero-storage views of native built-ins, and metadata interfaces describe
// program layout. `None` means the type has ordinary runtime semantics.
enum class StructuralRayTracingRuntimeTypeKind
{
    None,
    Stage,
    StageInput,
    Metadata,
};

// Return the stage for an exact canonical input-view type after removing source modifiers. This
// deliberately does not classify arbitrary conforming or containing types; those shapes have
// separate rules in `_findStructuralRuntimeType`.
static StructuralRayTracingStageKind _getDirectStageInputKind(
    const StructuralRayTracingDeclRegistry& registry,
    Type* type)
{
    while (auto modifiedType = as<ModifiedType>(type))
        type = modifiedType->getBase();
    auto declRefType = as<DeclRefType>(type);
    auto typeDecl = declRefType ? declRefType->getDeclRef().as<AggTypeDecl>().getDecl() : nullptr;
    return registry.getStageInputKind(typeDecl);
}

// Classify a canonical stage or layout-metadata interface by declaration identity.
static StructuralRayTracingRuntimeTypeKind _getInterfaceRuntimeTypeKind(
    const StructuralRayTracingDeclRegistry& registry,
    InterfaceDecl* interfaceDecl)
{
    if (registry.getStageKind(interfaceDecl) != StructuralRayTracingStageKind::Count)
        return StructuralRayTracingRuntimeTypeKind::Stage;
    if (registry.getMetadataKind(interfaceDecl) != StructuralRayTracingMetadataKind::Count)
        return StructuralRayTracingRuntimeTypeKind::Metadata;
    return StructuralRayTracingRuntimeTypeKind::None;
}

// Classify `type` itself when it is a structural interface or conforms to one. Direct declared
// bases cover declarations whose cached inheritance facets are not ready yet; the facet query then
// covers transitive and composed conformances. Both paths ultimately compare canonical interface
// declaration pointers through the registry.
static StructuralRayTracingRuntimeTypeKind _getDirectStructuralRuntimeTypeKind(
    SemanticsVisitor* visitor,
    const StructuralRayTracingDeclRegistry& registry,
    Type* type)
{
    if (auto declRefType = as<DeclRefType>(type))
    {
        if (auto typeDecl = declRefType->getDeclRef().as<AggTypeDecl>())
        {
            visitor->ensureDecl(typeDecl, DeclCheckState::ReadyForConformances);
            auto kind =
                _getInterfaceRuntimeTypeKind(registry, as<InterfaceDecl>(typeDecl.getDecl()));
            if (kind != StructuralRayTracingRuntimeTypeKind::None)
                return kind;

            // A variable can be checked before inheritance facets have been cached for its concrete
            // type. Inspect checked base declarations as well so classification does not depend on
            // declaration-checking order.
            for (auto inheritanceDecl :
                 typeDecl.getDecl()->getDirectMemberDeclsOfType<InheritanceDecl>())
            {
                visitor->ensureDecl(inheritanceDecl, DeclCheckState::CanUseBaseOfInheritanceDecl);
                auto baseType = as<DeclRefType>(inheritanceDecl->base.type);
                auto baseInterface = baseType ? baseType->getDeclRef().as<InterfaceDecl>()
                                              : DeclRef<InterfaceDecl>();
                kind = _getInterfaceRuntimeTypeKind(registry, baseInterface.getDecl());
                if (kind != StructuralRayTracingRuntimeTypeKind::None)
                    return kind;
            }
        }
    }

    for (auto facet : visitor->getShared()->getInheritanceInfo(type).facets)
    {
        auto interfaceDeclRef = facet->origin.declRef.as<InterfaceDecl>();
        auto kind = _getInterfaceRuntimeTypeKind(registry, interfaceDeclRef.getDecl());
        if (kind != StructuralRayTracingRuntimeTypeKind::None)
            return kind;
    }
    return StructuralRayTracingRuntimeTypeKind::None;
}

// Find the first compile-time-only structural type contained in `type`. The recursion follows every
// shape that can introduce runtime storage: type packs, specialized struct fields, arrays,
// optionals, pointers, and tuples.
//
// Consider `struct Wrapper<T> { T value; }` specialized as
// `Wrapper<ClosestHitInput<Context>>`. The field must be queried through a member declaration
// reference specialized to `Wrapper<...>`; reading the unspecialized field declaration would see
// only `T` and allow the input view to acquire storage. `getMemberDeclRef` and `getType` preserve
// that canonical substitution rather than reconstructing a parallel type shape.
//
// `seenDecls` is the active field-expansion stack. A declaration, rather than a runtime object, is
// the unit that owns fields; stopping when the same declaration is reached prevents ordinary
// recursive structures from causing unbounded semantic checking.
static StructuralRayTracingRuntimeTypeKind _findStructuralRuntimeType(
    SemanticsVisitor* visitor,
    Type* type,
    HashSet<Decl*>& seenDecls)
{
    if (!type || as<ErrorType>(type))
        return StructuralRayTracingRuntimeTypeKind::None;

    while (auto modifiedType = as<ModifiedType>(type))
        type = modifiedType->getBase();

    auto& registry = visitor->getLinkage()->getStructuralRayTracingDeclRegistry();
    if (!registry.isInitialized())
        return StructuralRayTracingRuntimeTypeKind::None;
    if (_getDirectStageInputKind(registry, type) != StructuralRayTracingStageKind::Count)
        return StructuralRayTracingRuntimeTypeKind::StageInput;
    auto directKind = _getDirectStructuralRuntimeTypeKind(visitor, registry, type);
    if (directKind != StructuralRayTracingRuntimeTypeKind::None)
        return directKind;

    if (auto typePack = as<ConcreteTypePack>(type))
    {
        for (Index i = 0; i < typePack->getTypeCount(); ++i)
        {
            auto kind = _findStructuralRuntimeType(visitor, typePack->getElementType(i), seenDecls);
            if (kind != StructuralRayTracingRuntimeTypeKind::None)
                return kind;
        }
    }

    if (auto structType = as<DeclRefType>(type))
    {
        if (auto structDecl = structType->getDeclRef().as<StructDecl>().getDecl())
        {
            if (!seenDecls.add(structDecl))
                return StructuralRayTracingRuntimeTypeKind::None;
            for (auto field : structDecl->getFields())
            {
                visitor->ensureDecl(field, DeclCheckState::CanUseTypeOfValueDecl);
                auto fieldDeclRef = visitor->getASTBuilder()
                                        ->getMemberDeclRef(structType->getDeclRef(), field)
                                        .as<VarDeclBase>();
                SLANG_RELEASE_ASSERT(fieldDeclRef);
                auto fieldType = getType(visitor->getASTBuilder(), fieldDeclRef);
                auto kind = _findStructuralRuntimeType(visitor, fieldType, seenDecls);
                if (kind != StructuralRayTracingRuntimeTypeKind::None)
                    return kind;
            }
            seenDecls.remove(structDecl);
        }
    }

    if (auto arrayType = as<ArrayExpressionType>(type))
        return _findStructuralRuntimeType(visitor, arrayType->getElementType(), seenDecls);
    if (auto optionalType = as<OptionalType>(type))
        return _findStructuralRuntimeType(visitor, optionalType->getValueType(), seenDecls);
    if (auto pointerType = as<PtrTypeBase>(type))
        return _findStructuralRuntimeType(visitor, pointerType->getValueType(), seenDecls);
    if (auto tupleType = as<TupleType>(type))
    {
        for (Index i = 0; i < tupleType->getMemberCount(); ++i)
        {
            auto kind = _findStructuralRuntimeType(visitor, tupleType->getMember(i), seenDecls);
            if (kind != StructuralRayTracingRuntimeTypeKind::None)
                return kind;
        }
    }
    return StructuralRayTracingRuntimeTypeKind::None;
}

// Start a fresh containment search for one independently checked type.
static StructuralRayTracingRuntimeTypeKind _findStructuralRuntimeType(
    SemanticsVisitor* visitor,
    Type* type)
{
    if (!type || as<ErrorType>(type))
        return StructuralRayTracingRuntimeTypeKind::None;
    HashSet<Decl*> seenDecls;
    return _findStructuralRuntimeType(visitor, type, seenDecls);
}

// Emit the diagnostic corresponding to a non-runtime structural category. `None` intentionally
// emits nothing, allowing callers that already have a type to use this as their common final step.
static void _diagnoseInvalidStructuralRayTracingRuntimeType(
    SemanticsVisitor* visitor,
    StructuralRayTracingRuntimeTypeKind kind,
    Type* type,
    SourceLoc location)
{
    if (kind == StructuralRayTracingRuntimeTypeKind::Stage)
    {
        visitor->getSink()->diagnose(
            Diagnostics::StructuralRayTracingStageRuntimeValue{.type = type, .location = location});
    }
    else if (kind == StructuralRayTracingRuntimeTypeKind::StageInput)
    {
        visitor->getSink()->diagnose(
            Diagnostics::StructuralRayTracingInputStorage{.type = type, .location = location});
    }
    else if (kind == StructuralRayTracingRuntimeTypeKind::Metadata)
    {
        visitor->getSink()->diagnose(Diagnostics::StructuralRayTracingMetadataRuntimeValue{
            .type = type,
            .location = location});
    }
}

// Diagnose a variable whose type would store a structural stage, input view, or layout value. The
// only permitted variable form is an exact stage-input view passed as a read-only value parameter;
// it models the compiler-provided input to stage code and therefore has no backing storage. Nested
// views and `out`, `ref`, borrowed, or payload parameters would expose an address or write channel
// and remain invalid.
void SemanticsVisitor::diagnoseInvalidStructuralRayTracingVariableType(VarDeclBase* varDecl)
{
    auto type = varDecl->type.type;
    auto kind = _findStructuralRuntimeType(this, type);
    if (kind == StructuralRayTracingRuntimeTypeKind::None)
        return;

    auto paramDecl = as<ParamDecl>(varDecl);
    auto isReadOnlyValueParameter = paramDecl && !paramDecl->hasModifier<OutModifier>() &&
                                    !paramDecl->hasModifier<RefModifier>() &&
                                    !paramDecl->hasModifier<BorrowModifier>() &&
                                    !paramDecl->hasModifier<HLSLPayloadModifier>();
    if (kind == StructuralRayTracingRuntimeTypeKind::StageInput && isReadOnlyValueParameter &&
        _getDirectStageInputKind(getLinkage()->getStructuralRayTracingDeclRegistry(), type) !=
            StructuralRayTracingStageKind::Count)
    {
        return;
    }

    _diagnoseInvalidStructuralRayTracingRuntimeType(this, kind, type, varDecl->loc);
}

// Diagnose a user-declared callable result that exposes a compile-time-only structural type.
// Constructors have no explicit result channel: their constructed type is checked at the invoke
// expression, which also provides the useful source location and avoids a duplicate diagnostic.
void SemanticsVisitor::diagnoseInvalidStructuralRayTracingCallableResult(CallableDecl* callableDecl)
{
    if (as<ConstructorDecl>(callableDecl))
        return;
    auto type = callableDecl->returnType.type;
    auto kind = _findStructuralRuntimeType(this, type);
    if (kind == StructuralRayTracingRuntimeTypeKind::None)
        return;

    auto location =
        callableDecl->returnType.exp ? callableDecl->returnType.exp->loc : callableDecl->loc;
    _diagnoseInvalidStructuralRayTracingRuntimeType(this, kind, type, location);
}

// Diagnose a property whose value type would expose any compile-time-only structural type. Unlike a
// stage-input parameter, a property is a reusable value-producing API and cannot represent a native
// stage input view safely.
void SemanticsVisitor::diagnoseInvalidStructuralRayTracingPropertyType(PropertyDecl* propertyDecl)
{
    auto type = propertyDecl->type.type;
    _diagnoseInvalidStructuralRayTracingRuntimeType(
        this,
        _findStructuralRuntimeType(this, type),
        type,
        propertyDecl->type.exp ? propertyDecl->type.exp->loc : propertyDecl->loc);
}

// Diagnose construction of any compile-time-only structural stage, stage-input, or layout type.
// Returning true tells ordinary invocation checking that this call shape was recognized and
// already diagnosed.
bool SemanticsVisitor::diagnoseInvalidStructuralRayTracingConstruction(InvokeExpr* invoke)
{
    auto typeType = as<TypeType>(invoke->functionExpr->type);
    if (!typeType)
        return false;
    auto type = typeType->getType();
    if (_findStructuralRuntimeType(this, type) == StructuralRayTracingRuntimeTypeKind::None)
        return false;

    getSink()->diagnose(Diagnostics::StructuralRayTracingTypeConstruction{
        .type = type,
        .location = invoke->functionExpr->loc});
    return true;
}

// Diagnose an invocation whose resolved result reveals a compile-time-only structural type. This
// check is needed even when the generic callable's unspecialized return type was only a type
// parameter; the invocation has the fully substituted result and is the canonical source of truth.
bool SemanticsVisitor::diagnoseInvalidStructuralRayTracingInvokeResult(InvokeExpr* invoke)
{
    auto kind = _findStructuralRuntimeType(this, invoke->type);
    if (kind == StructuralRayTracingRuntimeTypeKind::None)
        return false;

    _diagnoseInvalidStructuralRayTracingRuntimeType(this, kind, invoke->type, invoke->loc);
    return true;
}

// Find the first resolved generic argument or nested argument whose type contains a stage-input
// view. Runtime-storage traversal alone is insufficient here: a generic argument can carry a
// forbidden type into specialization even when it does not appear in a value field. Resolved
// `Type*` identity bounds the search across recursive substitutions, and concrete type packs are
// expanded explicitly.
static Type* _findStructuralStageInputInGenericArgument(
    SemanticsVisitor* visitor,
    Type* type,
    HashSet<Type*>& seenTypes)
{
    type = type ? as<Type>(type->resolve()) : nullptr;
    if (!type || !seenTypes.add(type))
        return nullptr;

    if (_findStructuralRuntimeType(visitor, type) ==
        StructuralRayTracingRuntimeTypeKind::StageInput)
    {
        return type;
    }

    if (auto declRefType = as<DeclRefType>(type))
    {
        Type* result = nullptr;
        SubstitutionSet(declRefType->getDeclRef())
            .forEachGenericSubstitution(
                [&](GenericDecl*, Val::OperandView<Val> arguments)
                {
                    for (auto argument : arguments)
                    {
                        auto argumentType = as<Type>(argument->resolve());
                        if (!result && argumentType)
                        {
                            result = _findStructuralStageInputInGenericArgument(
                                visitor,
                                argumentType,
                                seenTypes);
                        }
                    }
                });
        if (result)
            return result;
    }

    if (auto typePack = as<ConcreteTypePack>(type))
    {
        for (Index i = 0; i < typePack->getTypeCount(); ++i)
        {
            if (auto result = _findStructuralStageInputInGenericArgument(
                    visitor,
                    typePack->getElementType(i),
                    seenTypes))
            {
                return result;
            }
        }
    }
    return nullptr;
}

// Diagnose a call whose substitution arguments contain a structural stage-input view.
//
// Consider `consume<Wrapper<ClosestHitInput<Context>>>()`. The invocation has no stage-input value
// parameter, but its `DeclRef` substitution still specializes user code with the compiler-provided
// view. Walking the resolved substitution arguments closes that escape path; returning true tells
// the caller that invocation checking should stop after this diagnostic.
bool SemanticsVisitor::diagnoseInvalidStructuralRayTracingGenericArguments(InvokeExpr* invoke)
{
    auto functionDeclRef = as<DeclRefExpr>(invoke->functionExpr);
    if (!functionDeclRef)
        return false;

    Type* invalidType = nullptr;
    HashSet<Type*> seenTypes;
    SubstitutionSet(functionDeclRef->declRef)
        .forEachSubstitutionArg(
            [&](Val* argument)
            {
                if (invalidType)
                    return;
                auto type = as<Type>(argument->resolve());
                if (type)
                    invalidType = _findStructuralStageInputInGenericArgument(this, type, seenTypes);
            });
    if (!invalidType)
        return false;

    _diagnoseInvalidStructuralRayTracingRuntimeType(
        this,
        StructuralRayTracingRuntimeTypeKind::StageInput,
        invalidType,
        invoke->functionExpr->loc);
    return true;
}

// Diagnose an ordinary source call to a compiler-managed stage `invoke` implementation. The
// compiler calls this method only through a selected structural entry point; direct calls would
// bypass native stage scheduling and make its built-in input view unavailable. If global
// conformance registration has not run yet, inspect the enclosing checked witness table so the rule
// is independent of declaration-checking order.
bool SemanticsVisitor::diagnoseDirectStructuralRayTracingStageInvoke(
    InvokeExpr* invoke,
    FunctionDeclBase* functionDecl)
{
    auto& registry = getLinkage()->getStructuralRayTracingDeclRegistry();
    auto stageKind = registry.getStageKind(functionDecl);
    if (stageKind == StructuralRayTracingStageKind::Count)
        stageKind = _findStageImplementationFromParentConformance(registry, functionDecl);
    if (stageKind == StructuralRayTracingStageKind::Count)
        return false;

    getSink()->diagnose(
        Diagnostics::DirectStructuralRayTracingStageInvoke{.location = invoke->functionExpr->loc});
    return true;
}

} // namespace Slang
