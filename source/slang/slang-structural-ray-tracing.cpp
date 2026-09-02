#include "slang-structural-ray-tracing.h"

#include "slang-ast-builder.h"
#include "slang-ast-decl.h"
#include "slang-ir-insts.h"
#include "slang-module.h"

// This file establishes the nominal bridge between the separately compiled `slang.raytracing`
// source module and compiler semantics. Declaration names are used only while loading the packaged,
// trusted module. The resulting declaration pointers become the source of truth for every later
// AST query, which prevents an identically named declaration in user code from being mistaken for
// a compiler-owned ray-tracing contract.
//
// The same per-linkage registry also joins facts produced at different points in semantic checking.
// Conformance checking records stage implementations, expression checking records call edges and
// ray-tracing API uses, and module-level validation consumes those records after declarations have
// been checked.

namespace Slang
{

// ## Trusted declaration discovery

bool isTraceProgramDescriptorDecl(AggTypeDecl* typeDecl)
{
    auto magicType = typeDecl ? typeDecl->findModifier<MagicTypeModifier>() : nullptr;
    return magicType &&
           magicType->magicNodeType.getTag() == ASTNodeType::TraceProgramDescriptorType;
}

// Return the public stage-interface name used to bootstrap `kind` from the trusted source module.
const char* getStructuralRayTracingStageInterfaceName(StructuralRayTracingStageKind kind)
{
    switch (kind)
    {
    case StructuralRayTracingStageKind::ClosestHit:
        return "IClosestHitShader";
    case StructuralRayTracingStageKind::AnyHit:
        return "IAnyHitShader";
    case StructuralRayTracingStageKind::Intersection:
        return "IIntersectionShader";
    case StructuralRayTracingStageKind::Miss:
        return "IMissShader";
    case StructuralRayTracingStageKind::Callable:
        return "ICallableShader";
    default:
        return nullptr;
    }
}

// Return the canonical input-view type name associated with `kind`.
static const char* _getStageInputTypeName(StructuralRayTracingStageKind kind)
{
    switch (kind)
    {
    case StructuralRayTracingStageKind::ClosestHit:
        return "ClosestHitInput";
    case StructuralRayTracingStageKind::AnyHit:
        return "AnyHitInput";
    case StructuralRayTracingStageKind::Intersection:
        return "IntersectionInput";
    case StructuralRayTracingStageKind::Miss:
        return "MissInput";
    case StructuralRayTracingStageKind::Callable:
        return "CallableInput";
    default:
        return nullptr;
    }
}

// Return the canonical layout-metadata interface name associated with `kind`.
static const char* _getMetadataInterfaceName(StructuralRayTracingMetadataKind kind)
{
    switch (kind)
    {
    case StructuralRayTracingMetadataKind::ShaderGroupSlot:
        return "IShaderGroupSlot";
    case StructuralRayTracingMetadataKind::HitGroup:
        return "IHitGroup";
    case StructuralRayTracingMetadataKind::MissGroup:
        return "IMissGroup";
    case StructuralRayTracingMetadataKind::CallableGroup:
        return "ICallableGroup";
    case StructuralRayTracingMetadataKind::HitGroupList:
        return "IHitGroupList";
    case StructuralRayTracingMetadataKind::MissGroupList:
        return "IMissGroupList";
    case StructuralRayTracingMetadataKind::CallableGroupList:
        return "ICallableGroupList";
    case StructuralRayTracingMetadataKind::TraceProgramLayout:
        return "ITraceProgramLayout";
    default:
        return nullptr;
    }
}

// Find `declName` below an `rt` namespace in `container`. Generic declarations are represented by
// an outer `GenericDecl`, so compare the name of their inner declaration while continuing recursion
// through the original container. This name-based search is safe only because its caller supplies
// the trusted packaged module; all subsequent recognition uses the returned declaration pointer.
static Decl* _findNamedDeclInContainer(
    ContainerDecl* container,
    Name* rtName,
    Name* declName,
    bool insideRayTracingNamespace)
{
    for (auto decl : container->getDirectMemberDecls())
    {
        bool insideNamespace = insideRayTracingNamespace;
        if (auto namespaceDecl = as<NamespaceDecl>(decl))
            insideNamespace = insideNamespace || namespaceDecl->getName() == rtName;

        auto candidate = decl;
        if (auto genericDecl = as<GenericDecl>(candidate))
            candidate = genericDecl->inner;
        if (insideNamespace && candidate->getName() == declName)
            return candidate;

        if (auto childContainer = as<ContainerDecl>(decl))
        {
            if (auto result =
                    _findNamedDeclInContainer(childContainer, rtName, declName, insideNamespace))
            {
                return result;
            }
        }
    }
    return nullptr;
}

// Find a canonical declaration by source name in the trusted module's `rt` namespace.
static Decl* _findNamedDecl(Module* module, const char* name)
{
    auto namePool = module->getASTBuilder()->getNamePool();
    return _findNamedDeclInContainer(
        module->getModuleDecl(),
        namePool->getName("rt"),
        namePool->getName(name),
        false);
}

// Find the public stage interface that defines the source contract for `kind`.
static InterfaceDecl* _findStageInterface(Module* module, StructuralRayTracingStageKind kind)
{
    return as<InterfaceDecl>(
        _findNamedDecl(module, getStructuralRayTracingStageInterfaceName(kind)));
}

// Find the zero-storage input-view type passed to the `invoke` requirement for `kind`.
static AggTypeDecl* _findStageInputType(Module* module, StructuralRayTracingStageKind kind)
{
    return as<AggTypeDecl>(_findNamedDecl(module, _getStageInputTypeName(kind)));
}

// Find the direct `invoke` requirement declared by `interfaceDecl`. The registry calls this with
// the internal `IIntersectionStage` interface for intersection shaders because the public
// `IIntersectionShader` inherits that common requirement instead of redeclaring it.
static FunctionDeclBase* _findStageInvokeRequirement(InterfaceDecl* interfaceDecl)
{
    for (auto member : interfaceDecl->getDirectMemberDecls())
    {
        auto candidate = member;
        if (auto genericDecl = as<GenericDecl>(candidate))
            candidate = genericDecl->inner;
        if (auto functionDecl = as<FunctionDeclBase>(candidate))
        {
            if (functionDecl->getName() && functionDecl->getName()->text == "invoke")
                return functionDecl;
        }
    }
    return nullptr;
}

// Validate the source shape that produces the descriptor IR type's two operands.
//
// Consider the canonical declaration:
//
//     struct TraceProgramDescriptor<ProgramLayout>
//         where ProgramLayout : ITraceProgramLayout
//
// Lowering emits the `ProgramLayout` type followed by its conformance witness. Requiring exactly
// that parameter and constraint here prevents a stale or accidentally edited standard module from
// publishing a descriptor opcode with missing or additional operands.
static bool _isValidTraceProgramDescriptorShape(
    AggTypeDecl* descriptorDecl,
    InterfaceDecl* traceProgramLayoutInterface)
{
    auto genericDecl = descriptorDecl ? as<GenericDecl>(descriptorDecl->parentDecl) : nullptr;
    if (!genericDecl || genericDecl->inner != descriptorDecl ||
        genericDecl->getDirectMemberDeclsOfType<GenericTypeParamDecl>().getCount() != 1 ||
        genericDecl->getDirectMemberDeclsOfType<GenericTypePackParamDecl>().getCount() != 0 ||
        genericDecl->getDirectMemberDeclsOfType<GenericValueParamDecl>().getCount() != 0 ||
        genericDecl->getDirectMemberDeclsOfType<GenericValuePackParamDecl>().getCount() != 0)
    {
        return false;
    }

    auto typeParam = genericDecl->getDirectMemberDeclsOfType<GenericTypeParamDecl>()[0];
    auto constraints = genericDecl->getDirectMemberDeclsOfType<GenericTypeConstraintDecl>();
    if (constraints.getCount() != 1)
        return false;

    auto constraint = constraints[0];
    auto subType = as<DeclRefType>(constraint->sub.type);
    auto supType = as<DeclRefType>(constraint->sup.type);
    return !constraint->isEqualityConstraint && subType && supType &&
           subType->getDeclRef().getDecl() == typeParam &&
           supType->getDeclRef().getDecl() == traceProgramLayoutInterface;
}

bool StructuralRayTracingDeclRegistry::registerTrustedModule(
    Module* module,
    StructuralRayTracingStageKind* outMissingStage)
{
    // Resolve into local arrays first. Publishing only a complete set ensures `isInitialized()` is
    // a reliable invariant for consumers: if it is true, every stage has an interface, input view,
    // and exact `invoke` requirement.
    auto trustedModuleDecl = module->getModuleDecl();
    auto intersectionStageInterface =
        as<InterfaceDecl>(_findNamedDecl(module, "IIntersectionStage"));
    auto rayTracerType = as<AggTypeDecl>(_findNamedDecl(module, "RayTracer"));
    auto traceProgramDescriptorType =
        as<AggTypeDecl>(_findNamedDecl(module, "TraceProgramDescriptor"));
    if (!intersectionStageInterface || !rayTracerType)
    {
        if (outMissingStage)
            *outMissingStage = StructuralRayTracingStageKind::Count;
        return false;
    }

    InterfaceDecl* interfaces[int(StructuralRayTracingStageKind::Count)] = {};
    AggTypeDecl* inputTypes[int(StructuralRayTracingStageKind::Count)] = {};
    FunctionDeclBase* invokeRequirements[int(StructuralRayTracingStageKind::Count)] = {};
    for (int i = 0; i < int(StructuralRayTracingStageKind::Count); ++i)
    {
        auto kind = StructuralRayTracingStageKind(i);
        interfaces[i] = _findStageInterface(module, kind);
        inputTypes[i] = _findStageInputType(module, kind);
        auto invokeInterface = kind == StructuralRayTracingStageKind::Intersection
                                   ? intersectionStageInterface
                                   : interfaces[i];
        if (invokeInterface)
            invokeRequirements[i] = _findStageInvokeRequirement(invokeInterface);
        if (!interfaces[i] || !inputTypes[i] || !invokeRequirements[i])
        {
            if (outMissingStage)
                *outMissingStage = kind;
            return false;
        }
    }
    InterfaceDecl* metadataInterfaces[int(StructuralRayTracingMetadataKind::Count)] = {};
    for (int i = 0; i < int(StructuralRayTracingMetadataKind::Count); ++i)
    {
        auto kind = StructuralRayTracingMetadataKind(i);
        metadataInterfaces[i] =
            as<InterfaceDecl>(_findNamedDecl(module, _getMetadataInterfaceName(kind)));
        if (!metadataInterfaces[i])
        {
            if (outMissingStage)
                *outMissingStage = StructuralRayTracingStageKind::Count;
            return false;
        }
    }

    // The authorized module build records both identities directly on the declaration. Check the
    // serialized modifiers and generic shape before publishing it: a stale package must not create
    // an ordinary AST struct or a descriptor IR type with the wrong operands.
    auto descriptorIntrinsicType =
        traceProgramDescriptorType
            ? traceProgramDescriptorType->findModifier<IntrinsicTypeModifier>()
            : nullptr;
    auto traceProgramLayoutInterface =
        metadataInterfaces[int(StructuralRayTracingMetadataKind::TraceProgramLayout)];
    if (!isTraceProgramDescriptorDecl(traceProgramDescriptorType) || !descriptorIntrinsicType ||
        descriptorIntrinsicType->irOp != kIROp_TraceProgramDescriptorType ||
        descriptorIntrinsicType->irOperands.getCount() != 0 ||
        !_isValidTraceProgramDescriptorShape(
            traceProgramDescriptorType,
            traceProgramLayoutInterface))
    {
        if (outMissingStage)
            *outMissingStage = StructuralRayTracingStageKind::Count;
        return false;
    }

    // Publish only after every required declaration has passed validation. Consumers use
    // `isInitialized()` as the guarantee that all canonical identities are available together.
    m_trustedModuleDecl = trustedModuleDecl;
    m_intersectionStageInterface = intersectionStageInterface;
    m_rayTracerType = rayTracerType;
    for (int i = 0; i < int(StructuralRayTracingStageKind::Count); ++i)
    {
        m_stageInterfaces[i] = interfaces[i];
        m_stageInputTypes[i] = inputTypes[i];
        m_stageInvokeRequirements[i] = invokeRequirements[i];
    }
    for (int i = 0; i < int(StructuralRayTracingMetadataKind::Count); ++i)
    {
        m_metadataInterfaces[i] = metadataInterfaces[i];
    }
    return true;
}

// ## Canonical declaration queries

bool StructuralRayTracingDeclRegistry::isTrustedModule(Module* module) const
{
    return module && module->getModuleDecl() == m_trustedModuleDecl;
}

InterfaceDecl* StructuralRayTracingDeclRegistry::getStageInterface(
    StructuralRayTracingStageKind kind) const
{
    auto index = int(kind);
    if (index < 0 || index >= int(StructuralRayTracingStageKind::Count))
        return nullptr;
    return m_stageInterfaces[index];
}

StructuralRayTracingStageKind StructuralRayTracingDeclRegistry::getStageKind(
    InterfaceDecl* interfaceDecl) const
{
    if (!interfaceDecl)
        return StructuralRayTracingStageKind::Count;
    if (interfaceDecl == m_intersectionStageInterface)
        return StructuralRayTracingStageKind::Intersection;
    for (int i = 0; i < int(StructuralRayTracingStageKind::Count); ++i)
    {
        if (m_stageInterfaces[i] == interfaceDecl)
            return StructuralRayTracingStageKind(i);
    }
    return StructuralRayTracingStageKind::Count;
}

AggTypeDecl* StructuralRayTracingDeclRegistry::getStageInputType(
    StructuralRayTracingStageKind kind) const
{
    auto index = int(kind);
    if (index < 0 || index >= int(StructuralRayTracingStageKind::Count))
        return nullptr;
    return m_stageInputTypes[index];
}

StructuralRayTracingStageKind StructuralRayTracingDeclRegistry::getStageInputKind(
    AggTypeDecl* typeDecl) const
{
    if (!typeDecl)
        return StructuralRayTracingStageKind::Count;
    for (int i = 0; i < int(StructuralRayTracingStageKind::Count); ++i)
    {
        if (m_stageInputTypes[i] == typeDecl)
            return StructuralRayTracingStageKind(i);
    }
    return StructuralRayTracingStageKind::Count;
}

StructuralRayTracingMetadataKind StructuralRayTracingDeclRegistry::getMetadataKind(
    InterfaceDecl* interfaceDecl) const
{
    if (!interfaceDecl)
        return StructuralRayTracingMetadataKind::Count;
    for (int i = 0; i < int(StructuralRayTracingMetadataKind::Count); ++i)
    {
        if (m_metadataInterfaces[i] == interfaceDecl)
            return StructuralRayTracingMetadataKind(i);
    }
    return StructuralRayTracingMetadataKind::Count;
}

// Return whether `functionDecl` is a method with `expectedName` declared in an extension of the
// canonical `RayTracer` type in the trusted module. Both provenance checks matter: matching only
// the method and receiver names would let user code spoof a compiler-recognized trace operation.
static bool _isRayTracerMethod(
    FunctionDeclBase* functionDecl,
    ModuleDecl* trustedModuleDecl,
    AggTypeDecl* rayTracerType,
    UnownedStringSlice expectedName)
{
    if (!functionDecl || !trustedModuleDecl || !rayTracerType || !functionDecl->getName() ||
        functionDecl->getName()->text.getUnownedSlice() != expectedName)
    {
        return false;
    }

    ExtensionDecl* extensionDecl = nullptr;
    ModuleDecl* moduleDecl = nullptr;
    // A generic method may have wrapper declarations between the function and its extension. Walk
    // to the module while retaining the nearest enclosing extension that supplies the receiver.
    for (auto parent = functionDecl->parentDecl; parent; parent = parent->parentDecl)
    {
        if (!extensionDecl)
            extensionDecl = as<ExtensionDecl>(parent);
        if (auto candidateModule = as<ModuleDecl>(parent))
        {
            moduleDecl = candidateModule;
            break;
        }
    }
    if (moduleDecl != trustedModuleDecl || !extensionDecl)
        return false;

    auto targetType = as<DeclRefType>(extensionDecl->targetType.type);
    return targetType && targetType->getDeclRef().getDecl() == rayTracerType;
}

bool StructuralRayTracingDeclRegistry::isTraceMethod(FunctionDeclBase* functionDecl) const
{
    return _isRayTracerMethod(functionDecl, m_trustedModuleDecl, m_rayTracerType, toSlice("trace"));
}

bool StructuralRayTracingDeclRegistry::isCallShaderMethod(FunctionDeclBase* functionDecl) const
{
    return _isRayTracerMethod(
        functionDecl,
        m_trustedModuleDecl,
        m_rayTracerType,
        toSlice("callShader"));
}

FunctionDeclBase* StructuralRayTracingDeclRegistry::getStageInvokeRequirement(
    StructuralRayTracingStageKind kind) const
{
    auto index = int(kind);
    if (index < 0 || index >= int(StructuralRayTracingStageKind::Count))
        return nullptr;
    return m_stageInvokeRequirements[index];
}

void StructuralRayTracingDeclRegistry::registerStageImplementation(
    FunctionDeclBase* implementation,
    StructuralRayTracingStageKind kind)
{
    if (implementation && kind != StructuralRayTracingStageKind::Count)
        m_stageImplementations[implementation] = kind;
}

StructuralRayTracingStageKind StructuralRayTracingDeclRegistry::getStageKind(
    FunctionDeclBase* implementation) const
{
    if (!implementation)
        return StructuralRayTracingStageKind::Count;
    for (int i = 0; i < int(StructuralRayTracingStageKind::Count); ++i)
    {
        if (m_stageInvokeRequirements[i] == implementation)
            return StructuralRayTracingStageKind(i);
    }
    if (auto kind = m_stageImplementations.tryGetValue(implementation))
        return *kind;
    return StructuralRayTracingStageKind::Count;
}

// ## Cross-checker semantic records

bool StructuralRayTracingDeclRegistry::registerAPIUse(
    Module* module,
    RayTracingAPIFamily family,
    Decl* decl,
    Decl** outOtherDecl)
{
    *outOtherDecl = nullptr;
    if (!module || !decl)
        return false;

    auto& usage = m_apiUsage.getOrAddValue(module, RayTracingAPIUsage());
    auto& currentDecl =
        family == RayTracingAPIFamily::Structural ? usage.structuralDecl : usage.legacyDecl;
    auto otherDecl =
        family == RayTracingAPIFamily::Structural ? usage.legacyDecl : usage.structuralDecl;

    // Preserve the first declaration from each family. It gives a stable pair of locations for the
    // one mixed-API diagnostic regardless of how many later declarations use either API.
    if (!currentDecl)
        currentDecl = decl;
    if (!otherDecl || usage.diagnosed)
        return false;

    usage.diagnosed = true;
    *outOtherDecl = otherDecl;
    return true;
}

void StructuralRayTracingDeclRegistry::registerFunctionCall(
    FunctionDeclBase* caller,
    FunctionDeclBase* callee,
    SourceLoc callLoc)
{
    if (!caller || !callee || !isInitialized())
        return;

    m_functionCallees.getOrAddValue(caller, HashSet<FunctionDeclBase*>()).add(callee);

    // A location is needed only for `callShader`, where the eventual diagnostic should identify
    // the prohibited operation rather than the stage function from which reachability started.
    if (isCallShaderMethod(callee))
        m_callShaderCallers[caller] = callLoc;
}

bool StructuralRayTracingDeclRegistry::findReachableCallShader(
    FunctionDeclBase* function,
    SourceLoc& outCallLoc) const
{
    if (!function)
        return false;

    HashSet<FunctionDeclBase*> visited;
    List<FunctionDeclBase*> workList;
    workList.add(function);
    // Traverse each recorded function at most once. The call graph may contain recursion, while a
    // work-list index gives deterministic termination without requiring recursive C++ calls.
    for (Index i = 0; i < workList.getCount(); ++i)
    {
        auto current = workList[i];
        if (!visited.add(current))
            continue;
        if (auto callLoc = m_callShaderCallers.tryGetValue(current))
        {
            outCallLoc = *callLoc;
            return true;
        }
        if (auto callees = m_functionCallees.tryGetValue(current))
        {
            for (auto callee : *callees)
                workList.add(callee);
        }
    }
    return false;
}

} // namespace Slang
