#include "slang-portable-ray-tracing-utility.h"

#include "core/slang-string-util.h"
#include "slang-check-impl.h"
#include "slang-ir-insts.h"
#include "slang-lookup.h"
#include "slang-mangle.h"
#include "slang-module.h"
#include "slang-session.h"
#include "slang-syntax.h"

namespace Slang
{
namespace PortableRT
{

// This file owns the portable ray-tracing rules, their declaration identities, and the IR
// decorations that preserve checked program, shader-list, and stage-input type projections for
// later target lowering.
// Ordinary interfaces, associated-type constraints, overload resolution, and capability inference
// remain in the normal checker. Their call sites delegate here for the additional rules:
// stage types and inputs, stage selection, schema entry uniqueness, and native entry signatures.
// Each rule below states its example and the representation or API contract it preserves.
//
// Schemas, lists, and hit groups are ordinary values. Empty payloads are ordinary payloads.
// Concrete stage structs are ordinary values too. Generic type arguments follow ordinary rules.

// =============================================================================
// File-local helpers
// =============================================================================

// The annotation already identifies the AST type that DeclRefType::create constructs. Reading
// its class here avoids constructing a type merely to classify a declaration during checking.
static ASTNodeType _getMagicTypeClass(Decl* declaration)
{
    auto modifier = declaration ? declaration->findModifier<MagicTypeModifier>() : nullptr;
    return modifier && modifier->magicNodeType ? modifier->magicNodeType.getTag()
                                               : ASTNodeType::CountOf;
}

// Reads a semantic role from an ordinary declaration's checked attribute. For example,
// MissInput<C> remains an ordinary generic struct while its annotation identifies the stage view.
// The attribute is serialized with the declaration, so no module-import registration is needed.
static KnownBuiltinDeclName _getBuiltin(Decl* declaration)
{
    auto attribute = declaration ? declaration->findModifier<KnownBuiltinAttribute>() : nullptr;
    auto value = attribute ? as<ConstantIntVal>(attribute->name) : nullptr;
    return value ? KnownBuiltinDeclName(value->getValue()) : KnownBuiltinDeclName::COUNT;
}

static Stage _getDirectStageInputKind(Type* type)
{
    type = unwrapModifiedType(type);
    auto declRefType = as<DeclRefType>(type);
    auto typeDecl = declRefType ? declRefType->getDeclRef().as<AggTypeDecl>().getDecl() : nullptr;
    return getStageInputKind(typeDecl);
}

// Requires a concrete stage implementation to be a user-defined struct. Its invoke method is
// static, so ordinary fields and instances do not participate in stage dispatch. For example,
// `struct Miss : rt::IMissShader { uint state; ... }` is allowed: selecting Miss as a stage calls
// its static invoke function without constructing Miss or supplying a value for state.
static void _checkStageImplementationType(
    SemanticsVisitor* visitor,
    Type* stageType,
    SourceLoc conformanceLoc)
{
    SLANG_RELEASE_ASSERT(stageType);
    auto witnessedStageType = stageType;
    auto resolvedStageType = as<Type>(stageType->resolve());
    SLANG_RELEASE_ASSERT(resolvedStageType);

    auto stageDeclRef = isDeclRefTypeOf<AggTypeDecl>(resolvedStageType);
    const bool isCompilerBuiltinType =
        stageDeclRef && (stageDeclRef.getDecl()->findModifier<BuiltinTypeModifier>() ||
                         stageDeclRef.getDecl()->findModifier<MagicTypeModifier>());

    // A user-defined interface may refine a stage contract with additional requirements. It is
    // not itself a concrete stage implementation.
    if (stageDeclRef.as<InterfaceDecl>())
        return;

    // The RT API requires an ordinary struct even if another type satisfies the stage interface.
    // The StructDecl check rejects a class or an extension on a generic parameter T. It does not
    // reject `extension float : rt::IClosestHitShader`: float's built-in declaration is a struct.
    // The separate isCompilerBuiltinType check rejects float and other built-in types.
    // TODO: Allow a generic extension's target T and check the concrete types it applies to.
    // Simply skipping T here would leave those types unchecked by this rule.
    // Tracked in https://github.com/shader-slang/slang/issues/13618.
    if (!stageDeclRef.as<StructDecl>() || isCompilerBuiltinType)
    {
        const bool hasSourceAggregateDeclaration = stageDeclRef && !isCompilerBuiltinType;
        visitor->diagnoseOnce(Diagnostics::StructuralRayTracingStageImplementationMustBeStruct{
            .stageType = witnessedStageType,
            .location =
                hasSourceAggregateDeclaration ? stageDeclRef.getDecl()->loc : conformanceLoc});
    }
}

// Returns the section name for a structural schema requirement used in duplicate-entry diagnostics,
// or null for other associated-type requirements.
static const char* _getSchemaSectionName(AssocTypeDecl* requirement)
{
    switch (_getBuiltin(requirement))
    {
    case KnownBuiltinDeclName::RayTracingProgramHitGroups:
        return "hit-group";
    case KnownBuiltinDeclName::RayTracingProgramMissShaders:
        return "miss-shader";
    case KnownBuiltinDeclName::RayTracingProgramCallableShaders:
        return "callable-shader";
    default:
        break;
    }
    return nullptr;
}

static DeclRef<FuncDecl> _getStageImplementationFromSubtypeWitness(
    ASTBuilder* astBuilder,
    InterfaceDecl* stageInterface,
    SubtypeWitness* witness)
{
    witness = witness ? as<SubtypeWitness>(witness->resolve()) : nullptr;
    auto invokeRequirement = getStageInvokeRequirement(stageInterface);
    if (!invokeRequirement || !witness)
        return DeclRef<FuncDecl>();
    auto invokeWitness = tryLookUpRequirementWitness(astBuilder, witness, invokeRequirement);
    if (invokeWitness.getFlavor() != RequirementWitness::Flavor::declRef)
        return DeclRef<FuncDecl>();

    // The requirement witness is also the source of truth for the implementation's outer generic
    // substitutions. Returning only its `FuncDecl*` would turn `GenericMiss<MyPayload>.invoke`
    // back into the unspecialized declaration and make AST-to-IR lowering encounter a free generic
    // type. Preserve the checked declaration reference exactly as the conformance produced it.
    return invokeWitness.getDeclRef().as<FuncDecl>();
}

// Only the library's stage interfaces and input views have restricted value uses. A user struct
// implementing a stage interface is an ordinary value; identifying it never requires inheritance.
enum class RestrictedTypeKind
{
    None,
    StageInterface,
    StageInput,
};

static RestrictedTypeKind _getDirectRestrictedTypeKind(Type* type)
{
    type = unwrapModifiedType(type);
    if (_getDirectStageInputKind(type) != Stage::Unknown)
        return RestrictedTypeKind::StageInput;
    if (as<RayTracingStageInterfaceType>(type))
        return RestrictedTypeKind::StageInterface;
    return RestrictedTypeKind::None;
}

// Checks the supplied type and built-in containers without inspecting ordinary struct fields.
// Consider `struct Box<T> { T value; }`: its field is checked while its type is still T.
// Validating the field of Box<MissInput<C>> needs separate checking after substitution. Walking
// every ordinary struct here would force unfinished declarations, e.g. Node's own `Ptr<Node> next`.
//
// Arrays and pointers expose already-checked element types, so follow those types here. Fields
// depending on T or T.Item need validation after specialization, just like constructing T() or
// T.Item(); that limitation is tracked in shader-slang/slang#13686.
static RestrictedTypeKind _getRestrictedTypeKind(Type* type)
{
    type = unwrapModifiedType(type);
    auto kind = _getDirectRestrictedTypeKind(type);
    if (kind != RestrictedTypeKind::None)
        return kind;

    unwrapTypeElements(
        type,
        [&](Type* elementType)
        {
            kind = _getRestrictedTypeKind(elementType);
            return kind != RestrictedTypeKind::None;
        });
    return kind;
}

// Reports an invalid use of an RT builtin type: a stage input or the stage interface itself.
// "RT builtin" means a compiler-recognized library type, not necessarily a C++ BuiltinType.
// Concrete structs implementing the interface remain ordinary types. Returns true when an error
// is reported, or false when the supplied type needs no diagnostic.
static bool _diagnoseInvalidBuiltinTypeUse(DiagnosticSink* sink, Type* type, SourceLoc location)
{
    switch (_getRestrictedTypeKind(type))
    {
    case RestrictedTypeKind::StageInterface:
        // The interface itself, e.g. IMissShader, cannot be a runtime value. A concrete struct
        // implementing IMissShader is an ordinary type and does not enter this branch.
        sink->diagnose(
            Diagnostics::StructuralRayTracingStageRuntimeValue{.type = type, .location = location});
        return true;
    case RestrictedTypeKind::StageInput:
        // Allowed input parameters and local copies return from the variable checker before
        // reaching this helper. Here the context forbids an input, even in an array or pointer.
        sink->diagnose(
            Diagnostics::StructuralRayTracingInputStorage{.type = type, .location = location});
        return true;
    default:
        // Ordinary types need no diagnostic from this check.
        return false;
    }
}

// =============================================================================
// Shared functions used by other compiler files
// =============================================================================

// Allows the ray-tracing library to select its stage-interface and descriptor AST types.
// For example, `__magic_type(MissShaderType) interface IMissShader` and
// `__magic_type(TraceProgramDescriptorType) struct TraceProgramDescriptor<Schema>` both store
// their checked declaration reference. Existing type construction can therefore preserve their
// generic arguments without a custom representation. Other core magic classes may require
// different operands and cannot safely be applied to an ordinary declaration.
bool isBuiltinTypeModifier(MagicTypeModifier* modifier, Decl* decl)
{
    if (!modifier || !modifier->magicNodeType || !modifier->magicNodeType.getInfo()->createFunc)
        return false;
    return (as<InterfaceDecl>(decl) &&
            modifier->magicNodeType.isSubClassOf<RayTracingStageInterfaceType>()) ||
           (as<StructDecl>(decl) &&
            modifier->magicNodeType.isSubClassOf<TraceProgramDescriptorType>());
}

String getSourceTypeName(ASTBuilder* astBuilder, Type* stageType)
{
    auto declRefType = as<DeclRefType>(stageType ? stageType->resolve() : nullptr);
    if (!declRefType)
        return String();

    if (SubstitutionSet(declRefType->getDeclRef()).findGenericAppDeclRef())
    {
        // `GenericMiss<uint>` and `GenericMiss<float>` share one declaration path. Ordinary type
        // mangling preserves the selected arguments and provides a distinct name for each type.
        return getMangledTypeName(astBuilder, stageType->getCanonicalType());
    }

    // Ordinary declaration-reference printing gives names such as `Stages.Miss`, omitting modules.
    return declRefType->toString();
}

String getEntryPointName(UnownedStringSlice sourceTypeName, Stage stage)
{
    SLANG_RELEASE_ASSERT(isRaytracingStage(stage));

    // Keep qualified names readable in generated code: `Stages.Miss` becomes `Stages_Miss_miss`.
    // Include the selected stage so a struct implementing closest-hit and any-hit gets distinct
    // names for those two entry points.
    StringBuilder result;
    result << StringUtil::calcCharReplaced(sourceTypeName, '.', '_');
    result << "_" << getStageName(stage);
    return result.produceString();
}

Stage getStageKind(InterfaceDecl* interfaceDecl)
{
    switch (_getMagicTypeClass(interfaceDecl))
    {
    case ASTNodeType::ClosestHitShaderType:
        return Stage::ClosestHit;
    case ASTNodeType::AnyHitShaderType:
        return Stage::AnyHit;
    case ASTNodeType::IntersectionShaderType:
        return Stage::Intersection;
    case ASTNodeType::MissShaderType:
        return Stage::Miss;
    case ASTNodeType::CallableShaderType:
        return Stage::Callable;
    default:
        return Stage::Unknown;
    }
}

bool isExecutableStageInterface(InterfaceDecl* interfaceDecl)
{
    auto modifier = interfaceDecl ? interfaceDecl->findModifier<MagicTypeModifier>() : nullptr;
    return modifier && modifier->magicNodeType &&
           modifier->magicNodeType.isSubClassOf<RayTracingStageInterfaceType>();
}

Stage getStageInputKind(AggTypeDecl* typeDecl)
{
    switch (_getBuiltin(typeDecl))
    {
    case KnownBuiltinDeclName::RayTracingClosestHitInput:
        return Stage::ClosestHit;
    case KnownBuiltinDeclName::RayTracingAnyHitInput:
        return Stage::AnyHit;
    case KnownBuiltinDeclName::RayTracingIntersectionInput:
        return Stage::Intersection;
    case KnownBuiltinDeclName::RayTracingMissInput:
        return Stage::Miss;
    case KnownBuiltinDeclName::RayTracingCallableInput:
        return Stage::Callable;
    default:
        return Stage::Unknown;
    }
}

FunctionDeclBase* getStageInvokeRequirement(InterfaceDecl* interfaceDecl)
{
    if (!isExecutableStageInterface(interfaceDecl))
        return nullptr;
    for (auto member : interfaceDecl->getDirectMemberDeclsOfType<FunctionDeclBase>())
    {
        if (member->getName() && member->getName()->text == "invoke")
            return member;
    }
    return nullptr;
}

Type* getStageInputContextType(Type* inputType)
{
    auto declRef = isDeclRefTypeOf<AggTypeDecl>(unwrapModifiedType(inputType));
    if (!declRef || getStageInputKind(declRef.getDecl()) == Stage::Unknown)
        return nullptr;
    auto generic = as<GenericDecl>(declRef.getDecl()->parentDecl);
    SLANG_RELEASE_ASSERT(generic);
    auto application = SubstitutionSet(declRef).findGenericAppDeclRef(generic);
    SLANG_RELEASE_ASSERT(application);
    Type* result = nullptr;
    for (auto argument : application->getArgs())
    {
        if (auto type = as<Type>(argument->resolve()))
        {
            SLANG_RELEASE_ASSERT(!result);
            result = type;
        }
    }
    SLANG_RELEASE_ASSERT(result);
    return result;
}

// Consider ClosestHitInput<C>: Payload is declared by IPayloadContext, not IHitContext.
// The inheritance facet already contains the exact witness to that declaring interface.
// Preserve its canonical lookup, rather than rebuilding a substitution or assuming an order
// for interface requirements. Binding reflection and AST-to-IR lowering use the same result.
Type* getAssociatedType(
    Linkage* linkage,
    ASTBuilder* astBuilder,
    Type* owner,
    KnownBuiltinDeclName role)
{
    auto sharedSemantics = linkage->getSemanticsForReflection();
    SemanticsVisitor visitor{SemanticsContext(sharedSemantics)};
    for (auto facet : visitor.getShared()->getInheritanceInfo(owner).facets)
    {
        auto interfaceDecl = facet->origin.declRef.as<InterfaceDecl>();
        if (!interfaceDecl)
            continue;
        for (auto requirement :
             interfaceDecl.getDecl()->getDirectMemberDeclsOfType<AssocTypeDecl>())
        {
            if (_getBuiltin(requirement) == role)
                return DeclRefType::create(
                    astBuilder,
                    astBuilder->getLookupDeclRef(facet->subtypeWitness, requirement));
        }
    }
    SLANG_UNEXPECTED("portable ray-tracing contract is missing its associated-type role");
}

// Returns the checked list's type pack for duplicate-entry checking, without reparsing
// the generic argument list.
ConcreteTypePack* getConcreteEntryTypes(Type* entryListType)
{
    if (auto declRefType = as<DeclRefType>(entryListType->resolve()))
    {
        if (auto genericApp = SubstitutionSet(declRefType->getDeclRef()).findGenericAppDeclRef())
        {
            // Select the type pack by its semantic kind, independently of the positions of the
            // generic arguments. An empty list already has a zero-length type pack.
            for (auto argument : genericApp->getArgs())
            {
                if (auto typePack = as<ConcreteTypePack>(argument->resolve()))
                    return typePack;
            }
        }
    }
    return nullptr;
}

// Decorates the input's IRStructType with the types needed to synthesize native parameters.
// Consider the builtin declaration:
//
//     struct MissInput<Context> where Context : IPayloadContext { /* Accessors omitted. */ }
//
// Its empty IRStructType receives the operands (Miss, Context.TraceContext.AccelerationStructure,
// Context.Payload, Context.Record, void, void). These ordinary associated-type lookups expose
// their dependencies to linking. Specializing MissInput<MyContext> resolves them to concrete
// types before target synthesis, including when MyContext is supplied by another linked module.
// Hit inputs also retain Context.Primitive and Context.Primitive.Attributes, while CallableInput
// retains Context.CallableData in the payload slot. No later witness-table inspection is needed.
void addStageInputDecoration(
    IRBuilder* builder,
    Linkage* linkage,
    ASTBuilder* astBuilder,
    AggTypeDecl* decl,
    IRType* irType,
    IRType* (*lowerType)(Type*, void*),
    void* userData)
{
    auto stage = getStageInputKind(decl);
    if (stage == Stage::Unknown)
        return;

    auto generic = as<GenericDecl>(decl->parentDecl);
    SLANG_RELEASE_ASSERT(generic);
    // Each builtin input declares one context parameter. Its checked constraints supply the
    // canonical associated-type lookups, including inherited Record and TraceContext members.
    GenericTypeParamDecl* contextParameter = nullptr;
    for (auto parameter : generic->getDirectMemberDeclsOfType<GenericTypeParamDecl>())
    {
        SLANG_RELEASE_ASSERT(!contextParameter);
        contextParameter = parameter;
    }
    SLANG_RELEASE_ASSERT(contextParameter);
    auto contextType =
        DeclRefType::create(astBuilder, astBuilder->getDirectDeclRef(contextParameter));
    auto traceContextType = getAssociatedType(
        linkage,
        astBuilder,
        contextType,
        KnownBuiltinDeclName::RayTracingContextTraceContext);
    auto accelerationStructureType = getAssociatedType(
        linkage,
        astBuilder,
        traceContextType,
        KnownBuiltinDeclName::RayTracingTraceAccelerationStructure);
    auto payloadRole = stage == Stage::Callable
                           ? KnownBuiltinDeclName::RayTracingContextCallableData
                           : KnownBuiltinDeclName::RayTracingContextPayload;
    auto payloadType = getAssociatedType(linkage, astBuilder, contextType, payloadRole);
    auto recordType = getAssociatedType(
        linkage,
        astBuilder,
        contextType,
        KnownBuiltinDeclName::RayTracingContextRecord);
    IRType* primitiveType = builder->getVoidType();
    IRType* attributesType = builder->getVoidType();
    if (stage == Stage::ClosestHit || stage == Stage::AnyHit || stage == Stage::Intersection)
    {
        auto primitive = getAssociatedType(
            linkage,
            astBuilder,
            contextType,
            KnownBuiltinDeclName::RayTracingContextPrimitive);
        primitiveType = lowerType(primitive, userData);
        attributesType = lowerType(
            getAssociatedType(
                linkage,
                astBuilder,
                primitive,
                KnownBuiltinDeclName::RayTracingPrimitiveAttributes),
            userData);
    }
    IRInst* operands[] = {
        builder->getIntValue(builder->getIntType(), IRIntegerValue(stage)),
        lowerType(accelerationStructureType, userData),
        lowerType(payloadType, userData),
        lowerType(recordType, userData),
        primitiveType,
        attributesType,
    };
    builder->addDecoration(
        irType,
        kIROp_PortableRTStageInputDecoration,
        operands,
        SLANG_COUNT_OF(operands));
}

// Adds a decoration to the shader-list IRStructType with the types needed for backend synthesis.
// Consider this example, where FirstGroup and SecondGroup conform to IHitGroup:
//
//     typealias HitGroups = rt::HitGroupList<FirstGroup, SecondGroup>;
//
// Before the concrete groups are known, the decoration on HitGroupList conceptually looks like:
//
//     PortableRTShaderListDecoration
//     {
//         typepack          = Group
//         recordTypes       = expand (each Group).Context.Record
//         payloadTypes      = expand (each Group).Context.Payload
//         primitiveTypes    = expand (each Group).Context.Primitive
//         attributesTypes   = expand (each Group).Context.Primitive.Attributes
//         closestHitTypes   = expand (each Group).ClosestHit
//         anyHitTypes       = expand (each Group).AnyHit
//         intersectionTypes = expand (each Group).Intersection
//     }
//
// Group is the list's generic type pack. The expression `expand (each Group).Context.Record`
// means a symbolic witness lookup for each type in that pack. It lowers to an IRExpand whose
// body follows Context and then Record with ordinary witness lookup instructions. These use the
// list's existing generic conformance witnesses.
// After specialization, these expressions become concrete type packs in the original order.
// For example, if the two groups use FirstRecord and SecondRecord, recordTypes becomes
// [FirstRecord, SecondRecord]. The backend can then read that list directly.
//
// We need the decoration because the list has no fields, and IHitGroupList exposes only count.
// Storing the groups' conformance tables would still require the backend to follow other tables,
// such as the Context conformance and its base-interface conformances, to find Record. Keeping
// a table alive does not keep all its entries: the linker copies entries requested by IR uses.
// Preserving everything instead would require recursively copying those tables and their entries,
// including unused shader methods such as invoke and the code they depend on. These explicit
// lookups tell the normal linker which entries are needed, even before the groups are concrete.
// Specialization resolves the lookups, so backend synthesis needs no witness-table traversal.
//
// Miss and callable lists store their entry pack and expanded record types; the remaining packs
// are empty. The three hit-stage packs let the backend recognize No* placeholders; none of these
// lookups requests an invoke implementation.
void addShaderListDecoration(
    IRBuilder* builder,
    Linkage* linkage,
    ASTBuilder* astBuilder,
    AggTypeDecl* decl,
    IRType* irType,
    IRType* (*lowerType)(Type*, void*),
    void* userData)
{
    auto role = _getBuiltin(decl);
    if (role != KnownBuiltinDeclName::RayTracingHitGroupList &&
        role != KnownBuiltinDeclName::RayTracingMissShaderList &&
        role != KnownBuiltinDeclName::RayTracingCallableShaderList)
        return;

    auto generic = as<GenericDecl>(decl->parentDecl);
    SLANG_RELEASE_ASSERT(generic);
    GenericTypePackParamDecl* typePackParameter = nullptr;
    for (auto parameter : generic->getDirectMemberDeclsOfType<GenericTypePackParamDecl>())
    {
        SLANG_RELEASE_ASSERT(!typePackParameter);
        typePackParameter = parameter;
    }
    SLANG_RELEASE_ASSERT(typePackParameter);
    Type* typePack =
        DeclRefType::create(astBuilder, astBuilder->getDirectDeclRef(typePackParameter));
    auto eachEntry = astBuilder->getEachType(typePack);
    const bool isHitGroupList = role == KnownBuiltinDeclName::RayTracingHitGroupList;
    auto contextType = getAssociatedType(
        linkage,
        astBuilder,
        eachEntry,
        isHitGroupList ? KnownBuiltinDeclName::RayTracingHitGroupContext
                       : KnownBuiltinDeclName::RayTracingStageContext);
    auto recordType = getAssociatedType(
        linkage,
        astBuilder,
        contextType,
        KnownBuiltinDeclName::RayTracingContextRecord);

    // Each expansion uses the same checked type-pack parameter. Ordinary lookup lowering supplies
    // the conformance witnesses; the decoration stores the resulting types.
    Val* captures[] = {typePack};
    auto lowerExpansion = [&](Type* pattern)
    { return lowerType(astBuilder->getExpandType(pattern, makeArrayView(captures)), userData); };
    auto emptyPack = builder->getTypePack(0, nullptr);
    IRInst* payloadTypes = emptyPack;
    IRInst* primitiveTypes = emptyPack;
    IRInst* attributesTypes = emptyPack;
    IRInst* closestHitTypes = emptyPack;
    IRInst* anyHitTypes = emptyPack;
    IRInst* intersectionTypes = emptyPack;
    if (isHitGroupList)
    {
        auto payloadType = getAssociatedType(
            linkage,
            astBuilder,
            contextType,
            KnownBuiltinDeclName::RayTracingContextPayload);
        auto primitiveType = getAssociatedType(
            linkage,
            astBuilder,
            contextType,
            KnownBuiltinDeclName::RayTracingContextPrimitive);
        auto attributesType = getAssociatedType(
            linkage,
            astBuilder,
            primitiveType,
            KnownBuiltinDeclName::RayTracingPrimitiveAttributes);
        payloadTypes = lowerExpansion(payloadType);
        primitiveTypes = lowerExpansion(primitiveType);
        attributesTypes = lowerExpansion(attributesType);
        closestHitTypes = lowerExpansion(getAssociatedType(
            linkage,
            astBuilder,
            eachEntry,
            KnownBuiltinDeclName::RayTracingHitGroupClosestHit));
        anyHitTypes = lowerExpansion(getAssociatedType(
            linkage,
            astBuilder,
            eachEntry,
            KnownBuiltinDeclName::RayTracingHitGroupAnyHit));
        intersectionTypes = lowerExpansion(getAssociatedType(
            linkage,
            astBuilder,
            eachEntry,
            KnownBuiltinDeclName::RayTracingHitGroupIntersection));
    }
    IRInst* operands[] = {
        lowerType(typePack, userData),
        lowerExpansion(recordType),
        payloadTypes,
        primitiveTypes,
        attributesTypes,
        closestHitTypes,
        anyHitTypes,
        intersectionTypes,
    };
    builder->addDecoration(
        irType,
        kIROp_PortableRTShaderListDecoration,
        operands,
        SLANG_COUNT_OF(operands));
}

// Adds a decoration to IRTraceProgramDescriptorType with the schema types needed by the backend.
// Consider this declaration, where Schema conforms to ITraceProgramSchema:
//
//     rt::TraceProgramDescriptor<Schema> descriptor;
//
// The decoration belongs to the descriptor's IR type and conceptually looks like:
//
//     PortableRTProgramDecoration
//     {
//         accelerationStructureType = Schema.TraceContext.AccelerationStructure
//         motionType                = Schema.TraceContext.Motion
//         hitGroupsType             = Schema.HitGroups
//         missShadersType           = Schema.MissShaders
//         callableShadersType       = Schema.CallableShaders
//     }
//
// Each expression is a symbolic associated-type lookup through Schema's conformance witness.
// The first two also look up the TraceContext conformance to find its associated types.
// This uses the same ordinary witness lookup instructions as addShaderListDecoration. There is
// no IRExpand here: we look up five values from one schema, without iterating its list members.
//
// Schema may still be generic or supplied by another module. Once linking and specialization
// resolve these lookups, the decoration refers to concrete types. For example, if Schema.HitGroups
// is HitGroupList<FirstGroup, SecondGroup>, hitGroupsType refers to that specialized list struct.
// Its own shader-list decoration supplies the member-type packs described above. The acceleration
// structure and motion types tell Metal how to configure traversal; the three list types tell it
// which groups, miss shaders, and callable shaders belong to the program.
//
// We need these explicit lookups because the descriptor's existing Schema and conformance
// operands identify its type, but do not request the associated-type entries needed by synthesis.
// Storing the conformance alone would leave the backend to follow tables whose entries the linker
// may not have copied. The lookup instructions request the required entries before specialization,
// avoiding the need to recursively copy complete tables and unused method implementations.
// The backend then reads the resulting types directly. The decoration adds no runtime fields.
void addTraceProgramDescriptorDecoration(
    IRBuilder* builder,
    Linkage* linkage,
    ASTBuilder* astBuilder,
    TraceProgramDescriptorType* descriptorType,
    IRType* irType,
    IRType* (*lowerType)(Type*, void*),
    void* userData)
{
    // Intrinsic types are uniqued by their identity operands. Another lowering request can
    // return the same descriptor type, whose projections have already been attached.
    if (irType->findDecoration<IRPortableRTProgramDecoration>())
        return;
    auto declRef = descriptorType->getDeclRef();
    auto generic = as<GenericDecl>(declRef.getDecl()->parentDecl);
    SLANG_RELEASE_ASSERT(generic);
    auto application = SubstitutionSet(declRef).findGenericAppDeclRef(generic);
    SLANG_RELEASE_ASSERT(application);
    Type* schemaType = nullptr;
    for (auto argument : application->getArgs())
    {
        if (auto type = as<Type>(argument->resolve()))
        {
            SLANG_RELEASE_ASSERT(!schemaType);
            schemaType = type;
        }
    }
    SLANG_RELEASE_ASSERT(schemaType);
    auto traceContextType = getAssociatedType(
        linkage,
        astBuilder,
        schemaType,
        KnownBuiltinDeclName::RayTracingProgramTraceContext);
    IRInst* operands[] = {
        lowerType(
            getAssociatedType(
                linkage,
                astBuilder,
                traceContextType,
                KnownBuiltinDeclName::RayTracingTraceAccelerationStructure),
            userData),
        lowerType(
            getAssociatedType(
                linkage,
                astBuilder,
                traceContextType,
                KnownBuiltinDeclName::RayTracingTraceMotion),
            userData),
        lowerType(
            getAssociatedType(
                linkage,
                astBuilder,
                schemaType,
                KnownBuiltinDeclName::RayTracingProgramHitGroups),
            userData),
        lowerType(
            getAssociatedType(
                linkage,
                astBuilder,
                schemaType,
                KnownBuiltinDeclName::RayTracingProgramMissShaders),
            userData),
        lowerType(
            getAssociatedType(
                linkage,
                astBuilder,
                schemaType,
                KnownBuiltinDeclName::RayTracingProgramCallableShaders),
            userData),
    };
    builder->addDecoration(
        irType,
        kIROp_PortableRTProgramDecoration,
        operands,
        SLANG_COUNT_OF(operands));
}

// Returns the checked entry types used by the duplicate-entry diagnostic. The list's conformance
// witnesses are handled by ordinary conformance checking and are not needed to compare types.
static ConcreteTypePack* _getEntryTypes(ASTBuilder* astBuilder, Type* entryListType)
{
    if (auto typePack = getConcreteEntryTypes(entryListType))
        return typePack;
    // The list itself may still be a type parameter. Consider this example:
    //
    //     struct Schema<Misses : rt::IMissShaderList> : rt::ITraceProgramSchema
    //     {
    //         // Other requirements omitted.
    //         typealias MissShaders = Misses;
    //     }
    //
    // While checking Schema, we cannot enumerate Misses. Return an empty pack so the caller's
    // duplicate-entry loop has nothing to visit. This does not mean Misses is an empty list.
    // A list such as MissShaderList<T> already has an enumerable pack, even if T is generic.
    // TODO: Check duplicates after specialization too. This fallback does not schedule a later
    // check; Schema<MissShaderList<M, M>> currently passes this declaration-time restriction.
    return astBuilder->getTypePack(ArrayView<Type*>());
}

// Checks the selected native entry signature, including aliases and closed generic arguments.
// A helper may accept MissInput<C>, but selecting that helper with -entry requires this check.
// Source imports and saved modules use the same checked function reference. Types revealed only
// by later, separate specialization still need validation at that specialization boundary.
void diagnoseInputInNativeEntryPoint(EntryPoint* entryPoint, DiagnosticSink* sink)
{
    SLANG_RELEASE_ASSERT(!entryPoint->isStructuralRayTracingEntryPoint());
    auto diagnoseInputType = [sink](Type* type, SourceLoc location)
    {
        if (_getRestrictedTypeKind(type) == RestrictedTypeKind::StageInput)
        {
            sink->diagnose(Diagnostics::StructuralRayTracingInputInNativeEntryPoint{
                .type = type,
                .location = location});
        }
    };

    auto astBuilder = entryPoint->getLinkage()->getASTBuilder();
    auto function = entryPoint->getFuncDeclRef();
    for (auto parameter : getParameters(astBuilder, function))
    {
        auto parameterDecl = parameter.getDecl();
        auto location = parameterDecl->type.exp ? parameterDecl->type.exp->loc : parameterDecl->loc;
        diagnoseInputType(getType(astBuilder, parameter), location);
    }
    auto resultTypeExpr = function.getDecl()->returnType.exp;
    diagnoseInputType(
        getResultType(astBuilder, function),
        resultTypeExpr ? resultTypeExpr->loc : function.getDecl()->loc);
}

// Selects one executable conformance for a source struct named by -entry Stage or Stage.invoke.
// For example, a struct implementing both IMissShader and ICallableShader needs `-stage miss`
// or `-stage callable`; a struct implementing only IMissShader can omit the explicit stage.
// The selected witness supplies the exact invoke implementation and preserves dependent types.
// Generic stages currently need arguments in the entry name, for example `-entry "M<P>"`.
// TODO(#13800): Support separate CLI/API specialization while keeping invoke, its stage contract,
// and target validation consistent with the concrete arguments.
DeclRef<FuncDecl> findEntryPointByName(
    Linkage* linkage,
    Module* module,
    Name* name,
    Profile& ioProfile,
    DiagnosticSink* sink,
    bool* outFoundStruct,
    Type** outStageType)
{
    // Both `Stage` and `Stage.invoke` select the stage's checked conformance. Read the receiver
    // type from member lookup so `GenericStage<Payload>.invoke` keeps its generic arguments.
    // The selected conformance, rather than ordinary overload resolution, chooses the method
    // when one struct implements several stages.
    *outFoundStruct = false;
    *outStageType = nullptr;
    SharedSemanticsContext sharedContext(linkage, module, sink);
    SemanticsVisitor visitor(&sharedContext);
    auto expr = module->findDeclFromString(getText(name), sink);
    auto declRefExpr = as<DeclRefExpr>(expr);
    auto stageTypeDeclRef =
        declRefExpr ? declRefExpr->declRef.as<AggTypeDecl>() : DeclRef<AggTypeDecl>();
    bool selectedInvoke = false;
    if (!stageTypeDeclRef)
    {
        auto overloadedExpr = as<OverloadedExpr>(expr);
        auto memberName = declRefExpr      ? declRefExpr->declRef.getName()
                          : overloadedExpr ? overloadedExpr->lookupResult2.getName()
                                           : nullptr;
        if (getText(memberName) != "invoke")
            return DeclRef<FuncDecl>();
        auto base = visitor.GetBaseExpr(expr);
        auto typeType = base ? as<TypeType>(base->type) : nullptr;
        if (!typeType)
            return DeclRef<FuncDecl>();
        auto stageType = typeType->getType();
        if (declRefExpr)
        {
            // With a default implementation, `Stage.invoke` can reference the interface's
            // requirement. Its This witness retains Stage even though the base is the interface.
            if (auto thisWitness = findThisTypeWitness(
                    SubstitutionSet(declRefExpr->declRef),
                    isDeclRefTypeOf<InterfaceDecl>(stageType).getDecl()))
                stageType = thisWitness->getSub();
        }
        stageTypeDeclRef = isDeclRefTypeOf<AggTypeDecl>(stageType);
        selectedInvoke = true;
    }
    if (!stageTypeDeclRef || getModule(stageTypeDeclRef.getDecl()) != module)
        return DeclRef<FuncDecl>();

    *outFoundStruct = !selectedInvoke;
    for (auto dependency : module->getModuleDependencies())
    {
        auto moduleDecl = dependency->getModuleDecl();
        if (sharedContext.importedModulesSet.add(moduleDecl))
            sharedContext.importedModulesList.add(moduleDecl);
    }
    visitor.ensureDecl(stageTypeDeclRef, DeclCheckState::ReadyForConformances);

    DeclRef<FuncDecl> stageImplementations[SLANG_STAGE_COUNT] = {};
    bool ambiguousImplementations[SLANG_STAGE_COUNT] = {};
    auto stageType = DeclRefType::create(linkage->getASTBuilder(), stageTypeDeclRef);
    for (auto facet : visitor.getShared()->getInheritanceInfo(stageType).facets)
    {
        auto interfaceDeclRef = facet->origin.declRef.as<InterfaceDecl>();
        auto kind = getStageKind(interfaceDeclRef.getDecl());
        if (kind != Stage::Unknown)
        {
            if (auto implementation = _getStageImplementationFromSubtypeWitness(
                    visitor.getASTBuilder(),
                    interfaceDeclRef.getDecl(),
                    facet->subtypeWitness))
            {
                // Two module-local interfaces can identify the same logical stage. Repeated
                // facets selecting the same checked method are harmless, but distinct methods
                // cannot be selected by a stage name alone. Keep the first witness and diagnose
                // only if this is the stage requested by the entry point.
                if (stageImplementations[int(kind)])
                {
                    if (stageImplementations[int(kind)] != implementation)
                        ambiguousImplementations[int(kind)] = true;
                    continue;
                }
                stageImplementations[int(kind)] = implementation;
            }
        }
    }

    Count implementedStageCount = 0;
    Stage onlyImplementedStage = Stage::Unknown;
    for (UInt i = 0; i < SLANG_STAGE_COUNT; ++i)
    {
        if (stageImplementations[i])
        {
            ++implementedStageCount;
            onlyImplementedStage = Stage(i);
        }
    }

    if (implementedStageCount == 0)
    {
        // An ordinary struct may also have a static method named invoke.
        if (selectedInvoke)
            return DeclRef<FuncDecl>();
        sink->diagnose(Diagnostics::StructuralRayTracingEntryPointNotStage{
            .stageType = stageTypeDeclRef.getDecl()});
        return DeclRef<FuncDecl>();
    }
    *outFoundStruct = true;

    auto requestedStage = ioProfile.getStage();
    auto selectedStage = requestedStage;
    if (requestedStage != Stage::Unknown)
    {
        if (UInt(selectedStage) >= SLANG_STAGE_COUNT || !stageImplementations[int(selectedStage)])
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
    }
    else
    {
        sink->diagnose(Diagnostics::StructuralRayTracingEntryPointAmbiguousStage{
            .stageType = stageTypeDeclRef.getDecl()});
        return DeclRef<FuncDecl>();
    }

    if (ambiguousImplementations[int(selectedStage)])
    {
        sink->diagnose(Diagnostics::StructuralRayTracingEntryPointAmbiguousImplementation{
            .stage = getStageName(selectedStage),
            .stageType = stageTypeDeclRef.getDecl()});
        return DeclRef<FuncDecl>();
    }

    auto invokeMethod = stageImplementations[int(selectedStage)];
    if (!invokeMethod)
    {
        sink->diagnose(Diagnostics::InternalCompilerError{.location = stageTypeDeclRef.getLoc()});
        return DeclRef<FuncDecl>();
    }

    // Selecting an IMissShader implementation gives this entry a portable_miss profile.
    // Its public stage is still Miss, but capability checks must use the portable contract
    // instead of the native miss contract, which excludes Metal. Keep any requested version.
    ioProfile.setPortableRayTracingStage(selectedStage);
    *outStageType = stageType;
    return invokeMethod;
}

// Sets the names of an entry point selected through a checked stage conformance.
// Preserve the requested public name, such as Stages.Miss or Stages.Miss.invoke. Both spellings
// derive the same target symbol from the stage type and select the same checked invoke method.
void initializeEntryPoint(EntryPoint* entryPoint, Name* entryPointName, Type* stageType)
{
    SLANG_ASSERT(entryPoint->isStructuralRayTracingEntryPoint());
    entryPoint->setNameOverride(entryPointName);
    auto sourceTypeName = getSourceTypeName(entryPoint->getLinkage()->getASTBuilder(), stageType);
    entryPoint->setEntryPointNameOverride(
        getEntryPointName(sourceTypeName.getUnownedSlice(), entryPoint->getStage()));
}

// =============================================================================
// Checks called by ordinary semantic checking
// =============================================================================

// Checks the extra RT API restrictions after ordinary interface conformance checking succeeds.
// For example, `struct Miss : rt::IMissShader` must be a user-defined struct.
// Ordinary conformance checking has already verified its required members, such as Context and
// invoke. Applying these restrictions at that point also covers a Miss listed in a schema
// without being selected directly with `-entry Miss`.
// A conformance to an ordinary interface, such as ITraceProgramSchema, needs no stage restrictions.
void checkStageRestrictions(
    SemanticsVisitor* visitor,
    DeclRef<InterfaceDecl> superInterfaceDeclRef,
    WitnessTable* witnessTable,
    SourceLoc conformanceLoc)
{
    auto stageKind = getStageKind(superInterfaceDeclRef.getDecl());
    if (stageKind == Stage::Unknown || !witnessTable)
        return;

    _checkStageImplementationType(visitor, witnessTable->witnessedType, conformanceLoc);
}

void diagnoseDuplicateSchemaEntries(
    ASTBuilder* astBuilder,
    DiagnosticSink* sink,
    Type* entryListType,
    AssocTypeDecl* associatedTypeRequirement,
    Type* schemaType,
    SourceLoc satisfyingDeclLoc)
{

    auto sectionName = _getSchemaSectionName(associatedTypeRequirement);
    if (!sectionName)
        return;

    SLANG_RELEASE_ASSERT(entryListType && schemaType);

    // Consider this example:
    //
    //     typealias HitGroups = HitGroupList<OpaqueHit, OpaqueHit>;
    //
    // The type pack in this checked associated-type witness is the source declaration's exact
    // list of implementations. Each implementation receives one dense function index; it
    // does not stand for a physical SBT record. A host can therefore reuse `OpaqueHit` in any
    // number of records, but listing it twice here would assign one implementation two indices and
    // make reflection ambiguous. Reject that source contract before lowering, because the IR and
    // reflection representations intentionally rely on unique entries.
    //
    // Canonical AST identity makes aliases name the same entry without reconstructing or
    // structurally comparing types here.
    auto entries = _getEntryTypes(astBuilder, entryListType->getCanonicalType());
    HashSet<Type*> seenEntries;
    for (Index i = 0; i < entries->getTypeCount(); ++i)
    {
        auto entryType = entries->getElementType(i)->getCanonicalType();
        if (seenEntries.add(entryType))
            continue;

        sink->diagnose(Diagnostics::DuplicateStructuralRayTracingEntry{
            .section = sectionName,
            .entry = entryType,
            .schema = schemaType->getCanonicalType(),
            .location = satisfyingDeclLoc});
    }
}

// Checks the declaration's own type, direction, scope, and initializer. Consider:
//
//     void helper(in rt::MissInput<C> input)
//     {
//         let copy = input;          // Refers to the existing invocation's input.
//         rt::MissInput<C> fresh;    // Has no existing input to refer to.
//     }
//
// A direct input is allowed as a non-uniform in parameter or an initialized non-static local.
// Globals, fields, and other parameter directions are rejected. The input types have private
// constructors, so ordinary constructor access checking rejects direct construction.
void diagnoseInvalidVariableType(DiagnosticSink* sink, VarDeclBase* varDecl)
{
    auto type = varDecl->getType();
    // Recognize the input type itself, such as MissInput<C>. Unknown means it is not a stage input.
    if (_getDirectStageInputKind(type) != Stage::Unknown)
    {
        if (auto param = as<ParamDecl>(varDecl))
        {
            // checkCallableDeclCommon also calls this helper for function parameters. An in
            // parameter receives an existing input from its caller; uniform would instead request
            // a host binding. Uniform parameters and other directions reach the rejection below.
            if (getExplicitlyDeclaredParamPassingMode(param) == ParamPassingMode::In &&
                !param->hasModifier<HLSLUniformModifier>())
                return;
        }
        else if (
            isLocalVar(varDecl) && !varDecl->hasModifier<HLSLStaticModifier>() && varDecl->initExpr)
        {
            return;
        }
    }
    // Reject input declarations that were not accepted above: globals, fields, static locals,
    // disallowed parameters, and uninitialized locals such as `MissInput<C> fresh;`. This also
    // checks restricted types inside built-in containers and the stage interface itself.
    // Reaching this call does not imply an error: ordinary types, including concrete stage
    // structs, pass without a diagnostic.
    _diagnoseInvalidBuiltinTypeUse(sink, type, varDecl->loc);
}

// Rejects results such as `rt::MissInput<C> makeInput();`. Inputs may be received as in parameters
// and copied locally, but a function cannot promise a new input. Constructors do not have
// a separately declared return type, so they are not subject to this check.
void diagnoseInvalidReturnType(DiagnosticSink* sink, CallableDecl* decl)
{
    if (as<ConstructorDecl>(decl))
        return;
    _diagnoseInvalidBuiltinTypeUse(
        sink,
        decl->returnType.type,
        decl->returnType.exp ? decl->returnType.exp->loc : decl->loc);
}

// Rejects a property such as `property rt::MissInput<C> input { get; }`. Payload and record
// properties have ordinary application-defined result types and remain legal.
void diagnoseInvalidPropertyType(DiagnosticSink* sink, PropertyDecl* propertyDecl)
{
    _diagnoseInvalidBuiltinTypeUse(
        sink,
        propertyDecl->type.type,
        propertyDecl->type.exp ? propertyDecl->type.exp->loc : propertyDecl->loc);
}

} // namespace PortableRT
} // namespace Slang
