#include "slang-reflection-portable-ray-tracing.h"

#include "slang-check-impl.h"
#include "slang-check-portable-ray-tracing.h"
#include "slang-type-layout.h"

namespace Slang
{
namespace PortableRT
{

// Reads a checked semantic annotation. These roles identify the library contract even when a user
// reaches it through another interface or gives their own type an unrelated source name.
static KnownBuiltinDeclName _getRole(Decl* declaration)
{
    auto annotation = declaration->findModifier<KnownBuiltinAttribute>();
    auto value = annotation ? as<ConstantIntVal>(annotation->name) : nullptr;
    return value ? KnownBuiltinDeclName(value->getValue()) : KnownBuiltinDeclName::COUNT;
}

// Builds public views from checked source types. The IR schema reader serves target synthesis;
// this builder serves queries made before a host selects entries, so it does not link IR at all.
struct ReflectionBuilder
{
    ProgramLayout* layout;
    Linkage* linkage;
    ASTBuilder* astBuilder;
    SemanticsVisitor* visitor;

    // A public name lookup can name any type. Recognize the checked schema contract, not a struct
    // spelling: schemas are ordinary conformances and need not be stage-implementation structs.
    // createSchema separately checks that the resulting list types can actually be enumerated.
    bool isSchema(Type* type)
    {
        if (as<GenericDeclRefType>(type))
            return false;
        for (auto facet : visitor->getShared()->getInheritanceInfo(type).facets)
        {
            auto interfaceDecl = facet->origin.declRef.as<InterfaceDecl>();
            if (!interfaceDecl)
                continue;
            for (auto requirement :
                 interfaceDecl.getDecl()->getDirectMemberDeclsOfType<AssocTypeDecl>())
            {
                if (_getRole(requirement) == KnownBuiltinDeclName::RayTracingProgramHitGroups)
                    return true;
            }
        }
        return false;
    }

    // Preserve the checker's canonical associated-type result, including specialization and
    // inherited requirements. Reflection must not create another type or conformance identity.
    Type* getAssociated(Type* type, KnownBuiltinDeclName role)
    {
        return getAssociatedType(linkage, astBuilder, type, role)->getCanonicalType();
    }

    // No* is a specific library declaration, not an empty body or an inherited default. A source
    // stage with an empty invoke remains a present stage, while placeholders reflect as null.
    RefPtr<StageReflection> createStage(Type* type, Stage kind)
    {
        auto declaration = isDeclRefTypeOf<AggTypeDecl>(type);
        SLANG_RELEASE_ASSERT(declaration);
        switch (_getRole(declaration.getDecl()))
        {
        case KnownBuiltinDeclName::RayTracingNoClosestHit:
            SLANG_RELEASE_ASSERT(kind == Stage::ClosestHit);
            return nullptr;
        case KnownBuiltinDeclName::RayTracingNoAnyHit:
            SLANG_RELEASE_ASSERT(kind == Stage::AnyHit);
            return nullptr;
        case KnownBuiltinDeclName::RayTracingNoIntersection:
            SLANG_RELEASE_ASSERT(kind == Stage::Intersection);
            return nullptr;
        default:
            break;
        }

        RefPtr<StageReflection> result = new StageReflection();
        result->type = type;
        result->stage = kind;
        if (!isMetalTarget(layout->getTargetReq()))
        {
            auto sourceName = getSourceTypeName(astBuilder, type);
            result->defaultEntryPointName = getEntryPointName(sourceName.getUnownedSlice(), kind);
        }
        return result;
    }

    // Adds the common source contract without allocating a selected-entry resource binding.
    void initializeEntry(EntryReflection* result, Type* type, Type* context, SlangUInt index)
    {
        result->type = type;
        result->contextType = context;
        result->functionIndex = index;
        result->recordType = getAssociated(context, KnownBuiltinDeclName::RayTracingContextRecord);

        // Reuse the ordinary target layout rules. This describes application data only: the
        // selected native entry's parameter reflection owns its shader-record resource binding.
        auto rules = isMetalTarget(layout->getTargetReq())
                         ? slang::LayoutRules::DefaultStructuredBuffer
                         : slang::LayoutRules::DefaultConstantBuffer;
        result->recordTypeLayout = layout->getTargetReq()->getTypeLayout(result->recordType, rules);
    }

    // Reads the explicit lists in declaration order. Payloads do not form separate source tables:
    // MissShaderList<RadianceMiss, ShadowMiss> assigns indices 0 and 1 even with different
    // payloads.
    RefPtr<SchemaReflection> createSchema(Type* type)
    {
        auto hits = getConcreteEntryTypes(
            getAssociated(type, KnownBuiltinDeclName::RayTracingProgramHitGroups));
        auto misses = getConcreteEntryTypes(
            getAssociated(type, KnownBuiltinDeclName::RayTracingProgramMissShaders));
        auto callables = getConcreteEntryTypes(
            getAssociated(type, KnownBuiltinDeclName::RayTracingProgramCallableShaders));
        if (!hits || !misses || !callables)
            return nullptr;

        RefPtr<SchemaReflection> result = new SchemaReflection();
        result->type = type;
        result->traceContextType =
            getAssociated(type, KnownBuiltinDeclName::RayTracingProgramTraceContext);
        for (Index i = 0; i < hits->getTypeCount(); ++i)
        {
            auto groupType = hits->getElementType(i)->getCanonicalType();
            auto context =
                getAssociated(groupType, KnownBuiltinDeclName::RayTracingHitGroupContext);
            RefPtr<HitGroupReflection> group = new HitGroupReflection();
            initializeEntry(group, groupType, context, SlangUInt(i));
            group->payloadType =
                getAssociated(context, KnownBuiltinDeclName::RayTracingContextPayload);
            group->primitiveType =
                getAssociated(context, KnownBuiltinDeclName::RayTracingContextPrimitive);
            group->attributesType = getAssociated(
                group->primitiveType,
                KnownBuiltinDeclName::RayTracingPrimitiveAttributes);
            group->closestHit = createStage(
                getAssociated(groupType, KnownBuiltinDeclName::RayTracingHitGroupClosestHit),
                Stage::ClosestHit);
            group->anyHit = createStage(
                getAssociated(groupType, KnownBuiltinDeclName::RayTracingHitGroupAnyHit),
                Stage::AnyHit);
            group->intersection = createStage(
                getAssociated(groupType, KnownBuiltinDeclName::RayTracingHitGroupIntersection),
                Stage::Intersection);
            result->hitGroups.add(group);
        }
        for (Index i = 0; i < misses->getTypeCount(); ++i)
        {
            auto shaderType = misses->getElementType(i)->getCanonicalType();
            auto context = getAssociated(shaderType, KnownBuiltinDeclName::RayTracingStageContext);
            RefPtr<MissShaderReflection> shader = new MissShaderReflection();
            initializeEntry(shader, shaderType, context, SlangUInt(i));
            shader->payloadType =
                getAssociated(context, KnownBuiltinDeclName::RayTracingContextPayload);
            shader->miss = createStage(shaderType, Stage::Miss);
            result->missShaders.add(shader);
        }
        for (Index i = 0; i < callables->getTypeCount(); ++i)
        {
            auto shaderType = callables->getElementType(i)->getCanonicalType();
            auto context = getAssociated(shaderType, KnownBuiltinDeclName::RayTracingStageContext);
            RefPtr<CallableShaderReflection> shader = new CallableShaderReflection();
            initializeEntry(shader, shaderType, context, SlangUInt(i));
            shader->callableDataType =
                getAssociated(context, KnownBuiltinDeclName::RayTracingContextCallableData);
            shader->callable = createStage(shaderType, Stage::Callable);
            result->callableShaders.add(shader);
        }
        return result;
    }
};

SchemaReflection* findSchemaReflection(ProgramLayout* programLayout, const char* name)
{
    if (!programLayout || !name)
        return nullptr;
    auto& cache = programLayout->portableRTSchemas;
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto program = programLayout->getProgram();
    auto linkage = program->getLinkage();
    SLANG_AST_BUILDER_RAII(linkage->getASTBuilder());
    DiagnosticSink sink(linkage->getSourceManager(), Lexer::sourceLocationLexer);

    SLANG_EXCEPTION_TRY
    {
        auto type = program->getTypeFromString(name, &sink);
        if (!type || as<ErrorType>(type) || sink.getErrorCount())
            return nullptr;
        type = type->getCanonicalType();
        RefPtr<SchemaReflection> cached;
        if (cache.schemas.tryGetValue(type, cached))
            return cached;

        auto sharedSemantics = linkage->getSemanticsForReflection();
        SemanticsVisitor visitor{SemanticsContext(sharedSemantics)};
        ReflectionBuilder builder{programLayout, linkage, linkage->getASTBuilder(), &visitor};
        if (!builder.isSchema(type))
            return nullptr;
        auto result = builder.createSchema(type);
        if (!result || sink.getErrorCount())
            return nullptr;

        // Publish only a complete, immutable source view. Asking for reflection never creates an
        // entry point, adds a liveness root, or starts target code generation.
        cache.schemas.add(type, result);
        return result;
    }
#if SLANG_HAS_EXCEPTIONS
    catch (...)
    {
        return nullptr;
    }
#endif
}

} // namespace PortableRT
} // namespace Slang
