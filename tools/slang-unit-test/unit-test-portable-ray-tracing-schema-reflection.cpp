#include "core/slang-string.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

// A host can enumerate a checked schema before selecting any of its stages. Reflecting it must
// preserve list order and ordinary type identity without making otherwise-unused code live.
SLANG_UNIT_TEST(portableRayTracingSourceSchemaReflection)
{
    const char* source = R"(
        import slang.raytracing.portable;

        struct Material { float3 color; float opacity; }
        struct Attributes { float coordinate; }
        struct TraceContext : rt::ITraceContext
        {
            typealias AccelerationStructure = rt::AccelerationStructure;
            typealias Motion = rt::NoMotion;
        }
        struct HitContext<PrimitiveType : rt::IIntersectionPrimitive> : rt::IHitContext
        {
            typealias TraceContext = ::TraceContext;
            typealias Payload = uint;
            typealias Record = Material;
            typealias Primitive = PrimitiveType;
        }
        struct MissContext<P> : rt::IPayloadContext
        {
            typealias TraceContext = ::TraceContext;
            typealias Payload = P;
            typealias Record = void;
        }
        struct CallableContext : rt::ICallableContext
        {
            typealias TraceContext = ::TraceContext;
            typealias CallableData = float2;
            typealias Record = Material;
        }
        namespace Stages
        {
            struct ClosestHit : rt::IClosestHitShader
            {
                typealias Context = HitContext<rt::TrianglePrimitive>;
                static void invoke(rt::ClosestHitInput<Context> input) {}
            }
        }
        struct Intersection : rt::IIntersectionShader
        {
            typealias Context = HitContext<rt::BoundingBoxPrimitive<Attributes>>;
            static void invoke(rt::IntersectionInput<Context> input) {}
        }
        struct ProceduralGroup : rt::IHitGroup
        {
            typealias Context = Intersection.Context;
            typealias ClosestHit = rt::NoClosestHit<Context>;
            typealias AnyHit = rt::NoAnyHit<Context>;
            typealias Intersection = ::Intersection;
        }
        interface ITaggedHitGroup : rt::IHitGroup {}
        struct TriangleGroup : ITaggedHitGroup
        {
            typealias Context = Stages.ClosestHit.Context;
            typealias ClosestHit = Stages.ClosestHit;
            typealias AnyHit = rt::NoAnyHit<Context>;
            typealias Intersection = rt::NoIntersection<Context>;
        }
        interface IRefinedMiss : rt::IMissShader {}
        struct GenericMiss<P> : IRefinedMiss
        {
            typealias Context = MissContext<P>;
            static void invoke(rt::MissInput<Context> input) {}
        }
        struct Callable : rt::ICallableShader
        {
            typealias Context = CallableContext;
            static void invoke(rt::CallableInput<Context> input) {}
        }
        interface ITaggedSchema : rt::ITraceProgramSchema {}
        struct Schema<P> : ITaggedSchema
        {
            typealias TraceContext = ::TraceContext;
            typealias HitGroups = rt::HitGroupList<ProceduralGroup, TriangleGroup>;
            typealias MissShaders = rt::MissShaderList<GenericMiss<P>, GenericMiss<int>>;
            typealias CallableShaders = rt::CallableShaderList<Callable>;
        }
        typealias SchemaAlias = Schema<uint>;
        struct EmptySchema : rt::ITraceProgramSchema
        {
            typealias TraceContext = ::TraceContext;
            typealias HitGroups = rt::EmptyHitGroupList;
            typealias MissShaders = rt::EmptyMissShaderList;
            typealias CallableShaders = rt::EmptyCallableShaderList;
        }
        // Unlike executable stages, a schema is an ordinary interface conformance. Reflection
        // must not introduce a new rule requiring its implementation to be a struct.
        class ClassSchema : rt::ITraceProgramSchema
        {
            typealias TraceContext = ::TraceContext;
            typealias HitGroups = rt::EmptyHitGroupList;
            typealias MissShaders = rt::EmptyMissShaderList;
            typealias CallableShaders = rt::EmptyCallableShaderList;
        }
        RWStructuredBuffer<uint> output;
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main() { output[0] = 7; }
    )";

    for (auto target : {SLANG_HLSL, SLANG_METAL})
    {
        slang::TargetDesc targetDesc = {};
        targetDesc.format = target;
        targetDesc.profile = unitTestContext->slangGlobalSession->findProfile(
            target == SLANG_HLSL ? "sm_6_6" : "metallib_2_4");
        slang::CompilerOptionEntry experimental = {};
        experimental.name = slang::CompilerOptionName::ExperimentalFeature;
        experimental.value.kind = slang::CompilerOptionValueKind::Int;
        experimental.value.intValue0 = 1;
        slang::SessionDesc sessionDesc = {};
        sessionDesc.targetCount = 1;
        sessionDesc.targets = &targetDesc;
        sessionDesc.compilerOptionEntryCount = 1;
        sessionDesc.compilerOptionEntries = &experimental;
        ComPtr<slang::ISession> session;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
            unitTestContext->slangGlobalSession->createSession(sessionDesc, session.writeRef())));
        ComPtr<slang::IBlob> diagnostics;
        auto module = session->loadModuleFromSourceString(
            "sourceSchema",
            "sourceSchema.slang",
            source,
            diagnostics.writeRef());
        if (!module && diagnostics)
            getTestReporter()->message(
                TestMessageType::Info,
                static_cast<const char*>(diagnostics->getBufferPointer()));
        SLANG_CHECK_ABORT(module);

        // Reload a serialized module in another session. The query must use the same checked
        // declarations and witnesses, not a registry populated only when parsing source.
        ComPtr<slang::IBlob> serialized;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->serialize(serialized.writeRef())));
        ComPtr<slang::ISession> importingSession;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(unitTestContext->slangGlobalSession->createSession(
            sessionDesc,
            importingSession.writeRef())));
        module = importingSession->loadModuleFromIRBlob(
            "sourceSchema",
            "sourceSchema.slang-module",
            serialized,
            diagnostics.writeRef());
        SLANG_CHECK_ABORT(module);
        session = importingSession;

        auto layout = module->getLayout(0, diagnostics.writeRef());
        SLANG_CHECK_ABORT(layout);
        auto schema = layout->findTraceProgramSchema("Schema<uint>");
        SLANG_CHECK_ABORT(schema);
        SLANG_CHECK(schema == layout->findTraceProgramSchema("SchemaAlias"));
        SLANG_CHECK(schema->getType() == layout->findTypeByName("Schema<uint>"));
        SLANG_CHECK(schema->getTraceContextType() == layout->findTypeByName("TraceContext"));
        SLANG_CHECK(layout->findTraceProgramSchema("Material") == nullptr);
        SLANG_CHECK(layout->findTraceProgramSchema("rt.ITraceProgramSchema") == nullptr);
        SLANG_CHECK(layout->findTraceProgramSchema("Schema") == nullptr);
        SLANG_CHECK(layout->findTraceProgramSchema("noSuchType") == nullptr);
        SLANG_CHECK_ABORT(schema->getHitGroupCount() == 2);
        SLANG_CHECK_ABORT(schema->getMissShaderCount() == 2);
        SLANG_CHECK_ABORT(schema->getCallableShaderCount() == 1);
        SLANG_CHECK(schema->getHitGroup(2) == nullptr);
        SLANG_CHECK(schema->getMissShader(2) == nullptr);
        SLANG_CHECK(schema->getCallableShader(1) == nullptr);

        auto procedural = schema->getHitGroup(0);
        auto triangle = schema->getHitGroup(1);
        SLANG_CHECK(procedural->getFunctionIndex() == 0);
        SLANG_CHECK(triangle->getFunctionIndex() == 1);
        SLANG_CHECK(procedural->getType() == layout->findTypeByName("ProceduralGroup"));
        SLANG_CHECK(triangle->getType() == layout->findTypeByName("TriangleGroup"));
        SLANG_CHECK(procedural->getClosestHit() == nullptr);
        SLANG_CHECK(procedural->getAnyHit() == nullptr);
        SLANG_CHECK_ABORT(procedural->getIntersection());
        SLANG_CHECK(procedural->getIntersection()->getStage() == SLANG_STAGE_INTERSECTION);
        SLANG_CHECK(
            procedural->getIntersectionAttributesType() == layout->findTypeByName("Attributes"));
        SLANG_CHECK(triangle->getIntersection() == nullptr);
        SLANG_CHECK(triangle->getAnyHit() == nullptr);
        SLANG_CHECK_ABORT(triangle->getClosestHit());
        SLANG_CHECK(triangle->getClosestHit()->getStage() == SLANG_STAGE_CLOSEST_HIT);
        SLANG_CHECK(
            triangle->getClosestHit()->getType() == layout->findTypeByName("Stages.ClosestHit"));
        SLANG_CHECK(triangle->getPayloadType() == layout->findTypeByName("uint"));
        SLANG_CHECK(triangle->getRecordType() == layout->findTypeByName("Material"));
        SLANG_CHECK(triangle->getPrimitiveType() == layout->findTypeByName("rt.TrianglePrimitive"));
        SLANG_CHECK_ABORT(triangle->getRecordTypeLayout());
        SLANG_CHECK(triangle->getRecordTypeLayout()->getSize() == 16);
        auto expectedRules = target == SLANG_HLSL ? slang::LayoutRules::DefaultConstantBuffer
                                                  : slang::LayoutRules::DefaultStructuredBuffer;
        SLANG_CHECK(
            triangle->getRecordTypeLayout() ==
            layout->getTypeLayout(triangle->getRecordType(), expectedRules));
        auto entryName = triangle->getClosestHit()->getEntryPointName();
        if (target == SLANG_HLSL)
        {
            SLANG_CHECK(
                entryName && UnownedStringSlice(entryName) == "Stages_ClosestHit_closesthit");
        }
        else
        {
            SLANG_CHECK(entryName == nullptr);
        }

        auto miss = schema->getMissShader(0);
        SLANG_CHECK(miss->getFunctionIndex() == 0);
        SLANG_CHECK(miss->getPayloadType() == layout->findTypeByName("uint"));
        SLANG_CHECK(miss->getRecordType() == layout->findTypeByName("void"));
        SLANG_CHECK(miss->getContextType() == layout->findTypeByName("MissContext<uint>"));
        SLANG_CHECK(schema->getMissShader(1)->getPayloadType() == layout->findTypeByName("int"));
        auto callable = schema->getCallableShader(0);
        SLANG_CHECK(callable->getFunctionIndex() == 0);
        SLANG_CHECK(callable->getDataType() == layout->findTypeByName("float2"));
        SLANG_CHECK(callable->getCallable()->getStage() == SLANG_STAGE_CALLABLE);
        auto floatSchema = layout->findTraceProgramSchema("Schema<float>");
        SLANG_CHECK_ABORT(floatSchema && floatSchema != schema);
        SLANG_CHECK(
            floatSchema->getMissShader(0)->getPayloadType() == layout->findTypeByName("float"));
        auto empty = layout->findTraceProgramSchema("EmptySchema");
        SLANG_CHECK_ABORT(empty);
        SLANG_CHECK(empty->getHitGroupCount() == 0);
        SLANG_CHECK(empty->getMissShaderCount() == 0);
        SLANG_CHECK(empty->getCallableShaderCount() == 0);
        SLANG_CHECK(layout->findTraceProgramSchema("ClassSchema") != nullptr);

        ComPtr<slang::IEntryPoint> entryPoint;
        SLANG_CHECK_ABORT(
            SLANG_SUCCEEDED(module->findEntryPointByName("main", entryPoint.writeRef())));
        slang::IComponentType* components[] = {module, entryPoint};
        ComPtr<slang::IComponentType> composite;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(session->createCompositeComponentType(
            components,
            2,
            composite.writeRef(),
            diagnostics.writeRef())));
        ComPtr<slang::IComponentType> linked;
        SLANG_CHECK_ABORT(
            SLANG_SUCCEEDED(composite->link(linked.writeRef(), diagnostics.writeRef())));
        auto linkedLayout = linked->getLayout();
        auto beforeCodegen = linkedLayout->findTraceProgramSchema("Schema<uint>");
        SLANG_CHECK_ABORT(beforeCodegen);
        ComPtr<slang::IBlob> code;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
            linked->getEntryPointCode(0, 0, code.writeRef(), diagnostics.writeRef())));
        SLANG_CHECK(beforeCodegen == linkedLayout->findTraceProgramSchema("Schema<uint>"));
        UnownedStringSlice emitted(
            static_cast<const char*>(code->getBufferPointer()),
            code->getBufferSize());
        SLANG_CHECK(emitted.indexOf(toSlice("ClosestHit")) < 0);
        SLANG_CHECK(emitted.indexOf(toSlice("GenericMiss")) < 0);
    }
}
