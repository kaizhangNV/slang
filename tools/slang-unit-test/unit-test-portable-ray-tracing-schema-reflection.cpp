#include "core/slang-string.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

// Existing reflection can query schema counts and known stage types before selecting any stage.
// Querying these types must not make their implementations live.
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
        // A schema can also be a class implementing ITraceProgramSchema.
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

        // The same queries must work after loading a serialized module in a new session.
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
        auto schema = layout->findTypeByName("Schema<uint>");
        auto floatSchema = layout->findTypeByName("Schema<float>");
        SLANG_CHECK_ABORT(schema && floatSchema);
        SLANG_CHECK(schema != floatSchema); // Different generic arguments give different types.
        SLANG_CHECK(schema == layout->findTypeByName("SchemaAlias"));

        // Read each list's static count without enumerating its generic arguments.
        struct ListCount
        {
            const char* typeName;
            int count;
        };
        const ListCount listCounts[] = {
            {"Schema<uint>.HitGroups", 2},
            {"SchemaAlias.HitGroups", 2},
            {"Schema<uint>.MissShaders", 2},
            {"Schema<uint>.CallableShaders", 1},
            {"Schema<float>.MissShaders", 2},
            {"EmptySchema.HitGroups", 0},
            {"EmptySchema.MissShaders", 0},
            {"EmptySchema.CallableShaders", 0},
            {"ClassSchema.HitGroups", 0},
        };
        for (auto expected : listCounts)
        {
            auto listType = layout->findTypeByName(expected.typeName);
            SLANG_CHECK_ABORT(listType);
            auto count = layout->findVarByNameInType(listType, "count");
            SLANG_CHECK_ABORT(count);
            ComPtr<slang::IBlob> value;
            SLANG_CHECK_ABORT(SLANG_SUCCEEDED(count->getDefaultValueBlob(value.writeRef())));
            SLANG_CHECK_ABORT(value && value->getBufferSize() == sizeof(int));
            SLANG_CHECK(*static_cast<const int*>(value->getBufferPointer()) == expected.count);
        }

        // The host already knows these stage names; qualified lookup resolves their type aliases.
        struct TypeQuery
        {
            const char* path;
            const char* expectedType;
        };
        const TypeQuery associatedTypes[] = {
            {"Schema<uint>.TraceContext", "TraceContext"},
            {"SchemaAlias.HitGroups", "Schema<uint>.HitGroups"},
            {"ProceduralGroup.Context", "Intersection.Context"},
            {"ProceduralGroup.Intersection", "Intersection"},
            {"ProceduralGroup.Context.Primitive", "rt.BoundingBoxPrimitive<Attributes>"},
            {"ProceduralGroup.Context.Primitive.Attributes", "Attributes"},
            {"TriangleGroup.ClosestHit", "Stages.ClosestHit"},
            {"TriangleGroup.Context.Payload", "uint"},
            {"TriangleGroup.Context.Record", "Material"},
            {"TriangleGroup.Context.Primitive", "rt.TrianglePrimitive"},
            {"GenericMiss<uint>.Context", "MissContext<uint>"},
            {"GenericMiss<uint>.Context.Payload", "uint"},
            {"GenericMiss<uint>.Context.Record", "void"},
            {"GenericMiss<int>.Context.Payload", "int"},
            {"GenericMiss<float>.Context.Payload", "float"},
            {"Callable.Context.CallableData", "float2"},
            {"Callable.Context.Record", "Material"},
        };
        for (auto expected : associatedTypes)
        {
            auto type = layout->findTypeByName(expected.path);
            auto expectedType = layout->findTypeByName(expected.expectedType);
            SLANG_CHECK_ABORT(type && expectedType);
            SLANG_CHECK(type == expectedType);
        }

        // Inherited interfaces and class schemas use the ordinary conformance query.
        const TypeQuery conformances[] = {
            {"Schema<uint>", "rt.ITraceProgramSchema"},
            {"ClassSchema", "rt.ITraceProgramSchema"},
            {"TriangleGroup", "rt.IHitGroup"},
            {"GenericMiss<uint>", "rt.IMissShader"},
        };
        for (auto expected : conformances)
        {
            auto type = layout->findTypeByName(expected.path);
            auto interfaceType = layout->findTypeByName(expected.expectedType);
            SLANG_CHECK_ABORT(type && interfaceType);
            SLANG_CHECK(layout->isSubType(type, interfaceType));
        }

        auto record = layout->findTypeByName("TriangleGroup.Context.Record");
        auto rules = target == SLANG_HLSL ? slang::LayoutRules::DefaultConstantBuffer
                                          : slang::LayoutRules::DefaultStructuredBuffer;
        auto recordLayout = layout->getTypeLayout(record, rules);
        SLANG_CHECK_ABORT(recordLayout);
        SLANG_CHECK(recordLayout->getSize() == 16); // float3 color + float opacity.
        SLANG_CHECK_ABORT(recordLayout->getFieldCount() == 2);
        SLANG_CHECK(recordLayout->getFieldByIndex(0)->getOffset() == 0);  // color.
        SLANG_CHECK(recordLayout->getFieldByIndex(1)->getOffset() == 12); // opacity.

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
        auto beforeCodegen = linkedLayout->findTypeByName("TriangleGroup.ClosestHit");
        SLANG_CHECK_ABORT(beforeCodegen);
        ComPtr<slang::IBlob> code;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
            linked->getEntryPointCode(0, 0, code.writeRef(), diagnostics.writeRef())));
        SLANG_CHECK(beforeCodegen == linkedLayout->findTypeByName("TriangleGroup.ClosestHit"));
        UnownedStringSlice emitted(
            static_cast<const char*>(code->getBufferPointer()),
            code->getBufferSize());
        // Reflecting stages must not emit them alongside the unrelated compute entry point.
        SLANG_CHECK(emitted.indexOf(toSlice("ClosestHit")) < 0);
        SLANG_CHECK(emitted.indexOf(toSlice("GenericMiss")) < 0);
        SLANG_CHECK(emitted.indexOf(toSlice("Intersection")) < 0);
        SLANG_CHECK(emitted.indexOf(toSlice("Callable")) < 0);
    }
}
