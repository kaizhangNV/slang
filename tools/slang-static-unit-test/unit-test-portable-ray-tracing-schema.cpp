#include "slang-com-ptr.h"
#include "slang.h"
#include "slang/slang-code-gen.h"
#include "slang/slang-ir-dce.h"
#include "slang/slang-ir-link.h"
#include "slang/slang-ir-portable-ray-tracing-schema.h"
#include "slang/slang-ir-specialize.h"
#include "slang/slang-ir-util.h"
#include "slang/slang-module.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

// Exercise the actual source producer, linker, specialization, and DCE boundary used by Metal
// synthesis. A selected miss stage traces another ray but never directly calls its reached
// stages: the descriptor's schema witness must retain them. Unrelated stages must not be roots.
SLANG_UNIT_TEST(portableRayTracingSchemaRetainsOnlyReferencedStages)
{
    const char* source = R"(
        import slang.raytracing.portable;

        struct Payload { uint value; }
        struct TraceContext : rt::ITraceContext
        {
            typealias AccelerationStructure = rt::AccelerationStructure;
            typealias Motion = rt::NoMotion;
        }
        struct HitContext : rt::IHitContext
        {
            typealias TraceContext = ::TraceContext;
            typealias Payload = ::Payload;
            typealias Record = void;
            typealias Primitive = rt::TrianglePrimitive;
        }
        struct MissContext : rt::IPayloadContext
        {
            typealias TraceContext = ::TraceContext;
            typealias Payload = ::Payload;
            typealias Record = void;
        }
        struct Attributes { float coordinate; }
        struct ProceduralContext : rt::IHitContext
        {
            typealias TraceContext = ::TraceContext;
            typealias Payload = ::Payload;
            typealias Record = void;
            typealias Primitive = rt::BoundingBoxPrimitive<Attributes>;
        }
        struct ClosestHit : rt::IClosestHitShader
        {
            typealias Context = HitContext;
            static void invoke(rt::ClosestHitInput<Context> input)
            {
                input.payload.value = 1;
            }
        }
        struct HitGroup : rt::IHitGroup
        {
            typealias Context = HitContext;
            typealias ClosestHit = ::ClosestHit;
            typealias AnyHit = rt::NoAnyHit<Context>;
            typealias Intersection = rt::NoIntersection<Context>;
        }
        struct Intersection : rt::IIntersectionShader
        {
            typealias Context = ProceduralContext;
            static void invoke(rt::IntersectionInput<Context> input) {}
        }
        struct ProceduralGroup : rt::IHitGroup
        {
            typealias Context = ProceduralContext;
            typealias ClosestHit = rt::NoClosestHit<Context>;
            typealias AnyHit = rt::NoAnyHit<Context>;
            typealias Intersection = ::Intersection;
        }
        interface IRefinedMiss : rt::IMissShader {}
        struct Miss : IRefinedMiss
        {
            typealias Context = MissContext;
            static void invoke(rt::MissInput<Context> input)
            {
                input.payload.value = 2;
            }
        }
        struct UnrelatedMiss : rt::IMissShader
        {
            typealias Context = MissContext;
            static void invoke(rt::MissInput<Context> input)
            {
                input.payload.value = 3;
            }
        }
        struct Schema : rt::ITraceProgramSchema
        {
            typealias TraceContext = ::TraceContext;
            typealias HitGroups = rt::HitGroupList<HitGroup, ProceduralGroup>;
            typealias MissShaders = rt::MissShaderList<Miss>;
            typealias CallableShaders = rt::EmptyCallableShaderList;
        }
        rt::TraceProgramDescriptor<Schema> program;
        RaytracingAccelerationStructure scene;
        struct TraceEntry : rt::IMissShader
        {
            typealias Context = MissContext;
            static void invoke(rt::MissInput<Context> input)
            {
                rt::RayTracer<Schema> tracer;
                tracer.trace({}, scene, program, input.payload);
            }
        }
    )";

    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_METAL;
    targetDesc.profile = unitTestContext->slangGlobalSession->findProfile("metallib_2_4");
    slang::CompilerOptionEntry experimental = {};
    experimental.name = slang::CompilerOptionName::ExperimentalFeature;
    experimental.value.kind = slang::CompilerOptionValueKind::Int;
    experimental.value.intValue0 = 1;
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;
    sessionDesc.compilerOptionEntryCount = 1;
    sessionDesc.compilerOptionEntries = &experimental;
    const char* standardModulePath = SLANG_TEST_STANDARD_MODULE_PATH;
    sessionDesc.searchPathCount = 1;
    sessionDesc.searchPaths = &standardModulePath;
    ComPtr<slang::ISession> session;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
        unitTestContext->slangGlobalSession->createSession(sessionDesc, session.writeRef())));

    ComPtr<slang::IBlob> diagnostics;
    auto module = session->loadModuleFromSourceString(
        "schemaRetention",
        "schemaRetention.slang",
        source,
        diagnostics.writeRef());
    if (!module && diagnostics)
        getTestReporter()->message(
            TestMessageType::Info,
            static_cast<const char*>(diagnostics->getBufferPointer()));
    SLANG_CHECK_ABORT(module);

    // A separately compiled library carries the same dependency edges. Reload in a fresh
    // session so neither AST pointers nor cached source-module IR can supply missing metadata.
    ComPtr<slang::IBlob> serializedModule;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->serialize(serializedModule.writeRef())));
    ComPtr<slang::ISession> importingSession;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(unitTestContext->slangGlobalSession->createSession(
        sessionDesc,
        importingSession.writeRef())));
    module = importingSession->loadModuleFromIRBlob(
        "schemaRetention",
        "schemaRetention.slang-module",
        serializedModule,
        diagnostics.writeRef());
    SLANG_CHECK_ABORT(module);
    session = importingSession;

    ComPtr<slang::IEntryPoint> entryPoint;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->findAndCheckEntryPoint(
        "TraceEntry",
        SLANG_STAGE_MISS,
        entryPoint.writeRef(),
        diagnostics.writeRef())));
    slang::IComponentType* components[] = {module, entryPoint};
    ComPtr<slang::IComponentType> composite;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(session->createCompositeComponentType(
        components,
        SLANG_COUNT_OF(components),
        composite.writeRef(),
        diagnostics.writeRef())));
    ComPtr<slang::IComponentType> linkedProgram;
    SLANG_CHECK_ABORT(
        SLANG_SUCCEEDED(composite->link(linkedProgram.writeRef(), diagnostics.writeRef())));

    auto program = static_cast<ComponentType*>(linkedProgram.get());
    auto targetProgram = program->getTargetProgram(program->getLinkage()->targets[0]);
    DiagnosticSink sink;
    List<Index> entryPointIndices;
    entryPointIndices.add(0);
    CodeGenContext::Shared shared(targetProgram, entryPointIndices, &sink, nullptr);
    CodeGenContext codeGenContext(&shared);
    auto linkedIR = linkIR(&codeGenContext);
    SLANG_CHECK_ABORT(sink.getErrorCount() == 0);
    SLANG_CHECK_ABORT(linkedIR.module);
    replaceGlobalConstants(linkedIR.module);
    specializeModule(linkedIR.module, targetProgram, &sink, {});
    SLANG_CHECK_ABORT(sink.getErrorCount() == 0);
    eliminateDeadCode(linkedIR.module);

    IRTraceProgramDescriptorType* descriptorType = nullptr;
    for (auto inst : linkedIR.module->getModuleInst()->getChildren())
    {
        if (auto descriptor = as<IRTraceProgramDescriptorType>(inst))
        {
            SLANG_CHECK(!descriptorType);
            descriptorType = descriptor;
        }
        if (auto name = inst->findDecoration<IRNameHintDecoration>())
            SLANG_CHECK(name->getName().indexOf(toSlice("UnrelatedMiss")) < 0);
    }
    SLANG_CHECK_ABORT(descriptorType);
    PortableRT::Schema schema;
    SLANG_CHECK_ABORT(PortableRT::readSchema(descriptorType, schema));
    SLANG_CHECK(schema.type == descriptorType->getSchema());
    SLANG_CHECK_ABORT(schema.hitGroups.getCount() == 2);
    SLANG_CHECK(schema.missShaders.getCount() == 1);
    SLANG_CHECK(schema.callableShaders.getCount() == 0);
    SLANG_CHECK(schema.hitGroups[0].closestHit.isPresent);
    SLANG_CHECK(!schema.hitGroups[0].anyHit.isPresent);
    SLANG_CHECK(!schema.hitGroups[0].intersection.isPresent);
    SLANG_CHECK(!schema.hitGroups[1].closestHit.isPresent);
    SLANG_CHECK(!schema.hitGroups[1].anyHit.isPresent);
    SLANG_CHECK(schema.hitGroups[1].intersection.isPresent);
    SLANG_CHECK(schema.missShaders[0].isPresent);
    SLANG_CHECK(
        schema.hitGroups[0].closestHit.invoke->getParent() == linkedIR.module->getModuleInst());
    SLANG_CHECK(schema.missShaders[0].invoke->getParent() == linkedIR.module->getModuleInst());
}
