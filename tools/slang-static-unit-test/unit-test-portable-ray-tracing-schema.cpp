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

// Exercise the source producer, linker, specialization, and DCE boundary with and without an
// IR-module roundtrip. Reached stage contracts remain available, while unselected invoke bodies
// and their observable side effects must never become dependencies of the trace entry.
static void _testSchemaReading(UnitTestContext* unitTestContext, bool serializeModule)
{
    const char* source = R"(
        import slang.raytracing.portable;

        RWStructuredBuffer<uint> stageBodyOutput;
        [noinline]
        void referencedStageBody()
        {
            stageBodyOutput[0] = 42;
        }

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
            typealias Record = uint;
        }
        struct CallableContext : rt::ICallableContext
        {
            typealias TraceContext = ::TraceContext;
            typealias CallableData = float2;
            typealias Record = float4;
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
                referencedStageBody();
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
            static void invoke(rt::IntersectionInput<Context> input)
            {
                referencedStageBody();
            }
        }
        struct ProceduralGroup : rt::IHitGroup
        {
            typealias Context = ProceduralContext;
            typealias ClosestHit = rt::NoClosestHit<Context>;
            typealias AnyHit = rt::NoAnyHit<Context>;
            typealias Intersection = ::Intersection;
        }
        struct OpaqueGroup : rt::IHitGroup
        {
            typealias Context = HitContext;
            typealias ClosestHit = rt::NoClosestHit<Context>;
            typealias AnyHit = rt::NoAnyHit<Context>;
            typealias Intersection = rt::NoIntersection<Context>;
        }
        interface IRefinedMiss : rt::IMissShader {}
        struct Miss : IRefinedMiss
        {
            typealias Context = MissContext;
            static void invoke(rt::MissInput<Context> input)
            {
                referencedStageBody();
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
        struct Callable : rt::ICallableShader
        {
            typealias Context = CallableContext;
            static void invoke(rt::CallableInput<Context> input)
            {
                referencedStageBody();
                input.data += input.record.xy;
            }
        }
        struct Schema : rt::ITraceProgramSchema
        {
            typealias TraceContext = ::TraceContext;
            typealias HitGroups = rt::HitGroupList<HitGroup, ProceduralGroup, OpaqueGroup>;
            typealias MissShaders = rt::MissShaderList<Miss>;
            typealias CallableShaders = rt::CallableShaderList<Callable>;
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

    if (serializeModule)
    {
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
    }

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
        {
            SLANG_CHECK(name->getName().indexOf(toSlice("UnrelatedMiss")) < 0);
            SLANG_CHECK(name->getName().indexOf(toSlice("referencedStageBody")) < 0);
        }
        if (auto witness = as<IRWitnessTable>(inst))
        {
            auto interfaceType = getResolvedInstForDecorations(witness->getConformanceType());
            if (auto contract =
                    interfaceType->findDecoration<IRPortableRTStageContractDecoration>())
                SLANG_CHECK(!findWitnessTableEntry(witness, contract->getInvokeKey()));
        }
    }
    SLANG_CHECK_ABORT(descriptorType);
    PortableRT::Schema schema;
    SLANG_CHECK_ABORT(PortableRT::readSchemaContract(descriptorType, schema));
    SLANG_CHECK(schema.type == descriptorType->getSchema());
    SLANG_CHECK_ABORT(schema.hitGroups.getCount() == 3);
    SLANG_CHECK_ABORT(schema.missShaders.getCount() == 1);
    SLANG_CHECK_ABORT(schema.callableShaders.getCount() == 1);
    SLANG_CHECK(schema.hitGroups[0].closestHit.isPresent);
    SLANG_CHECK(!schema.hitGroups[0].anyHit.isPresent);
    SLANG_CHECK(!schema.hitGroups[0].intersection.isPresent);
    SLANG_CHECK(!schema.hitGroups[1].closestHit.isPresent);
    SLANG_CHECK(!schema.hitGroups[1].anyHit.isPresent);
    SLANG_CHECK(schema.hitGroups[1].intersection.isPresent);
    auto& opaque = schema.hitGroups[2];
    SLANG_CHECK(!opaque.closestHit.isPresent);
    SLANG_CHECK(!opaque.anyHit.isPresent);
    SLANG_CHECK(!opaque.intersection.isPresent);
    SLANG_CHECK(opaque.payloadType == schema.hitGroups[0].closestHit.payloadType);
    SLANG_CHECK(opaque.recordType == schema.hitGroups[0].closestHit.recordType);
    SLANG_CHECK(opaque.primitiveType == schema.hitGroups[0].closestHit.primitiveType);
    SLANG_CHECK(opaque.attributesType == schema.hitGroups[0].closestHit.attributesType);
    SLANG_CHECK(schema.missShaders[0].isPresent);
    SLANG_CHECK(schema.missShaders[0].payloadType == opaque.payloadType);
    SLANG_CHECK(as<IRUIntType>(schema.missShaders[0].recordType));
    SLANG_CHECK(!schema.missShaders[0].primitiveType);
    SLANG_CHECK(!schema.missShaders[0].attributesType);
    SLANG_CHECK(schema.callableShaders[0].isPresent);
    auto callableData = as<IRVectorType>(schema.callableShaders[0].payloadType);
    SLANG_CHECK_ABORT(callableData);
    SLANG_CHECK(as<IRFloatType>(callableData->getElementType()));
    SLANG_CHECK(as<IRIntLit>(callableData->getElementCount())->getValue() == 2);
    auto callableRecord = as<IRVectorType>(schema.callableShaders[0].recordType);
    SLANG_CHECK_ABORT(callableRecord);
    SLANG_CHECK(as<IRFloatType>(callableRecord->getElementType()));
    SLANG_CHECK(as<IRIntLit>(callableRecord->getElementCount())->getValue() == 4);
    SLANG_CHECK(!schema.callableShaders[0].primitiveType);
    SLANG_CHECK(!schema.callableShaders[0].attributesType);

    List<const PortableRT::SchemaStage*> stages;
    for (auto& group : schema.hitGroups)
    {
        const PortableRT::SchemaStage* groupStages[] = {
            &group.closestHit,
            &group.anyHit,
            &group.intersection,
        };
        for (auto stage : groupStages)
        {
            SLANG_CHECK(stage->contextType == group.contextType);
            SLANG_CHECK(stage->payloadType == group.payloadType);
            SLANG_CHECK(stage->recordType == group.recordType);
            SLANG_CHECK(stage->primitiveType == group.primitiveType);
            SLANG_CHECK(stage->attributesType == group.attributesType);
            stages.add(stage);
        }
    }
    stages.add(&schema.missShaders[0]);
    stages.add(&schema.callableShaders[0]);
    for (auto stage : stages)
    {
        SLANG_CHECK(stage->witness);
        SLANG_CHECK(stage->contextWitness);
        SLANG_CHECK(!stage->invoke);
        SLANG_CHECK(!stage->input);
    }
}

SLANG_UNIT_TEST(portableRayTracingSchemaRetainsOnlyContract)
{
    _testSchemaReading(unitTestContext, false);
}

SLANG_UNIT_TEST(portableRayTracingSchemaContractWithoutInvocations)
{
    _testSchemaReading(unitTestContext, true);
}
