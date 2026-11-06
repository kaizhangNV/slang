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
// IR-module roundtrip. Associated-type projections must resolve to concrete packs, while
// unselected invoke bodies and their side effects never become dependencies of the trace entry.
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
        // One concrete type supplies both contracts. Ordinary lookup expressions must use
        // the declared conformance even when the concrete type also implements another interface.
        struct TraceContext : rt::ITraceContext, rt::IPayloadContext
        {
            typealias AccelerationStructure = rt::AccelerationStructure;
            typealias Motion = rt::NoMotion;
            typealias TraceContext = ::TraceContext;
            typealias Payload = ::Payload;
            typealias Record = uint;
        }
        struct HitContext : rt::IHitContext
        {
            typealias TraceContext = ::TraceContext;
            typealias Payload = ::Payload;
            typealias Record = void;
            typealias Primitive = rt::TrianglePrimitive;
        }
        typealias MissContext = TraceContext;
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
        // Both group members name this type. Projection preserves their separate stage slots
        // without retaining either method implementation.
        struct SharedHit : rt::IClosestHitShader, rt::IAnyHitShader
        {
            typealias Context = HitContext;
            static void invoke(rt::ClosestHitInput<Context> input)
            {
                referencedStageBody();
                input.payload.value = 1;
            }
            static void invoke(rt::AnyHitInput<Context> input)
            {
                referencedStageBody();
                input.payload.value = 4;
            }
        }
        struct HitGroup : rt::IHitGroup
        {
            typealias Context = HitContext;
            typealias ClosestHit = SharedHit;
            typealias AnyHit = SharedHit;
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

        // This descriptor is first lowered with an unknown schema. Specializing the helper
        // must reuse the concrete descriptor type above without losing its projected types.
        [noinline]
        rt::TraceProgramDescriptor<S> copyDescriptor<S : rt::ITraceProgramSchema>(
            rt::TraceProgramDescriptor<S> descriptor)
        {
            return descriptor;
        }

        struct TraceEntry : rt::IMissShader
        {
            typealias Context = MissContext;
            static void invoke(rt::MissInput<Context> input)
            {
                rt::RayTracer<Schema> tracer;
                tracer.trace({}, scene, program, input.payload);
                tracer.trace({}, scene, copyDescriptor<Schema>(program), input.payload);
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
    IRFunc* descriptorCopy = nullptr;
    for (auto inst : linkedIR.module->getModuleInst()->getChildren())
    {
        if (auto descriptor = as<IRTraceProgramDescriptorType>(inst))
        {
            SLANG_CHECK(!descriptorType);
            descriptorType = descriptor;
        }
        if (auto name = inst->findDecoration<IRNameHintDecoration>())
        {
            if (name->getName() == toSlice("copyDescriptor"))
                if (auto function = as<IRFunc>(inst))
                    descriptorCopy = function;
            SLANG_CHECK(name->getName().indexOf(toSlice("UnrelatedMiss")) < 0);
            // Every schema stage calls this noinline helper. A retained invoke body would
            // therefore also keep the helper and its output resource alive.
            SLANG_CHECK(name->getName().indexOf(toSlice("referencedStageBody")) < 0);
            SLANG_CHECK(name->getName().indexOf(toSlice("stageBodyOutput")) < 0);
        }
    }
    SLANG_CHECK_ABORT(descriptorType);
    SLANG_CHECK_ABORT(descriptorCopy);
    auto descriptorCopyType = as<IRFuncType>(descriptorCopy->getDataType());
    SLANG_CHECK_ABORT(descriptorCopyType);
    SLANG_CHECK(descriptorCopyType->getResultType() == descriptorType);
    SLANG_CHECK_ABORT(descriptorCopyType->getParamCount() == 1);
    SLANG_CHECK(descriptorCopyType->getParamType(0) == descriptorType);
    UInt programDecorationCount = 0;
    for (auto decoration : descriptorType->getDecorations())
        if (as<IRPortableRTProgramDecoration>(decoration))
            ++programDecorationCount;
    SLANG_CHECK(programDecorationCount == 1);
    auto programTypes = descriptorType->findDecoration<IRPortableRTProgramDecoration>();
    SLANG_CHECK_ABORT(programTypes);
    auto hitList = getResolvedInstForDecorations(programTypes->getHitGroupsType())
                       ->findDecoration<IRPortableRTShaderListDecoration>();
    auto missList = getResolvedInstForDecorations(programTypes->getMissShadersType())
                        ->findDecoration<IRPortableRTShaderListDecoration>();
    auto callableList = getResolvedInstForDecorations(programTypes->getCallableShadersType())
                            ->findDecoration<IRPortableRTShaderListDecoration>();
    SLANG_CHECK_ABORT(hitList && missList && callableList);

    // Every hit-group projection has one result per source group, including the group whose
    // three stages are absent. No expansion or witness lookup may remain for the reader to run.
    for (UInt operandIndex = 0; operandIndex < hitList->getOperandCount(); ++operandIndex)
    {
        auto pack = as<IRTypePack>(hitList->getOperand(operandIndex));
        SLANG_CHECK_ABORT(pack);
        SLANG_CHECK(pack->getOperandCount() == 3);
    }
    auto closestHitTypes = as<IRTypePack>(hitList->getClosestHitTypes());
    auto anyHitTypes = as<IRTypePack>(hitList->getAnyHitTypes());
    SLANG_CHECK(closestHitTypes->getOperand(0) == anyHitTypes->getOperand(0));
    for (auto list : {missList, callableList})
    {
        auto entryTypes = as<IRTypePack>(list->getTypepack());
        auto recordTypes = as<IRTypePack>(list->getRecordTypes());
        SLANG_CHECK_ABORT(entryTypes && recordTypes);
        SLANG_CHECK(entryTypes->getOperandCount() == 1);
        SLANG_CHECK(recordTypes->getOperandCount() == 1);
        IRInst* hitOnlyProjections[] = {
            list->getPayloadTypes(),
            list->getPrimitiveTypes(),
            list->getAttributesTypes(),
            list->getClosestHitTypes(),
            list->getAnyHitTypes(),
            list->getIntersectionTypes(),
        };
        for (auto projection : hitOnlyProjections)
        {
            auto pack = as<IRTypePack>(projection);
            SLANG_CHECK_ABORT(pack);
            SLANG_CHECK(pack->getOperandCount() == 0);
        }
    }

    PortableRT::Schema schema;
    PortableRT::readSchemaContract(descriptorType, schema);
    SLANG_CHECK(schema.accelerationStructure == programTypes->getAccelerationStructureType());
    SLANG_CHECK(schema.motion == programTypes->getMotionType());
    SLANG_CHECK_ABORT(schema.hitGroups.getCount() == 3);
    SLANG_CHECK_ABORT(schema.missShaders.getCount() == 1);
    SLANG_CHECK_ABORT(schema.callableShaders.getCount() == 1);
    SLANG_CHECK(schema.hitGroups[0].closestHit.isPresent);
    SLANG_CHECK(schema.hitGroups[0].closestHit.stage == Stage::ClosestHit);
    SLANG_CHECK(schema.hitGroups[0].anyHit.isPresent);
    SLANG_CHECK(schema.hitGroups[0].anyHit.stage == Stage::AnyHit);
    SLANG_CHECK(!schema.hitGroups[0].intersection.isPresent);
    SLANG_CHECK(!schema.hitGroups[1].closestHit.isPresent);
    SLANG_CHECK(!schema.hitGroups[1].anyHit.isPresent);
    SLANG_CHECK(schema.hitGroups[1].intersection.isPresent);
    SLANG_CHECK(schema.hitGroups[1].intersection.stage == Stage::Intersection);
    SLANG_CHECK(schema.hitGroups[1].primitiveType != schema.hitGroups[0].primitiveType);
    SLANG_CHECK(schema.hitGroups[1].attributesType != schema.hitGroups[0].attributesType);
    auto& opaque = schema.hitGroups[2];
    SLANG_CHECK(!opaque.closestHit.isPresent);
    SLANG_CHECK(!opaque.anyHit.isPresent);
    SLANG_CHECK(!opaque.intersection.isPresent);
    SLANG_CHECK(opaque.payloadType == schema.hitGroups[0].payloadType);
    SLANG_CHECK(opaque.recordType == schema.hitGroups[0].recordType);
    SLANG_CHECK(opaque.primitiveType == schema.hitGroups[0].primitiveType);
    SLANG_CHECK(opaque.attributesType == schema.hitGroups[0].attributesType);
    SLANG_CHECK(schema.missShaders[0].isPresent);
    SLANG_CHECK(as<IRUIntType>(schema.missShaders[0].recordType));
    SLANG_CHECK(schema.callableShaders[0].isPresent);
    auto callableRecord = as<IRVectorType>(schema.callableShaders[0].recordType);
    SLANG_CHECK_ABORT(callableRecord);
    SLANG_CHECK(as<IRFloatType>(callableRecord->getElementType()));
    SLANG_CHECK(as<IRIntLit>(callableRecord->getElementCount())->getValue() == 4);

    List<const PortableRT::SchemaStage*> stages;
    for (auto& group : schema.hitGroups)
    {
        stages.add(&group.closestHit);
        stages.add(&group.anyHit);
        stages.add(&group.intersection);
    }
    stages.add(&schema.missShaders[0]);
    stages.add(&schema.callableShaders[0]);
    for (auto stage : stages)
    {
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
