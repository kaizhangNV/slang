#include "core/slang-string.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

// The consumer knows neither the shader list nor its record types. Reuse its serialized IR
// with two providers, so each linked program must specialize the list projections using the
// provider's concrete schema. The callable index is dynamic and makes the record stride observable.
SLANG_UNIT_TEST(portableRayTracingExternSchemaProjections)
{
    const char* consumerSource = R"(
        import slang.raytracing.portable;

        extern struct ExternalSchema : rt::ITraceProgramSchema;
        struct DispatchContext : rt::ICallableContext
        {
            typealias TraceContext = ExternalSchema.TraceContext;
            typealias CallableData = uint;
            typealias Record = void;
        }
        rt::TraceProgramDescriptor<ExternalSchema> program;
        uniform uint callableIndex;
        RWStructuredBuffer<uint> output;

        [shader("raygeneration")]
        void RayGeneration()
        {
            uint data = 5;
            rt::RayTracer<ExternalSchema> tracer;
            tracer.callShader<DispatchContext>(callableIndex, program, data);
            output[0] = data;
        }
    )";
    const char* providerSource = R"(
        import portableExternalSchemaConsumer;
        import slang.raytracing.portable;

        struct TraceContext : rt::ITraceContext
        {
            typealias AccelerationStructure = rt::AccelerationStructure;
            typealias Motion = rt::NoMotion;
        }
        struct Record { uint4 values[RECORD_VECTOR_COUNT]; }
        struct CallableContext<R> : rt::ICallableContext
        {
            typealias TraceContext = ::TraceContext;
            typealias CallableData = uint;
            typealias Record = R;
        }
        RWStructuredBuffer<uint> stageOnlyOutput;
        [noinline]
        void stageOnlyBody() { stageOnlyOutput[0] = 37; }

        struct Callable<C : rt::ICallableContext> : rt::ICallableShader
        {
            typealias Context = C;
            static void invoke(rt::CallableInput<Context> input)
            {
                stageOnlyBody();
            }
        }
        struct ProviderSchema<S : rt::ICallableShader> : rt::ITraceProgramSchema
        {
            typealias TraceContext = ::TraceContext;
            typealias HitGroups = rt::HitGroupList<>;
            typealias MissShaders = rt::MissShaderList<>;
            typealias CallableShaders =
                rt::CallableShaderList<Callable<CallableContext<uint>>, S>;
        }
        export struct ExternalSchema : rt::ITraceProgramSchema =
            ProviderSchema<Callable<CallableContext<Record>>>;
    )";

    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_METAL;
    targetDesc.profile = unitTestContext->slangGlobalSession->findProfile("metallib_2_4");
    slang::CompilerOptionEntry options[2] = {};
    options[0].name = slang::CompilerOptionName::ExperimentalFeature;
    options[0].value.kind = slang::CompilerOptionValueKind::Int;
    options[0].value.intValue0 = 1;
    options[1].name = slang::CompilerOptionName::ValidateIr;
    options[1].value.kind = slang::CompilerOptionValueKind::Int;
    options[1].value.intValue0 = 1;
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;
    sessionDesc.compilerOptionEntryCount = SLANG_COUNT_OF(options);
    sessionDesc.compilerOptionEntries = options;

    ComPtr<slang::ISession> compilingSession;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(unitTestContext->slangGlobalSession->createSession(
        sessionDesc,
        compilingSession.writeRef())));
    ComPtr<slang::IBlob> diagnostics;
    auto consumer = compilingSession->loadModuleFromSourceString(
        "portableExternalSchemaConsumer",
        "portableExternalSchemaConsumer.slang",
        consumerSource,
        diagnostics.writeRef());
    if (!consumer && diagnostics)
        getTestReporter()->message(
            TestMessageType::Info,
            static_cast<const char*>(diagnostics->getBufferPointer()));
    SLANG_CHECK_ABORT(consumer);
    ComPtr<slang::IBlob> serializedConsumer;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(consumer->serialize(serializedConsumer.writeRef())));

    const char* recordVectorCounts[] = {"1", "3"};
    for (auto recordVectorCount : recordVectorCounts)
    {
        slang::PreprocessorMacroDesc recordSize = {"RECORD_VECTOR_COUNT", recordVectorCount};
        sessionDesc.preprocessorMacroCount = 1;
        sessionDesc.preprocessorMacros = &recordSize;
        ComPtr<slang::ISession> importingSession;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(unitTestContext->slangGlobalSession->createSession(
            sessionDesc,
            importingSession.writeRef())));
        consumer = importingSession->loadModuleFromIRBlob(
            "portableExternalSchemaConsumer",
            "portableExternalSchemaConsumer.slang-module",
            serializedConsumer,
            diagnostics.writeRef());
        SLANG_CHECK_ABORT(consumer);
        auto provider = importingSession->loadModuleFromSourceString(
            "portableExternalSchemaProvider",
            "portableExternalSchemaProvider.slang",
            providerSource,
            diagnostics.writeRef());
        if (!provider && diagnostics)
            getTestReporter()->message(
                TestMessageType::Info,
                static_cast<const char*>(diagnostics->getBufferPointer()));
        SLANG_CHECK_ABORT(provider);

        ComPtr<slang::IEntryPoint> entry;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(consumer->findAndCheckEntryPoint(
            "RayGeneration",
            SLANG_STAGE_RAY_GENERATION,
            entry.writeRef(),
            diagnostics.writeRef())));
        slang::IComponentType* components[] = {consumer, provider, entry};
        ComPtr<slang::IComponentType> composite;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(importingSession->createCompositeComponentType(
            components,
            SLANG_COUNT_OF(components),
            composite.writeRef(),
            diagnostics.writeRef())));
        ComPtr<slang::IComponentType> linked;
        SLANG_CHECK_ABORT(
            SLANG_SUCCEEDED(composite->link(linked.writeRef(), diagnostics.writeRef())));
        ComPtr<slang::IBlob> code;
        auto result = linked->getEntryPointCode(0, 0, code.writeRef(), diagnostics.writeRef());
        if (SLANG_FAILED(result) && diagnostics)
            getTestReporter()->message(
                TestMessageType::Info,
                static_cast<const char*>(diagnostics->getBufferPointer()));
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(result));
        SLANG_CHECK_ABORT(code && code->getBufferSize());
        String text(UnownedStringSlice(
            static_cast<const char*>(code->getBufferPointer()),
            code->getBufferSize()));
        SLANG_CHECK(text.indexOf("[[kernel]]") >= 0);
        SLANG_CHECK(text.indexOf("callableFunctions") >= 0);
        SLANG_CHECK(text.indexOf("_slang_callable_record") >= 0);
        // The common record stride includes a 16-byte header and the largest record in the
        // two-element list: 16 or 48 bytes.
        SLANG_CHECK(text.indexOf(recordVectorCount[0] == '1' ? " * 32U" : " * 64U") >= 0);
        // The resource layout is shared with independently compiled stages, so its fields
        // may include stageOnlyOutput. The noinline implementation itself must remain absent.
        SLANG_CHECK(text.indexOf("stageOnlyBody") < 0);
        SLANG_CHECK(text.indexOf("[[visible]]") < 0);
        SLANG_CHECK(text.indexOf("metalRTCallShader") < 0);

        ComPtr<slang::IMetadata> metadata;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
            linked->getEntryPointMetadata(0, 0, metadata.writeRef(), diagnostics.writeRef())));
        auto physical = static_cast<slang::IMetalRayTracingMetadata*>(
            metadata->castAs(slang::IMetalRayTracingMetadata::getTypeGuid()));
        SLANG_CHECK_ABORT(physical && physical->getProgramCount() == 1);
        slang::MetalRayTracingProgramInfo programInfo;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(physical->getProgramInfo(0, &programInfo)));
        SLANG_CHECK(programInfo.functionCount == 0);
        SLANG_CHECK(programInfo.callableRecordStride == (recordVectorCount[0] == '1' ? 32u : 64u));
        // The descriptor ABI always has seven fields. Empty hit/miss projections instead
        // determine zero function-table lengths; only the two callable slots need storage.
        SLANG_CHECK(programInfo.resourceCount == 7);
        for (uint32_t i = 0; i < programInfo.resourceCount; ++i)
        {
            slang::MetalRayTracingResourceInfo resource;
            SLANG_CHECK_ABORT(SLANG_SUCCEEDED(physical->getProgramResourceInfo(0, i, &resource)));
            SLANG_CHECK(
                resource.functionTableSize ==
                (resource.kind == slang::MetalRayTracingResourceKind::CallableVisibleFunctionTable
                     ? 2u
                     : 0u));
        }
    }
}
