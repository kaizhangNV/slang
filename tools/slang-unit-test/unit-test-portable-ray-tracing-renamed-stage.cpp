#include "core/slang-io.h"
#include "core/slang-string.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

// Renaming a selected export must not rename the source implementation that a schema references.
// On Metal the selected stage owns the visible adapter and its export name. Ray generation
// uses the schema's fixed table contract without emitting a second stage implementation.
SLANG_UNIT_TEST(portableRayTracingRenamedStageAndSchema)
{
    const char* path = "tests/ray-tracing-2/target/portable/metal-standalone-shared-stage.slang";
    String source;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(File::readAllText(path, source)));

    slang::TargetDesc targets[3] = {};
    targets[0].format = SLANG_HLSL;
    targets[0].profile = unitTestContext->slangGlobalSession->findProfile("lib_6_6");
    targets[1].format = SLANG_SPIRV_ASM;
    targets[1].profile = unitTestContext->slangGlobalSession->findProfile("spirv_1_5");
    targets[2].format = SLANG_METAL;
    targets[2].profile = unitTestContext->slangGlobalSession->findProfile("metallib_2_4");
    slang::CompilerOptionEntry experimental = {};
    experimental.name = slang::CompilerOptionName::ExperimentalFeature;
    experimental.value.kind = slang::CompilerOptionValueKind::Int;
    experimental.value.intValue0 = 1;
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = SLANG_COUNT_OF(targets);
    sessionDesc.targets = targets;
    sessionDesc.compilerOptionEntryCount = 1;
    sessionDesc.compilerOptionEntries = &experimental;
    ComPtr<slang::ISession> session;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
        unitTestContext->slangGlobalSession->createSession(sessionDesc, session.writeRef())));
    ComPtr<slang::IBlob> diagnostics;
    auto module = session->loadModuleFromSourceString(
        "renamedPortableStage",
        path,
        source.getBuffer(),
        diagnostics.writeRef());
    SLANG_CHECK_ABORT(module);
    ComPtr<slang::IEntryPoint> raygen;
    ComPtr<slang::IEntryPoint> selected;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->findAndCheckEntryPoint(
        "RayGeneration",
        SLANG_STAGE_RAY_GENERATION,
        raygen.writeRef(),
        diagnostics.writeRef())));
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->findAndCheckEntryPoint(
        "SharedHit",
        SLANG_STAGE_CLOSEST_HIT,
        selected.writeRef(),
        diagnostics.writeRef())));
    ComPtr<slang::IComponentType> renamed;
    SLANG_CHECK_ABORT(
        SLANG_SUCCEEDED(selected->renameEntryPoint("renamedPortableHit", renamed.writeRef())));
    slang::IComponentType* components[] = {module, raygen, renamed};
    ComPtr<slang::IComponentType> composite;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(session->createCompositeComponentType(
        components,
        SLANG_COUNT_OF(components),
        composite.writeRef(),
        diagnostics.writeRef())));
    ComPtr<slang::IComponentType> linked;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(composite->link(linked.writeRef(), diagnostics.writeRef())));

    for (SlangInt targetIndex = 0; targetIndex < SLANG_COUNT_OF(targets); ++targetIndex)
    {
        auto layout = linked->getLayout(targetIndex, diagnostics.writeRef());
        SLANG_CHECK_ABORT(layout && layout->getEntryPointCount() == 2);
        auto entry = layout->getEntryPointByIndex(1);
        SLANG_CHECK(entry->getStage() == SLANG_STAGE_CLOSEST_HIT);
        SLANG_CHECK(String(entry->getName()) == "SharedHit");
        SLANG_CHECK(String(entry->getNameOverride()) == "renamedPortableHit");
        auto schema = layout->findTraceProgramSchema("Schema");
        SLANG_CHECK_ABORT(schema && schema->getHitGroupCount() == 1);
        auto stage = schema->getHitGroup(0)->getClosestHit();
        SLANG_CHECK_ABORT(stage);
        // Source schema reflection reports the default native name, not a selection-specific
        // override. Metal schema reflection has no selected export name: the selected component
        // and its compiled artifact metadata own that name.
        const bool isMetal = targets[targetIndex].format == SLANG_METAL;
        if (isMetal)
        {
            SLANG_CHECK(stage->getEntryPointName() == nullptr);
        }
        else
        {
            SLANG_CHECK(String(stage->getEntryPointName()) == "SharedHit_closesthit");
        }

        ComPtr<slang::IBlob> code;
        auto result = linked->getTargetCode(targetIndex, code.writeRef(), diagnostics.writeRef());
        if (SLANG_FAILED(result) && diagnostics)
            getTestReporter()->message(
                TestMessageType::Info,
                static_cast<const char*>(diagnostics->getBufferPointer()));
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(result) && code);
        auto emitted = UnownedStringSlice(
            static_cast<const char*>(code->getBufferPointer()),
            code->getBufferSize());
        SLANG_CHECK(emitted.indexOf(UnownedStringSlice("renamedPortableHit")) >= 0);
        if (targets[targetIndex].format == SLANG_METAL)
        {
            SLANG_CHECK(
                emitted.indexOf(UnownedStringSlice("[[visible]] void renamedPortableHit(")) >= 0);
            SLANG_CHECK(
                emitted.indexOf(UnownedStringSlice("__slang_portable_rt_0_closesthit_0")) < 0);
            SLANG_CHECK(emitted.indexOf(UnownedStringSlice("addInstanceID")) >= 0);
        }
        else if (targets[targetIndex].format == SLANG_SPIRV_ASM)
        {
            SLANG_CHECK(emitted.indexOf(UnownedStringSlice("OpEntryPoint ClosestHitKHR")) >= 0);
            SLANG_CHECK(emitted.indexOf(UnownedStringSlice("\"renamedPortableHit\"")) >= 0);
        }
        else
        {
            SLANG_CHECK(emitted.indexOf(UnownedStringSlice("[shader(\"closesthit\")]")) >= 0);
        }
        SLANG_CHECK(String(entry->getNameOverride()) == "renamedPortableHit");
        if (isMetal)
        {
            SLANG_CHECK(stage->getEntryPointName() == nullptr);
        }
        else
        {
            SLANG_CHECK(String(stage->getEntryPointName()) == "SharedHit_closesthit");
        }
    }
}
