#include "compiler-core/slang-diagnostic-sink.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "slang/slang-entry-point.h"
#include "slang/slang-ir-insts.h"
#include "slang/slang-session.h"
#include "slang/slang-target-program.h"
#include "unit-test/slang-unit-test.h"

#include <spirv/unified1/spirv.h>

using namespace Slang;

namespace
{

enum class EntryPointSelectionMode
{
    SpecializedName,
    SpecializeAPI,
};

// Include compiler diagnostics when a setup or code-generation check fails.
static void _reportDiagnostics(slang::IBlob* diagnostics)
{
    if (diagnostics)
        getTestReporter()->message(
            TestMessageType::Info,
            static_cast<const char*>(diagnostics->getBufferPointer()));
}

// Check the emitted module version after IR linking and SPIR-V code generation.
static void _checkEmittedSPIRVVersion(slang::IComponentType* program, uint32_t minimumVersion)
{
    ComPtr<slang::IBlob> diagnostics;
    ComPtr<slang::IComponentType> linkedProgram;
    auto linkResult = program->link(linkedProgram.writeRef(), diagnostics.writeRef());
    if (SLANG_FAILED(linkResult))
        _reportDiagnostics(diagnostics);
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(linkResult));
    ComPtr<slang::IBlob> code;
    auto codeResult = linkedProgram->getTargetCode(0, code.writeRef(), diagnostics.writeRef());
    if (SLANG_FAILED(codeResult))
        _reportDiagnostics(diagnostics);
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(codeResult));
    SLANG_CHECK_ABORT(code != nullptr);
    SLANG_CHECK_ABORT(code->getBufferSize() >= 5 * sizeof(uint32_t));
    auto words = static_cast<const uint32_t*>(code->getBufferPointer());
    SLANG_CHECK_ABORT(words[0] == SpvMagicNumber);
    SLANG_CHECK(words[1] >= minimumVersion);
}

// Check that two selections of main<T> retain their own requirements in layout IR.
// The API case specializes one shared EntryPoint, so neither its pointer nor its FuncDecl can
// distinguish the two selections. The names of the IR imports identify the selected functions.
static void _checkEntryPointCapabilityLayout(
    EntryPointSelectionMode selectionMode,
    bool checkLinkedVersions = false)
{
    const char* source = R"(
        [require(_spirv_1_5)]
        struct First {}
        [require(_spirv_1_6)]
        struct Second {}

        void helper<T>() {}

        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main<T>() { helper<T>(); }
    )";

    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK_ABORT(
        SLANG_SUCCEEDED(slang_createGlobalSession(SLANG_API_VERSION, globalSession.writeRef())));

    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile =
        globalSession->findProfile(checkLinkedVersions ? "spirv_1_0" : "spirv_1_6");
    slang::CompilerOptionEntry targetOptions[2] = {};
    targetOptions[0].name = slang::CompilerOptionName::Optimization;
    targetOptions[0].value.kind = slang::CompilerOptionValueKind::Int;
    targetOptions[0].value.intValue0 = SLANG_OPTIMIZATION_LEVEL_NONE;
    targetOptions[1].name = slang::CompilerOptionName::SkipSPIRVValidation;
    targetOptions[1].value.kind = slang::CompilerOptionValueKind::Int;
    targetOptions[1].value.intValue0 = 1;
    targetDesc.compilerOptionEntries = targetOptions;
    targetDesc.compilerOptionEntryCount = SLANG_COUNT_OF(targetOptions);
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;
    ComPtr<slang::ISession> session;
    SLANG_CHECK_ABORT(
        SLANG_SUCCEEDED(globalSession->createSession(sessionDesc, session.writeRef())));

    ComPtr<slang::IBlob> diagnostics;
    auto module = session->loadModuleFromSourceString(
        "EntryPointCapabilityLayout",
        "EntryPointCapabilityLayout.slang",
        source,
        diagnostics.writeRef());
    SLANG_CHECK_ABORT(module != nullptr);

    ComPtr<slang::IComponentType> selectedEntries[2];
    ComponentType* entries[2];
    if (selectionMode == EntryPointSelectionMode::SpecializeAPI)
    {
        ComPtr<slang::IEntryPoint> baseEntry;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->findAndCheckEntryPoint(
            "main",
            SLANG_STAGE_COMPUTE,
            baseEntry.writeRef(),
            diagnostics.writeRef())));
        SLANG_CHECK_ABORT(baseEntry->getSpecializationParamCount() == 1);
        auto internalBase = static_cast<EntryPoint*>(baseEntry.get());
        auto baseRequirements = internalBase->getInferredCapabilityRequirements();

        auto moduleLayout = module->getLayout();
        SLANG_CHECK_ABORT(moduleLayout != nullptr);
        const char* typeNames[] = {"First", "Second"};
        for (Index i = 0; i < SLANG_COUNT_OF(selectedEntries); ++i)
        {
            auto type = moduleLayout->findTypeByName(typeNames[i]);
            SLANG_CHECK_ABORT(type != nullptr);
            auto argument = slang::SpecializationArg::fromType(type);
            SLANG_CHECK_ABORT(SLANG_SUCCEEDED(baseEntry->specialize(
                &argument,
                1,
                selectedEntries[i].writeRef(),
                diagnostics.writeRef())));

            auto internalEntry = static_cast<ComponentType*>(selectedEntries[i].get());
            entries[i] = internalEntry;
            SLANG_CHECK_ABORT(
                internalEntry->getEntryPoint(0).get() == static_cast<EntryPoint*>(baseEntry.get()));
            SLANG_CHECK(internalBase->getInferredCapabilityRequirements() == baseRequirements);
        }
    }
    else
    {
        const char* selections[] = {"main<First>", "main<Second>"};
        for (Index i = 0; i < SLANG_COUNT_OF(selectedEntries); ++i)
        {
            ComPtr<slang::IEntryPoint> selectedEntry;
            SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->findAndCheckEntryPoint(
                selections[i],
                SLANG_STAGE_COMPUTE,
                selectedEntry.writeRef(),
                diagnostics.writeRef())));
            selectedEntries[i] = selectedEntry;
            entries[i] = static_cast<EntryPoint*>(selectedEntry.get());
        }
    }

    SLANG_CHECK_ABORT(entries[0] != entries[1]);
    SLANG_CHECK_ABORT(
        entries[0]->getEntryPoint(0)->getFuncDecl() == entries[1]->getEntryPoint(0)->getFuncDecl());
    if (selectionMode == EntryPointSelectionMode::SpecializeAPI)
    {
        SLANG_CHECK_ABORT(entries[0]->getSpecializationParamCount() == 0);
        SLANG_CHECK_ABORT(entries[1]->getSpecializationParamCount() == 0);
    }

    String mangledNames[] = {
        entries[0]->getEntryPointMangledName(0),
        entries[1]->getEntryPointMangledName(0)};
    SLANG_CHECK_ABORT(mangledNames[0] != mangledNames[1]);
    auto linkage = entries[0]->getLinkage();

    if (checkLinkedVersions)
    {
        // Both specializations originate from main<T>; give their exported entries distinct names.
        const char* entryNames[] = {"firstMain", "secondMain"};
        for (Index i = 0; i < SLANG_COUNT_OF(selectedEntries); ++i)
        {
            ComPtr<slang::IComponentType> renamedEntry;
            SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
                selectedEntries[i]->renameEntryPoint(entryNames[i], renamedEntry.writeRef())));
            selectedEntries[i] = renamedEntry;
        }
        const uint32_t requiredVersions[] = {0x00010500, 0x00010600};
        for (Index i = 0; i < SLANG_COUNT_OF(selectedEntries); ++i)
        {
            slang::IComponentType* components[] = {module, selectedEntries[i].get()};
            ComPtr<slang::IComponentType> program;
            SLANG_CHECK_ABORT(SLANG_SUCCEEDED(session->createCompositeComponentType(
                components,
                SLANG_COUNT_OF(components),
                program.writeRef(),
                diagnostics.writeRef())));
            _checkEmittedSPIRVVersion(program, requiredVersions[i]);
        }
    }

    // Check both component orders. Each imported function must keep its own requirement,
    // regardless of which entry a declaration-keyed dictionary would encounter last.
    for (Index order = 0; order < 2; ++order)
    {
        slang::IComponentType* components[] = {
            module,
            selectedEntries[order].get(),
            selectedEntries[1 - order].get()};
        ComPtr<slang::IComponentType> program;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(session->createCompositeComponentType(
            components,
            SLANG_COUNT_OF(components),
            program.writeRef(),
            diagnostics.writeRef())));

        auto internalProgram = static_cast<ComponentType*>(program.get());
        auto targetProgram = internalProgram->getTargetProgram(linkage->targets[0]);
        DiagnosticSink sink;
        auto layoutIR = targetProgram->getOrCreateIRModuleForLayout(&sink);
        SLANG_CHECK_ABORT(layoutIR != nullptr);
        SLANG_CHECK_ABORT(sink.getErrorCount() == 0);

        for (Index entryIndex = 0; entryIndex < 2; ++entryIndex)
        {
            IRInst* entryInst = nullptr;
            for (auto inst : layoutIR->getGlobalInsts())
            {
                auto linkageDecoration = inst->findDecoration<IRLinkageDecoration>();
                if (linkageDecoration && linkageDecoration->getMangledName() ==
                                             mangledNames[entryIndex].getUnownedSlice())
                {
                    SLANG_CHECK(entryInst == nullptr);
                    entryInst = inst;
                }
            }
            SLANG_CHECK_ABORT(entryInst != nullptr);

            bool requiresSpirv15 = false;
            bool requiresSpirv16 = false;
            for (auto decoration : entryInst->getDecorations())
            {
                if (auto requirement = as<IRRequireCapabilityAtomDecoration>(decoration))
                {
                    requiresSpirv15 |= requirement->getAtom() == Slang::CapabilityName::_spirv_1_5;
                    requiresSpirv16 |= requirement->getAtom() == Slang::CapabilityName::_spirv_1_6;
                }
            }
            if (entryIndex == 0)
            {
                SLANG_CHECK(requiresSpirv15);
                SLANG_CHECK(!requiresSpirv16);
            }
            else
            {
                SLANG_CHECK(requiresSpirv16);
            }
        }
        if (checkLinkedVersions)
            _checkEmittedSPIRVVersion(program, 0x00010600);
    }
}

} // namespace

SLANG_UNIT_TEST(entryPointCapabilityLayoutPreservesAPISpecializations)
{
    _checkEntryPointCapabilityLayout(EntryPointSelectionMode::SpecializeAPI);
}

SLANG_UNIT_TEST(entryPointCapabilityLayoutPreservesClosedSelections)
{
    _checkEntryPointCapabilityLayout(EntryPointSelectionMode::SpecializedName);
}

SLANG_UNIT_TEST(specializedEntryPointSPIRVVersionsSurviveLinking)
{
    _checkEntryPointCapabilityLayout(EntryPointSelectionMode::SpecializeAPI, true);
}

SLANG_UNIT_TEST(genericEntryPointRejectsIncompatibleTypeArgument)
{
    const char* source = R"(
        [require(metal)]
        struct MetalOnly {}
        struct Plain {}

        void helper<T>() {}

        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main<T>() { helper<T>(); }
    )";

    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK_ABORT(
        SLANG_SUCCEEDED(slang_createGlobalSession(SLANG_API_VERSION, globalSession.writeRef())));
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_HLSL;
    targetDesc.profile = globalSession->findProfile("sm_6_0");
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;
    ComPtr<slang::ISession> session;
    SLANG_CHECK_ABORT(
        SLANG_SUCCEEDED(globalSession->createSession(sessionDesc, session.writeRef())));

    ComPtr<slang::IBlob> diagnostics;
    auto module = session->loadModuleFromSourceString(
        "GenericEntryPointCapabilities",
        "GenericEntryPointCapabilities.slang",
        source,
        diagnostics.writeRef());
    SLANG_CHECK_ABORT(module != nullptr);
    ComPtr<slang::IEntryPoint> baseEntry;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->findAndCheckEntryPoint(
        "main",
        SLANG_STAGE_COMPUTE,
        baseEntry.writeRef(),
        diagnostics.writeRef())));
    auto moduleLayout = module->getLayout();
    SLANG_CHECK_ABORT(moduleLayout != nullptr);

    const char* typeNames[] = {"Plain", "MetalOnly"};
    const char* selections[] = {"main<Plain>", "main<MetalOnly>"};
    for (Index selectionMode = 0; selectionMode < 2; ++selectionMode)
    {
        for (Index typeIndex = 0; typeIndex < 2; ++typeIndex)
        {
            diagnostics.setNull();
            SlangResult result;
            if (selectionMode == 0)
            {
                auto type = moduleLayout->findTypeByName(typeNames[typeIndex]);
                SLANG_CHECK_ABORT(type != nullptr);
                auto argument = slang::SpecializationArg::fromType(type);
                ComPtr<slang::IComponentType> specializedEntry;
                result = baseEntry->specialize(
                    &argument,
                    1,
                    specializedEntry.writeRef(),
                    diagnostics.writeRef());
            }
            else
            {
                ComPtr<slang::IEntryPoint> selectedEntry;
                result = module->findAndCheckEntryPoint(
                    selections[typeIndex],
                    SLANG_STAGE_COMPUTE,
                    selectedEntry.writeRef(),
                    diagnostics.writeRef());
            }

            if (typeIndex == 0)
            {
                SLANG_CHECK(SLANG_SUCCEEDED(result));
            }
            else
            {
                SLANG_CHECK(SLANG_FAILED(result));
                SLANG_CHECK_ABORT(diagnostics != nullptr);
                auto diagnosticText = UnownedStringSlice(
                    static_cast<const char*>(diagnostics->getBufferPointer()),
                    diagnostics->getBufferSize());
                SLANG_CHECK(
                    diagnosticText.indexOf(toSlice("unavailable features in entry point")) != -1);
            }
        }
    }
}

SLANG_UNIT_TEST(specializedEntryPointCapabilitiesSurviveSPIRVLinking)
{
    const char* source = R"(
        struct Plain {}
        [require(spvShader64BitIndexingEXT)]
        struct WideIndex {}

        void helper<T>() {}

        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main<T>() { helper<T>(); }
    )";

    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK_ABORT(
        SLANG_SUCCEEDED(slang_createGlobalSession(SLANG_API_VERSION, globalSession.writeRef())));
    auto capability = globalSession->findCapability("spvShader64BitIndexingEXT");
    SLANG_CHECK_ABORT(capability != SLANG_CAPABILITY_UNKNOWN);
    slang::CompilerOptionEntry targetOptions[4] = {};
    targetOptions[0].name = slang::CompilerOptionName::Capability;
    targetOptions[0].value.kind = slang::CompilerOptionValueKind::Int;
    targetOptions[0].value.intValue0 = int32_t(capability);
    targetOptions[1].name = slang::CompilerOptionName::VulkanUseEntryPointName;
    targetOptions[1].value.kind = slang::CompilerOptionValueKind::Int;
    targetOptions[1].value.intValue0 = 1;
    // Emit binary directly so this compiler test needs no downstream optimizer or disassembler.
    targetOptions[2].name = slang::CompilerOptionName::Optimization;
    targetOptions[2].value.kind = slang::CompilerOptionValueKind::Int;
    targetOptions[2].value.intValue0 = SLANG_OPTIMIZATION_LEVEL_NONE;
    targetOptions[3].name = slang::CompilerOptionName::SkipSPIRVValidation;
    targetOptions[3].value.kind = slang::CompilerOptionValueKind::Int;
    targetOptions[3].value.intValue0 = 1;
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = globalSession->findProfile("spirv_1_6");
    targetDesc.compilerOptionEntries = targetOptions;
    targetDesc.compilerOptionEntryCount = SLANG_COUNT_OF(targetOptions);
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;
    ComPtr<slang::ISession> session;
    SLANG_CHECK_ABORT(
        SLANG_SUCCEEDED(globalSession->createSession(sessionDesc, session.writeRef())));

    ComPtr<slang::IBlob> diagnostics;
    auto module = session->loadModuleFromSourceString(
        "SpecializedEntryPointCapabilities",
        "SpecializedEntryPointCapabilities.slang",
        source,
        diagnostics.writeRef());
    SLANG_CHECK_ABORT(module != nullptr);
    ComPtr<slang::IEntryPoint> baseEntry;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->findAndCheckEntryPoint(
        "main",
        SLANG_STAGE_COMPUTE,
        baseEntry.writeRef(),
        diagnostics.writeRef())));
    auto moduleLayout = module->getLayout();
    SLANG_CHECK_ABORT(moduleLayout != nullptr);

    const char* typeNames[] = {"Plain", "WideIndex"};
    const char* entryNames[] = {"plainMain", "wideMain"};
    ComPtr<slang::IComponentType> renamedEntries[2];
    for (Index i = 0; i < SLANG_COUNT_OF(renamedEntries); ++i)
    {
        auto type = moduleLayout->findTypeByName(typeNames[i]);
        SLANG_CHECK_ABORT(type != nullptr);
        auto argument = slang::SpecializationArg::fromType(type);
        ComPtr<slang::IComponentType> specializedEntry;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
            baseEntry
                ->specialize(&argument, 1, specializedEntry.writeRef(), diagnostics.writeRef())));
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
            specializedEntry->renameEntryPoint(entryNames[i], renamedEntries[i].writeRef())));
    }

    // The capability and extension belong to the whole SPIR-V module, but the execution mode
    // belongs only to main<WideIndex>. Linking must neither drop it nor put it on main<Plain>.
    for (Index order = 0; order < 2; ++order)
    {
        slang::IComponentType* components[] = {
            module,
            renamedEntries[order].get(),
            renamedEntries[1 - order].get()};
        ComPtr<slang::IComponentType> program;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(session->createCompositeComponentType(
            components,
            SLANG_COUNT_OF(components),
            program.writeRef(),
            diagnostics.writeRef())));
        ComPtr<slang::IComponentType> linkedProgram;
        auto linkResult = program->link(linkedProgram.writeRef(), diagnostics.writeRef());
        if (SLANG_FAILED(linkResult))
            _reportDiagnostics(diagnostics);
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(linkResult));
        ComPtr<slang::IBlob> code;
        auto codeResult = linkedProgram->getTargetCode(0, code.writeRef(), diagnostics.writeRef());
        if (SLANG_FAILED(codeResult))
            _reportDiagnostics(diagnostics);
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(codeResult));
        SLANG_CHECK_ABORT(code != nullptr);

        auto words = static_cast<const uint32_t*>(code->getBufferPointer());
        auto wordCount = code->getBufferSize() / sizeof(uint32_t);
        SLANG_CHECK_ABORT(code->getBufferSize() % sizeof(uint32_t) == 0);
        SLANG_CHECK_ABORT(wordCount >= 5);
        SLANG_CHECK_ABORT(words[0] == SpvMagicNumber);
        uint32_t entryIDs[2] = {};
        uint32_t wideModeTarget = 0;
        Index wideModeCount = 0;
        bool hasCapability = false;
        bool hasExtension = false;
        for (size_t offset = 5; offset < wordCount;)
        {
            auto inst = words + offset;
            auto instWordCount = inst[0] >> SpvWordCountShift;
            auto op = SpvOp(inst[0] & SpvOpCodeMask);
            SLANG_CHECK_ABORT(instWordCount > 0 && offset + instWordCount <= wordCount);
            switch (op)
            {
            case SpvOpEntryPoint:
                {
                    SLANG_CHECK_ABORT(instWordCount >= 4);
                    auto name = UnownedStringSlice(reinterpret_cast<const char*>(inst + 3));
                    for (Index i = 0; i < SLANG_COUNT_OF(entryNames); ++i)
                    {
                        if (name == UnownedStringSlice(entryNames[i]))
                        {
                            SLANG_CHECK(inst[1] == SpvExecutionModelGLCompute);
                            SLANG_CHECK(entryIDs[i] == 0);
                            entryIDs[i] = inst[2];
                        }
                    }
                }
                break;
            case SpvOpExecutionMode:
                SLANG_CHECK_ABORT(instWordCount >= 3);
                if (inst[2] == SpvExecutionModeShader64BitIndexingEXT)
                {
                    ++wideModeCount;
                    wideModeTarget = inst[1];
                }
                break;
            case SpvOpCapability:
                SLANG_CHECK_ABORT(instWordCount >= 2);
                hasCapability |= inst[1] == SpvCapabilityShader64BitIndexingEXT;
                break;
            case SpvOpExtension:
                SLANG_CHECK_ABORT(instWordCount >= 2);
                hasExtension |= UnownedStringSlice(reinterpret_cast<const char*>(inst + 1)) ==
                                toSlice("SPV_EXT_shader_64bit_indexing");
                break;
            default:
                break;
            }
            offset += instWordCount;
        }
        SLANG_CHECK(entryIDs[0] != 0 && entryIDs[1] != 0);
        SLANG_CHECK(entryIDs[0] != entryIDs[1]);
        SLANG_CHECK(hasCapability);
        SLANG_CHECK(hasExtension);
        SLANG_CHECK(wideModeCount == 1);
        SLANG_CHECK(wideModeTarget == entryIDs[1]);
    }
}
