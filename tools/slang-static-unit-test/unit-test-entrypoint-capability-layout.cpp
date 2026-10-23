#include "compiler-core/slang-diagnostic-sink.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "slang/slang-ast-builder.h"
#include "slang/slang-entry-point.h"
#include "slang/slang-ir-insts.h"
#include "slang/slang-mangle.h"
#include "slang/slang-session.h"
#include "slang/slang-target-program.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

SLANG_UNIT_TEST(entryPointCapabilityLayoutPreservesSpecializations)
{
    const char* source = R"(
        [require(_spirv_1_5)]
        struct First {}
        [require(_spirv_1_6)]
        struct Second {}

        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main<T>() {}
    )";

    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK_ABORT(
        SLANG_SUCCEEDED(slang_createGlobalSession(SLANG_API_VERSION, globalSession.writeRef())));

    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = globalSession->findProfile("spirv_1_6");
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

    const char* selections[] = {"main<First>", "main<Second>"};
    ComPtr<slang::IEntryPoint> selectedEntries[2];
    EntryPoint* entries[2];
    for (Index i = 0; i < SLANG_COUNT_OF(selectedEntries); ++i)
    {
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->findAndCheckEntryPoint(
            selections[i],
            SLANG_STAGE_COMPUTE,
            selectedEntries[i].writeRef(),
            diagnostics.writeRef())));
        entries[i] = static_cast<EntryPoint*>(selectedEntries[i].get());
    }
    SLANG_CHECK_ABORT(entries[0] != entries[1]);
    SLANG_CHECK_ABORT(entries[0]->getFuncDecl() == entries[1]->getFuncDecl());

    auto linkage = entries[0]->getLinkage();
    auto astBuilder = linkage->getASTBuilder();
    SLANG_AST_BUILDER_RAII(astBuilder);

    // Generic entry-point validation currently misses requirements from the selected type
    // arguments (shader-slang/slang#13893). Supply different validated requirements here to test
    // the layout consumer independently of that bug. Both entries share main's FuncDecl, so a
    // dictionary keyed by that declaration would replace one entry's requirements with the other's.
    entries[0]->setInferredCapabilityRequirements(
        astBuilder->getCapabilitySetVal(Slang::CapabilityName::_spirv_1_5));
    entries[1]->setInferredCapabilityRequirements(
        astBuilder->getCapabilitySetVal(Slang::CapabilityName::_spirv_1_6));

    String mangledNames[] = {
        getMangledName(astBuilder, entries[0]->getFuncDeclRef()),
        getMangledName(astBuilder, entries[1]->getFuncDeclRef())};
    SLANG_CHECK_ABORT(mangledNames[0] != mangledNames[1]);

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
    }
}
