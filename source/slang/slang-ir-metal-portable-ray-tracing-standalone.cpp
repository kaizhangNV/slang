#include "slang-ir-metal-portable-ray-tracing-standalone.h"

#include "slang-ir-metal-portable-ray-tracing-candidates.h"
#include "slang-ir-portable-ray-tracing-util.h"
#include "slang-ir-util.h"

namespace Slang
{
using namespace PortableRT::Metal;

List<VisibleAdapter> prepareMetalStandaloneStages(
    IRModule* module,
    TargetProgram*,
    List<IRFunc*>& entryPoints,
    bool& requiresResultTypeLowering,
    DiagnosticSink* sink)
{
    struct SelectedStage
    {
        PortableRT::SchemaStage stage;
        String name;
    };

    List<VisibleAdapter> result;
    List<IRFunc*> nativeEntries;
    List<SelectedStage> selectedStages;
    IRBuilder builder(module);
    for (auto invoke : entryPoints)
    {
        auto input = PortableRT::getEntryPointInputInfo(invoke);
        if (!input)
        {
            nativeEntries.add(invoke);
            continue;
        }
        auto entry = invoke->findDecoration<IREntryPointDecoration>();
        SLANG_RELEASE_ASSERT(entry);

        // Selecting Miss supplies a checked invoke and Context without selecting a program
        // schema. The same source contract fixes the VFT signature in the caller's compilation;
        // only this adapter needs the implementation and its logical input decoration.
        SelectedStage selected;
        selected.stage.stage = entry->getProfile().getStage();
        selected.stage.isPresent = true;
        selected.stage.invoke = invoke;
        selected.stage.input = input;
        selected.name = entry->getName()->getStringSlice();
        selectedStages.add(_Move(selected));

        // The final adapter owns the selected export. Keep the original function callable,
        // without an entry-point signature or its reflected shader-record parameter layout.
        entry->removeAndDeallocate();
        if (auto keepAlive = invoke->findDecoration<IRKeepAliveDecoration>())
            keepAlive->removeAndDeallocate();
        if (auto layout = invoke->findDecoration<IRLayoutDecoration>())
            layout->removeAndDeallocate();
    }

    // Candidate adapters clone parts of the source call graph. Remove remaining entry markers
    // before cloning, so linked implementations used as helpers also remain ordinary functions.
    PortableRT::demoteStageImplementations(module);

    for (const auto& selected : selectedStages)
    {
        const auto& stage = selected.stage;
        auto input = stage.input;
        auto symbolName = selected.name.getUnownedSlice();
        VisibleAdapter root;
        if (stage.stage == Stage::AnyHit || stage.stage == Stage::Intersection)
        {
            root = createSelectedCandidateAdapter(
                module,
                stage,
                symbolName,
                requiresResultTypeLowering,
                sink);
        }
        else
        {
            auto acceleration =
                cast<IRRaytracingAccelerationStructureType>(input->getAccelerationStructureType());
            bool multiLevel =
                acceleration->getOperandCount() && getIntVal(acceleration->getOperand(0)) != 0;
            auto requirements = getStageContractRequirements(stage.stage, multiLevel);
            auto descriptorPointer =
                builder.getPtrType(builder.getUIntType(), AddressSpace::Uniform);
            if (stage.stage == Stage::Callable)
            {
                root = createCallableAdapter(
                    module,
                    stage,
                    requirements,
                    symbolName,
                    descriptorPointer);
            }
            else
            {
                SLANG_RELEASE_ASSERT(
                    stage.stage == Stage::ClosestHit || stage.stage == Stage::Miss);
                root = createVisibleAdapter(
                    module,
                    stage,
                    requirements,
                    symbolName,
                    descriptorPointer);
            }
        }
        result.add(_Move(root));
    }
    entryPoints = _Move(nativeEntries);
    return result;
}

} // namespace Slang
