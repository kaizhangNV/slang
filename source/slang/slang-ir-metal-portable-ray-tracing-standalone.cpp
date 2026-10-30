#include "slang-ir-metal-portable-ray-tracing-standalone.h"

#include "slang-ir-metal-portable-ray-tracing-candidates.h"
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
    List<VisibleAdapter> result;
    List<IRFunc*> nativeEntries;
    IRBuilder builder(module);
    for (auto wrapper : entryPoints)
    {
        auto adapter = wrapper->findDecoration<IRPortableRTStageAdapterDecoration>();
        if (!adapter)
        {
            nativeEntries.add(wrapper);
            continue;
        }
        auto entry = wrapper->findDecoration<IREntryPointDecoration>();
        SLANG_RELEASE_ASSERT(entry);
        auto input = getResolvedInstForDecorations(adapter->getInputType())
                         ->findDecoration<IRPortableRTStageInputDecoration>();
        SLANG_RELEASE_ASSERT(input);

        // Selecting Miss supplies a checked invoke and Context without selecting a program
        // schema. The same source contract fixes the VFT signature in the caller's compilation;
        // only this adapter needs the implementation and its logical input metadata.
        PortableRT::SchemaStage stage;
        stage.stage = entry->getProfile().getStage();
        stage.isPresent = true;
        stage.invoke = cast<IRFunc>(adapter->getInvoke());
        stage.input = input;
        stage.contextType = input->getContextType();
        stage.contextWitness = cast<IRWitnessTable>(input->getContextWitness());
        stage.payloadType = input->getPayloadType();
        stage.recordType = input->getRecordType();
        if (stage.stage == Stage::ClosestHit || stage.stage == Stage::AnyHit ||
            stage.stage == Stage::Intersection)
        {
            stage.primitiveType = input->getPrimitiveType();
            stage.attributesType = input->getAttributesType();
        }

        auto symbolName = entry->getName()->getStringSlice();
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
            auto acceleration = PortableRT::getStageAccelerationStructure(input);
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

        // The selected export name now belongs to its physical VFT adapter. Drop the temporary
        // native wrapper's roots so ordinary DCE removes it, while the source invoke remains
        // callable by both this adapter and ordinary source helpers.
        entry->removeAndDeallocate();
        adapter->removeAndDeallocate();
        if (auto keepAlive = wrapper->findDecoration<IRKeepAliveDecoration>())
            keepAlive->removeAndDeallocate();
        if (auto layout = wrapper->findDecoration<IRLayoutDecoration>())
            layout->removeAndDeallocate();
    }
    entryPoints = _Move(nativeEntries);
    return result;
}

} // namespace Slang
