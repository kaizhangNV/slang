#include "slang-ir-metal-portable-ray-tracing-standalone.h"

#include "slang-ir-legalize-matrix-types.h"
#include "slang-ir-util.h"
#include "slang-rich-diagnostics.h"

namespace Slang
{
using namespace PortableRT::Metal;

namespace
{

/// Gives an ordinary standalone helper an input for a used query. Reference getters retain
/// thread-local pointer identity, while immutable values can be passed normally. Indexed instance
/// queries use the trace context's topology, just like schema adapters. A shared source method
/// must receive the same representation whether selected independently or through a schema.
IRType* getStandaloneParameterType(
    IRBuilder& builder,
    IRInst* operation,
    IRPortableRTStageInputDecoration* input)
{
    if (operation->getOp() == kIROp_MetalRTGetInstanceIndex ||
        operation->getOp() == kIROp_MetalRTGetInstanceID)
    {
        auto acceleration = PortableRT::getStageAccelerationStructure(input);
        bool isMultiLevel =
            acceleration->getOperandCount() && getIntVal(acceleration->getOperand(0)) != 0;
        return isMultiLevel ? builder.getType(kIROp_MetalRTInstancePath) : builder.getUIntType();
    }
    if (auto pointer = as<IRPtrTypeBase>(operation->getDataType()))
        return builder.getPtrType(pointer->getValueType(), AddressSpace::ThreadLocal);
    return operation->getDataType();
}

} // namespace

List<VisibleAdapter> prepareMetalStandaloneStages(
    IRModule* module,
    TargetProgram* targetProgram,
    List<IRFunc*>& entryPoints,
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
        auto stage = entry->getProfile().getStage();
        if (stage != Stage::ClosestHit && stage != Stage::Miss && stage != Stage::Callable)
        {
            // A standalone candidate has no owning hit group, so neither its AnyHit callback nor
            // its mutable candidate-state ABI is known. Do not silently replace ReportHit or an
            // early-termination operation with a helper that cannot honor those semantics.
            sink->diagnose(
                Diagnostics::MetalRayTracingStandaloneCandidate{.location = wrapper->sourceLoc});
            continue;
        }

        VisibleAdapter root;
        root.function = wrapper;
        auto input = adapter->getInputType()->findDecoration<IRPortableRTStageInputDecoration>();
        SLANG_RELEASE_ASSERT(input);
        List<IRInst*> operations;
        collectStageInputOperations(wrapper, operations);
        HashSet<KeyValuePair<IROp, IRType*>> seen;
        auto firstInstruction = wrapper->getFirstBlock()->getFirstOrdinaryInst();
        for (auto operation : operations)
        {
            auto type = operation->getDataType();
            if (!seen.add({operation->getOp(), type}))
                continue;
            builder.setInsertBefore(firstInstruction);
            IRInst* value;
            if (operation->getOp() == kIROp_PortableRTGetHitAttributes &&
                isBuiltinHitAttributeView(operation))
            {
                // Triangle/curve attribute containers are zero-storage views. Their individual
                // getters, not the aggregate token, contribute actual invocation parameters.
                value = builder.emitDefaultConstruct(type);
            }
            else if (as<IRVoidType>(type) || isEmptyType(type))
            {
                value = as<IRVoidType>(type) ? builder.getVoidValue()
                                             : builder.emitDefaultConstruct(type);
            }
            else
            {
                auto parameterType = getStandaloneParameterType(builder, operation, input);
                // An exported helper has no caller from which address-space specialization can
                // infer an aggregate parameter's borrow. Choose its concrete read-only ABI here;
                // helpers inside the implementation still receive ordinary logical values.
                // Tuple and integer-matrix records become structs/arrays in later ordinary
                // legalization. Their exported parameter needs the same concrete borrow now;
                // waiting for constref transformation would leave its address space generic.
                auto matrix = as<IRMatrixType>(parameterType);
                bool borrow = isCompositeType(parameterType) || as<IRTupleType>(parameterType) ||
                              (matrix && isMatrixTypeLegalizedToVectorArray(targetProgram, matrix));
                auto parameter = builder.createParam(
                    borrow ? builder.getBorrowInParamType(parameterType, AddressSpace::ThreadLocal)
                           : parameterType);
                parameter->insertBefore(wrapper->getFirstBlock()->getFirstOrdinaryInst());
                builder.addNameHintDecoration(
                    parameter,
                    UnownedStringSlice(getIROpInfo(operation->getOp()).name));
                value = borrow ? builder.emitLoad(parameter) : parameter;
            }
            root.bindings.add({operation->getOp(), type, value});
        }

        // Consider a selected SharedMiss which CallingMiss also invokes directly. Only this
        // wrapper acquires the export name and helper ABI; SharedMiss.invoke remains an ordinary
        // implementation and participates in the same invocation-value threading as every caller.
        auto exportName = entry->getName()->getStringSlice();
        builder.addExportDecoration(wrapper, exportName);
        builder.addExternCppDecoration(wrapper, exportName);
        builder.addDecoration(wrapper, kIROp_MetalRTStandaloneStageDecoration);
        entry->removeAndDeallocate();
        adapter->removeAndDeallocate();
        if (auto layout = wrapper->findDecoration<IRLayoutDecoration>())
            layout->removeAndDeallocate();
        fixUpFuncType(wrapper);
        result.add(_Move(root));
    }
    entryPoints = _Move(nativeEntries);
    return result;
}

} // namespace Slang
