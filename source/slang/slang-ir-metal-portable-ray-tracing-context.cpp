#include "slang-ir-metal-portable-ray-tracing-context.h"

#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-rich-diagnostics.h"

namespace Slang
{
namespace
{

// The explicit-context producer marks this parameter because its debug name is not semantic.
// Ordinary stage parameters may have the same spelling without representing global state.
IRParam* findContextParameter(IRFunc* function)
{
    for (auto parameter : function->getParams())
        if (parameter->findDecoration<IRExplicitGlobalContextDecoration>())
            return parameter;
    return nullptr;
}

// Hides the caller-specific context type from a visible table's public function signature.
// The adapter immediately restores the concrete pointer before accessing its fields. This is
// the same ordinary invocation storage, not a second copy of global shader variables.
void eraseContextParameterType(IRModule* module, IRFunc* function, IRParam* parameter)
{
    IRBuilder builder(module);
    builder.setInsertBefore(function->getFirstBlock()->getFirstOrdinaryInst());
    auto typedContext = builder.emitBitCast(parameter->getDataType(), parameter);
    builder.addDecoration(typedContext, kIROp_ExplicitGlobalContextDecoration);
    parameter->replaceUsesWith(typedContext);
    typedContext->setOperand(0, parameter);
    parameter->setFullType(builder.getPtrType(builder.getUInt8Type(), AddressSpace::ThreadLocal));
    fixUpFuncType(function);
}

// All arms of a visible table need one signature, including arms that do not read any globals.
void appendUnusedContextParameter(IRModule* module, IRFunc* function)
{
    IRBuilder builder(module);
    auto parameter =
        builder.createParam(builder.getPtrType(builder.getUInt8Type(), AddressSpace::ThreadLocal));
    builder.addNameHintDecoration(parameter, toSlice("kernelContext"));
    builder.addDecoration(parameter, kIROp_ExplicitGlobalContextDecoration);
    parameter->insertBefore(function->getFirstBlock()->getFirstOrdinaryInst());
    fixUpFuncType(function);
}

// Consider an AnyHit method reading RWStructuredBuffer<uint> values. Global-context lowering
// gives its native callback a parameter with the original MetalBuffer layout, emitted as
// device uint* values [[buffer(n)]]. Textures and samplers cannot be direct IFT parameters;
// they must instead reside inside a buffer-backed ParameterBlock supplied by the application.
bool isBindableIntersectionParameter(IRParam* parameter)
{
    auto layout = findVarLayout(parameter);
    if (!layout)
        return true;

    bool hasBufferBinding = false;
    for (auto offset : layout->getOffsetAttrs())
    {
        switch (offset->getResourceKind())
        {
        case LayoutResourceKind::MetalTexture:
        case LayoutResourceKind::SamplerState:
            return false;
        case LayoutResourceKind::MetalBuffer:
            hasBufferBinding = true;
            break;
        default:
            break;
        }
    }
    if (!hasBufferBinding)
        return true;

    auto type = parameter->getDataType();
    if (auto handle = as<IRDescriptorHandleType>(type))
        type = handle->getResourceType();
    return as<IRPtrTypeBase>(type) || as<IRHLSLStructuredBufferTypeBase>(type) ||
           as<IRByteAddressBufferTypeBase>(type) || as<IRUniformParameterGroupType>(type);
}

} // namespace

void finalizeMetalPortableRayTracingGlobalContext(IRModule* module, DiagnosticSink* sink)
{
    List<IRFunc*> visibleFunctions;
    List<IRMetalVisibleFunctionTable*> tables;
    for (auto inst : module->getGlobalInsts())
    {
        if (auto table = as<IRMetalVisibleFunctionTable>(inst))
            tables.add(table);
        auto function = as<IRFunc>(inst);
        if (!function)
            continue;
        auto context = findContextParameter(function);
        if (function->findDecoration<IRMetalIntersectionFunctionDecoration>())
        {
            // The driver supplies each callback's buffer parameters. A remaining ordinary
            // context parameter means mutable global state was reached: there is no caller
            // from which to inherit that state, and silently allocating fresh state is wrong.
            if (context)
            {
                sink->diagnose(Diagnostics::MetalRayTracingCandidateGlobalVariable{
                    .location =
                        context->sourceLoc.isValid() ? context->sourceLoc : function->sourceLoc});
                continue;
            }
            for (auto parameter : function->getParams())
                if (!isBindableIntersectionParameter(parameter))
                    sink->diagnose(Diagnostics::MetalRayTracingCandidateResourceKind{
                        .location = parameter->sourceLoc});
        }
        else if (function->findDecoration<IRMetalVisibleFunctionDecoration>())
        {
            visibleFunctions.add(function);
            if (context)
                eraseContextParameterType(module, function, context);
        }
    }

    IRBuilder builder(module);
    HashSet<IRMetalVisibleFunctionTable*> tablesWithContext;
    for (auto table : tables)
    {
        IRFuncType* contextSignature = nullptr;
        for (auto function : visibleFunctions)
        {
            auto decoration = function->findDecoration<IRMetalVisibleFunctionDecoration>();
            if (decoration->getTableType() == table && findContextParameter(function))
            {
                contextSignature = cast<IRFuncType>(function->getDataType());
                SLANG_RELEASE_ASSERT(
                    contextSignature->getParamCount() ==
                    table->getFunctionType()->getParamCount() + 1);
                break;
            }
        }
        if (!contextSignature)
            continue;

        for (auto function : visibleFunctions)
        {
            auto decoration = function->findDecoration<IRMetalVisibleFunctionDecoration>();
            if (decoration->getTableType() != table)
                continue;
            if (!findContextParameter(function))
                appendUnusedContextParameter(module, function);
            SLANG_RELEASE_ASSERT(function->getDataType() == contextSignature);
        }
        IRInst* operands[] = {contextSignature, table->getStageKind()};
        auto physicalTable = cast<IRMetalVisibleFunctionTable>(
            builder.getType(kIROp_MetalVisibleFunctionTable, 2, operands));
        table->replaceUsesWith(physicalTable);
        tablesWithContext.add(physicalTable);
    }

    // Explicit-global-context lowering has already propagated the caller's context through
    // direct helpers and these indirect dispatch edges. Only the table-wide ABI flags remain.
    // This includes a trace in a helper that never directly reads a global resource itself.
    for (auto inst : module->getGlobalInsts())
    {
        auto function = as<IRFunc>(inst);
        if (!function)
            continue;
        for (auto block : function->getBlocks())
        {
            for (auto operation : block->getChildren())
            {
                if (auto trace = as<IRMetalRTTraceDispatch>(operation))
                {
                    auto miss =
                        as<IRMetalVisibleFunctionTable>(trace->getMissFunctions()->getDataType());
                    auto hit = as<IRMetalVisibleFunctionTable>(
                        trace->getClosestHitFunctions()->getDataType());
                    bool missNeedsContext = miss && tablesWithContext.contains(miss);
                    bool hitNeedsContext = hit && tablesWithContext.contains(hit);
                    if (missNeedsContext || hitNeedsContext)
                    {
                        SLANG_RELEASE_ASSERT(
                            trace->getGlobalContext()
                                ->findDecoration<IRExplicitGlobalContextDecoration>());
                        // These operand positions are the fixed physical dispatch ABI, not
                        // positions in source witnesses or interface requirements.
                        trace->setOperand(27, builder.getBoolValue(missNeedsContext));
                        trace->setOperand(28, builder.getBoolValue(hitNeedsContext));
                    }
                }
                else if (auto call = as<IRMetalRTCallDispatch>(operation))
                {
                    auto resources =
                        cast<IRUniformParameterGroupType>(call->getDescriptorResourcesType());
                    auto field = findStructField(
                        cast<IRStructType>(resources->getElementType()),
                        call->getCallableFunctionsKey());
                    auto table = cast<IRMetalVisibleFunctionTable>(field->getFieldType());
                    if (tablesWithContext.contains(table))
                    {
                        SLANG_RELEASE_ASSERT(
                            call->getGlobalContext()
                                ->findDecoration<IRExplicitGlobalContextDecoration>());
                        call->setOperand(12, builder.getBoolValue(true));
                    }
                }
            }
        }
    }
}

} // namespace Slang
