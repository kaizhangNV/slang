#include "slang-ir-portable-ray-tracing-util.h"

#include "slang-ir-clone.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"

namespace Slang
{

namespace PortableRT
{
IRPortableRTStageInputDecoration* getStageInputInfo(IRType* type)
{
    if (auto pointerType = as<IRPtrTypeBase>(type))
        type = pointerType->getValueType();
    return getResolvedInstForDecorations(type)->findDecoration<IRPortableRTStageInputDecoration>();
}

IRPortableRTStageInputDecoration* getEntryPointInputInfo(IRFunc* function)
{
    if (!function->findDecoration<IREntryPointDecoration>() || function->getParamCount() != 1)
        return nullptr;
    return getStageInputInfo(function->getParamType(0));
}

void demoteStageImplementation(IRFunc* function)
{
    SLANG_RELEASE_ASSERT(getEntryPointInputInfo(function));
    // For example, a selected Metal Miss becomes an ordinary invoke helper behind a visible
    // adapter. Its old denormal modes must not remain as entry execution modes on that helper.
    for (auto decoration = function->getFirstDecoration(); decoration;)
    {
        auto next = decoration->getNextDecoration();
        switch (decoration->getOp())
        {
        case kIROp_EntryPointDecoration:
        case kIROp_FpDenormalPreserveDecoration:
        case kIROp_FpDenormalFlushToZeroDecoration:
            decoration->removeAndDeallocate();
            break;
        default:
            break;
        }
        decoration = next;
    }
}

static void _demoteStageImplementations(IRInst* parent)
{
    for (auto child : parent->getChildren())
    {
        if (auto function = as<IRFunc>(child))
        {
            if (getEntryPointInputInfo(function))
                demoteStageImplementation(function);
        }
        else if (as<IRGeneric>(child) || as<IRBlock>(child))
        {
            _demoteStageImplementations(child);
        }
    }
}

void demoteStageImplementations(IRModule* module)
{
    _demoteStageImplementations(module->getModuleInst());
}
} // namespace PortableRT

PortableRTInvocationParameter::PortableRTInvocationParameter(
    IRModule* module,
    IRType* valueType,
    const char* name)
    : m_module(module), m_valueType(valueType), m_name(name)
{
}

void PortableRTInvocationParameter::bind(IRFunc* function, IRInst* value)
{
    SLANG_RELEASE_ASSERT(!m_values.containsKey(function));
    SLANG_RELEASE_ASSERT(value->getDataType() == m_valueType);
    m_values.add(function, value);
}

IRInst* PortableRTInvocationParameter::createParameter(IRFunc* function)
{
    auto block = function->getFirstBlock();
    SLANG_RELEASE_ASSERT(block);
    IRBuilder builder(m_module);
    auto parameter = builder.createParam(m_valueType);
    parameter->insertBefore(block->getFirstOrdinaryInst());
    builder.addNameHintDecoration(parameter, UnownedTerminatedStringSlice(m_name));
    return parameter;
}

IRInst* PortableRTInvocationParameter::get(IRFunc* function)
{
    if (auto value = m_values.tryGetValue(function))
        return *value;

    auto value = createParameter(function);
    // Register before visiting callers: a recursive helper graph still has one parameter for
    // this value per function, rather than recursively allocating new parameters forever.
    bind(function, value);
    fixUpFuncType(function);

    List<IRCall*> calls;
    for (auto use = function->firstUse; use; use = use->nextUse)
    {
        auto call = as<IRCall>(use->getUser());
        if (call && call->getCalleeUse() == use)
            calls.add(call);
    }
    IRBuilder builder(m_module);
    for (auto call : calls)
    {
        List<IRInst*> arguments;
        for (UInt i = 0; i < call->getArgCount(); ++i)
            arguments.add(call->getArg(i));
        arguments.add(get(getParentFunc(call)));
        builder.setInsertBefore(call);
        auto replacement = builder.emitCallInst(call->getDataType(), function, arguments);
        replacement->sourceLoc = call->sourceLoc;
        for (auto decoration : call->getDecorations())
            cloneDecoration(decoration, replacement);
        call->replaceUsesWith(replacement);
        call->removeAndDeallocate();
    }
    return value;
}

void PortableRTInvocationParameter::lower(IRInst* operation)
{
    operation->replaceUsesWith(get(getParentFunc(operation)));
    operation->removeAndDeallocate();
}

} // namespace Slang
