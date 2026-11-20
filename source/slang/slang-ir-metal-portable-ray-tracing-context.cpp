#include "slang-ir-metal-portable-ray-tracing-context.h"

#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-rich-diagnostics.h"

namespace Slang
{

void preserveMetalRTGlobalResources(IRModule* module, IRVarLayout* globalScopeVarLayout)
{
    if (!globalScopeVarLayout)
        return;

    IRBuilder builder(module);
    // Consider a miss shader that reads a global buffer while ray generation only calls trace.
    // Both compilations need the same resource fields in MetalRTGlobalResources. Source globals
    // retain their linkage and layout after uniform collection. The collected uniform buffer
    // has no linkage, but carries the component's scope layout. Compiler-generated buffers such
    // as coverage instrumentation belong to the individual compilation and are excluded here.
    List<IRGlobalParam*> globalParams;
    IRGlobalParam* uniformBuffer = nullptr;
    for (auto inst : module->getGlobalInsts())
    {
        auto parameter = as<IRGlobalParam>(inst);
        if (!parameter)
            continue;
        auto layout = findVarLayout(parameter);
        if (!layout || layout->findOffsetAttr(LayoutResourceKind::SpecializationConstant))
            continue;
        if (!parameter->findDecoration<IRLinkageDecoration>())
        {
            if (layout != globalScopeVarLayout)
                continue;
            auto type = parameter->getDataType();
            if (auto group = as<IRUniformParameterGroupType>(type))
                type = group->getElementType();
            SLANG_RELEASE_ASSERT(type->findDecoration<IRSynthesizedParameterGroupDecoration>());
            SLANG_RELEASE_ASSERT(!uniformBuffer);
            uniformBuffer = parameter;
        }
        else
            SLANG_RELEASE_ASSERT(getMangledName(parameter).getLength());
        globalParams.add(parameter);
    }

    // Linking different entry points may discover the globals in different orders. Put the
    // unnamed uniform buffer first, then sort by source linkage name. Unlike a binding number,
    // a linkage name distinguishes declarations even when their explicit bindings overlap.
    globalParams.sort([](IRGlobalParam* left, IRGlobalParam* right)
                      { return compare(getMangledName(left), getMangledName(right)) < 0; });
    for (Index i = 0; i < globalParams.getCount(); ++i)
    {
        auto parameter = globalParams[i];
        if (i)
            SLANG_RELEASE_ASSERT(getMangledName(globalParams[i - 1]) != getMangledName(parameter));
        builder.addDecoration(
            parameter,
            kIROp_MetalRTGlobalResourceDecoration,
            builder.getIntValue(builder.getIntType(), i));
        if (!parameter->findDecoration<IRKeepAliveDecoration>())
            builder.addKeepAliveDecoration(parameter);
    }
}

// Consider an AnyHit method reading RWStructuredBuffer<uint> values. Global-context lowering
// gives its native callback a parameter with the original MetalBuffer layout, emitted as
// device uint* values [[buffer(n)]]. Textures and samplers cannot be direct IFT parameters;
// they must instead reside inside a buffer-backed ParameterBlock supplied by the application.
bool isMetalPortableRayTracingBindableResource(IRInst* parameter)
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

void finalizeMetalPortableRayTracingGlobalContext(IRModule* module, DiagnosticSink* sink)
{
    // Visible adapters already have the schema's fixed resource-context parameter. Explicit
    // global-context lowering gives that parameter its meaning without changing table types.
    // Traversal callbacks instead retain the ordinary driver-bound buffer parameters.
    for (auto inst : module->getGlobalInsts())
    {
        auto function = as<IRFunc>(inst);
        if (!function || !function->findDecoration<IRMetalIntersectionFunctionDecoration>())
            continue;
        for (auto parameter : function->getParams())
            if (!isMetalPortableRayTracingBindableResource(parameter))
                sink->diagnose(Diagnostics::MetalRayTracingCandidateResourceKind{
                    .location = parameter->sourceLoc});
    }
}

} // namespace Slang
