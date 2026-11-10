#include "slang-ir-metal-portable-ray-tracing-context.h"

#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-rich-diagnostics.h"

namespace Slang
{

void preserveMetalRTGlobalResources(IRModule* module, const List<IRGlobalParam*>& globalParams)
{
    IRBuilder builder(module);
    IRIntegerValue layoutIndex = 0;
    // Consider a miss shader that reads a global buffer while ray generation only calls trace.
    // Both compilations need the same resource fields in MetalRTGlobalResources. Keep the linked
    // component's runtime globals after uniform collection has replaced ordinary uniforms with
    // their buffer. Its result list supplies layout order even though the layout's field keys no
    // longer refer to those globals. Record that order before optimization can remove unused ones.
    for (auto parameter : globalParams)
    {
        auto layout = findVarLayout(parameter);
        SLANG_RELEASE_ASSERT(layout);
        if (!layout->findOffsetAttr(LayoutResourceKind::SpecializationConstant))
        {
            builder.addDecoration(
                parameter,
                kIROp_MetalRTGlobalResourceDecoration,
                builder.getIntValue(builder.getIntType(), layoutIndex));
            if (!parameter->findDecoration<IRKeepAliveDecoration>())
                builder.addKeepAliveDecoration(parameter);
        }
        ++layoutIndex;
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
