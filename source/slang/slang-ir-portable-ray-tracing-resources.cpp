#include "slang-ir-portable-ray-tracing-resources.h"

#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{
namespace
{
void collectDescriptorTypes(IRInst* parent, List<IRTraceProgramDescriptorType*>& result)
{
    if (auto descriptor = as<IRTraceProgramDescriptorType>(parent))
        result.add(descriptor);
    for (auto child : parent->getChildren())
        collectDescriptorTypes(child, result);
}
} // namespace

void eraseNativeRayTracingDescriptors(IRModule* module)
{
    // Consider a noinline helper taking Holder<Schema>, where Holder stores a descriptor and
    // ordinary application data. Native trace calls do not read the descriptor, but the helper
    // signature or a bound Holder can still retain its opaque type. Replace that type itself,
    // rather than deleting only the globals used by trace calls, so nested fields and helper
    // signatures agree with their zero-resource reflection layout.
    //
    // Collect first: replacing a hoistable type can deduplicate other hoistable instructions,
    // which would invalidate a walk that changes the instruction list as it visits it.
    List<IRTraceProgramDescriptorType*> descriptors;
    collectDescriptorTypes(module->getModuleInst(), descriptors);
    IRBuilder builder(module);
    for (auto descriptor : descriptors)
    {
        descriptor->replaceUsesWith(builder.getVoidType());
        descriptor->removeAndDeallocate();
    }
}
} // namespace Slang
