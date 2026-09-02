#include "slang-ir-structural-ray-tracing.h"

#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"
#include "slang-mangle.h"
#include "slang-module.h"
#include "slang-rich-diagnostics.h"

// Structural ray-tracing stage interfaces need identities that survive AST lifetime, module
// serialization, linking, and specialization. The ordinary source module initially lowers every
// interface to `IRInterfaceType`. Once the compiler has verified that it loaded the packaged
// `slang.raytracing` module, this file replaces those ordinary opcodes with stage-specific
// interface opcodes. Downstream IR code can then recognize a stage contract from the instruction
// class instead of depending on source names or AST declaration pointers.

namespace Slang
{

// Map the shared source/AST stage classification to its persistent IR identity.
IROp getStructuralRayTracingStageInterfaceOp(StructuralRayTracingStageKind kind)
{
    switch (kind)
    {
    case StructuralRayTracingStageKind::ClosestHit:
        return kIROp_ClosestHitStageInterface;
    case StructuralRayTracingStageKind::AnyHit:
        return kIROp_AnyHitStageInterface;
    case StructuralRayTracingStageKind::Intersection:
        return kIROp_IntersectionStageInterface;
    case StructuralRayTracingStageKind::Miss:
        return kIROp_MissStageInterface;
    case StructuralRayTracingStageKind::Callable:
        return kIROp_CallableStageInterface;
    default:
        return kIROp_Invalid;
    }
}

void diagnoseUnloweredTraceProgramDescriptorTypes(IRModule* module, DiagnosticSink* sink)
{
    // Descriptor types are hoisted module-scope instructions, so an exact-op scan is sufficient;
    // there is no need to rediscover them by walking operand or use graphs. Target lowering owns
    // their physical representation and must remove every such instruction before emission.
    for (auto inst : module->getGlobalInsts())
    {
        if (inst->getOp() != kIROp_TraceProgramDescriptorType)
            continue;

        sink->diagnose(Diagnostics::TraceProgramDescriptorNotSupportedOnTarget{
            .location = findFirstUseLoc(inst)});
    }
}

// Return the interface type represented by a mangled symbol. A generic interface's symbol names the
// enclosing `IRGeneric`, whose return value is the actual interface-type instruction.
static IRInterfaceType* _findInterfaceType(IRInst* inst)
{
    if (auto generic = as<IRGeneric>(inst))
        inst = findInnerMostGenericReturnVal(generic);
    return as<IRInterfaceType>(inst);
}

bool identifyStructuralRayTracingStageInterfaces(
    Module* module,
    const StructuralRayTracingDeclRegistry& registry,
    StructuralRayTracingStageKind* outMissingStage)
{
    auto irModule = module->getIRModule();
    auto astBuilder = module->getASTBuilder();
    SLANG_AST_BUILDER_RAII(astBuilder);

    for (int i = 0; i < int(StructuralRayTracingStageKind::Count); ++i)
    {
        auto kind = StructuralRayTracingStageKind(i);
        auto interfaceDecl = registry.getStageInterface(kind);
        auto mangledName = getMangledName(astBuilder, interfaceDecl);
        auto symbols = irModule->findSymbolByMangledName(ImmutableHashedString(mangledName));
        auto expectedOp = getStructuralRayTracingStageInterfaceOp(kind);
        bool found = false;

        // Mangled-name lookup can return more than one symbol after module composition. Retag only
        // an ordinary interface or an interface already carrying this exact stage identity; an
        // unrelated instruction with the same symbol is never a valid normalization candidate.
        for (auto symbol : symbols)
        {
            auto interfaceType = _findInterfaceType(symbol);
            if (!interfaceType)
                continue;
            if (interfaceType->getOp() != kIROp_InterfaceType &&
                interfaceType->getOp() != expectedOp)
            {
                continue;
            }

            // All stage-interface ops have the same storage and operand layout as
            // IRInterfaceType. The trusted-module load is the point where the ordinary
            // serialized interface receives its compiler-owned nominal identity.
            interfaceType->m_op = expectedOp;
            found = true;
        }

        if (!found)
        {
            if (outMissingStage)
                *outMissingStage = kind;
            return false;
        }
    }
    return true;
}

} // namespace Slang
