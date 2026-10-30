#include "slang-ir-metal-portable-ray-tracing-metadata.h"

#include "compiler-core/slang-artifact-associated-impl.h"
#include "slang-ir-insts.h"
#include "slang-ir-metal-portable-ray-tracing-target.h"
#include "slang-ir-util.h"
#include "slang-profile.h"

namespace Slang
{

void addMetalRTProgramMetadata(
    IRModule* module,
    IRStructType* resources,
    UInt tags,
    IRIntegerValue maxLevels,
    IRIntegerValue hitStride,
    IRIntegerValue missStride,
    IRIntegerValue callableStride,
    UInt hitGroupCount,
    UInt missShaderCount,
    UInt callableShaderCount,
    UInt candidateFunctionTableSize)
{
    SLANG_RELEASE_ASSERT(!resources->findDecoration<IRMetalRTProgramMetadataDecoration>());
    IRBuilder builder(module);
    builder.setInsertInto(module);
    auto uintType = builder.getUIntType();
    // ABI creation precedes paired descriptor type/layout physicalization. That later producer
    // supplies the checked AST schema name; an empty string is only this construction-phase
    // placeholder, never a public lookup name. Collection requires the transfer to be complete.
    // Synthetic fixtures may already have attached the canonical name before creating the ABI.
    auto sourceName = resources->findDecoration<IRPortableRTSchemaNameDecoration>();
    IRInst* operands[] = {
        sourceName ? sourceName->getSchemaTypeMangledName()
                   : builder.getStringValue(UnownedStringSlice("")),
        builder.getIntValue(uintType, tags),
        builder.getIntValue(uintType, maxLevels),
        builder.getIntValue(uintType, hitStride),
        builder.getIntValue(uintType, missStride),
        builder.getIntValue(uintType, callableStride),
        builder.getIntValue(uintType, hitGroupCount),
        builder.getIntValue(uintType, missShaderCount),
        builder.getIntValue(uintType, callableShaderCount),
        builder.getIntValue(uintType, candidateFunctionTableSize)};
    auto program = builder.emitIntrinsicInst(
        builder.getVoidType(),
        kIROp_MetalRTProgramMetadata,
        SLANG_COUNT_OF(operands),
        operands);
    builder.addDecoration(resources, kIROp_MetalRTProgramMetadataDecoration, program);
}

void addMetalRTFunctionMetadata(
    IRModule* module,
    IRFunc* function,
    IRStructType* resources,
    UInt functionIndex,
    UInt requirements)
{
    SLANG_RELEASE_ASSERT(!function->findDecoration<IRMetalRTFunctionMetadataDecoration>());
    IRBuilder builder(module);
    IRInst* operands[] = {
        resources->findDecoration<IRMetalRTProgramMetadataDecoration>()->getProgram(),
        builder.getIntValue(builder.getUIntType(), functionIndex),
        builder.getIntValue(builder.getUIntType(), requirements)};
    builder.addDecoration(function, kIROp_MetalRTFunctionMetadataDecoration, operands, 3);
}

namespace
{

// Resource roles have an independent public enum. Keep its mapping explicit: clients bind the
// reflected argument-buffer IDs, not the numerical value of a public or internal enum.
slang::MetalRayTracingResourceKind getPublicResourceKind(MetalRTDescriptorResource resource)
{
    using Kind = slang::MetalRayTracingResourceKind;
    switch (resource)
    {
    case MetalRTDescriptorResource::IntersectionFunctionTable:
        return Kind::IntersectionFunctionTable;
    case MetalRTDescriptorResource::MissVisibleFunctionTable:
        return Kind::MissVisibleFunctionTable;
    case MetalRTDescriptorResource::ClosestHitVisibleFunctionTable:
        return Kind::ClosestHitVisibleFunctionTable;
    case MetalRTDescriptorResource::CallableVisibleFunctionTable:
        return Kind::CallableVisibleFunctionTable;
    case MetalRTDescriptorResource::Records:
        return Kind::Records;
    case MetalRTDescriptorResource::AnyHitVisibleFunctionTable:
        return Kind::AnyHitVisibleFunctionTable;
    case MetalRTDescriptorResource::IntersectionVisibleFunctionTable:
        return Kind::IntersectionVisibleFunctionTable;
    default:
        SLANG_UNEXPECTED("Invalid Metal descriptor resource role");
    }
}

// The executable adapter owns its role; no source stage names or synthesized symbol prefixes
// participate in this classification. An IFT function can combine source Intersection and AnyHit.
MetalRTDescriptorResource getFunctionResource(IRFunc* function)
{
    if (function->findDecoration<IRMetalIntersectionFunctionDecoration>())
        return MetalRTDescriptorResource::IntersectionFunctionTable;
    auto visible = function->findDecoration<IRMetalVisibleFunctionDecoration>();
    SLANG_RELEASE_ASSERT(visible);
    switch (Stage(visible->getStageKind()->getValue()))
    {
    case Stage::Miss:
        return MetalRTDescriptorResource::MissVisibleFunctionTable;
    case Stage::ClosestHit:
        return MetalRTDescriptorResource::ClosestHitVisibleFunctionTable;
    case Stage::Callable:
        return MetalRTDescriptorResource::CallableVisibleFunctionTable;
    case Stage::AnyHit:
        return MetalRTDescriptorResource::AnyHitVisibleFunctionTable;
    case Stage::Intersection:
        return MetalRTDescriptorResource::IntersectionVisibleFunctionTable;
    default:
        SLANG_UNEXPECTED("Invalid Metal visible-function stage");
    }
}

// Capacity is a schema property. A ray-generation artifact can require a full table while
// exporting none of its providers, and a separately compiled provider can occupy a later slot.
UInt getFunctionTableSize(IRMetalRTProgramMetadata* program, MetalRTDescriptorResource resource)
{
    switch (resource)
    {
    case MetalRTDescriptorResource::IntersectionFunctionTable:
        return UInt(program->getCandidateFunctionTableSize()->getValue());
    case MetalRTDescriptorResource::MissVisibleFunctionTable:
        return UInt(program->getMissShaderCount()->getValue());
    case MetalRTDescriptorResource::ClosestHitVisibleFunctionTable:
    case MetalRTDescriptorResource::AnyHitVisibleFunctionTable:
    case MetalRTDescriptorResource::IntersectionVisibleFunctionTable:
        return UInt(program->getHitGroupCount()->getValue());
    case MetalRTDescriptorResource::CallableVisibleFunctionTable:
        return UInt(program->getCallableShaderCount()->getValue());
    case MetalRTDescriptorResource::Records:
        return 0;
    default:
        SLANG_UNEXPECTED("Invalid Metal descriptor resource role");
    }
}

} // namespace

void collectMetalRTMetadata(const IRModule* module, ArtifactPostEmitMetadata& metadata)
{
    // A program record has stable identity even when resource legalization splits its descriptor
    // into ordinary and resource-bearing parts. Strings become public lookup keys only here.
    Dictionary<IRMetalRTProgramMetadata*, Index> programIndices;
    for (auto inst : module->getGlobalInsts())
    {
        auto decoration = as<IRMetalRTProgramMetadata>(inst);
        if (!decoration)
            continue;
        MetalRTProgramMetadata program;
        program.schemaTypeMangledName = decoration->getSchemaTypeMangledName()->getStringSlice();
        SLANG_RELEASE_ASSERT(program.schemaTypeMangledName.getLength());
        program.info.tagMask = uint32_t(decoration->getTagMask()->getValue());
        program.info.maxLevels = uint32_t(decoration->getMaxLevels()->getValue());
        program.info.hitRecordStride = uint64_t(decoration->getHitRecordStride()->getValue());
        program.info.missRecordStride = uint64_t(decoration->getMissRecordStride()->getValue());
        program.info.callableRecordStride =
            uint64_t(decoration->getCallableRecordStride()->getValue());
        program.info.recordHeaderSize = kMetalRTRecordHeaderSize;
        program.info.recordAlignment = kMetalRTRecordAlignment;
        for (UInt i = 0; i < UInt(MetalRTDescriptorResource::Count); ++i)
        {
            slang::MetalRayTracingResourceInfo resource;
            resource.kind = getPublicResourceKind(MetalRTDescriptorResource(i));
            resource.argumentBufferID = uint32_t(i);
            resource.functionTableSize =
                uint32_t(getFunctionTableSize(decoration, MetalRTDescriptorResource(i)));
            program.resources.add(resource);
        }
        programIndices.add(decoration, metadata.m_metalRayTracingPrograms.getCount());
        metadata.m_metalRayTracingPrograms.add(_Move(program));
    }

    for (auto inst : module->getGlobalInsts())
    {
        auto function = as<IRFunc>(inst);
        if (!function)
            continue;
        auto decoration = function->findDecoration<IRMetalRTFunctionMetadataDecoration>();
        if (!decoration)
            continue;
        auto programIndex = programIndices.tryGetValue(decoration->getProgram());
        SLANG_RELEASE_ASSERT(programIndex);
        auto& program = metadata.m_metalRayTracingPrograms[*programIndex];
        MetalRTFunctionMetadata entry;
        auto externalName = function->findDecoration<IRExternCppDecoration>();
        SLANG_RELEASE_ASSERT(externalName);
        entry.entryPointName = externalName->getName();
        auto resource = getFunctionResource(function);
        entry.info.tableKind = getPublicResourceKind(resource);
        entry.info.functionIndex = uint32_t(decoration->getFunctionIndex()->getValue());
        entry.info.requirements = uint32_t(decoration->getRequirements()->getValue());
        if (auto candidate = function->findDecoration<IRMetalIntersectionFunctionDecoration>())
            entry.info.primitive = MetalRTGeometryKind(candidate->getGeometryKind()->getValue());
        for (auto param : function->getParams())
        {
            entry.info.usesGlobalContext |=
                param->findDecoration<IRMetalRTGlobalResourceContextDecoration>() != nullptr;
            if (param->findDecoration<IRMetalRTCandidateDescriptorDecoration>())
            {
                auto layout = findVarLayout(param);
                SLANG_RELEASE_ASSERT(layout);
                auto offset = layout->findOffsetAttr(LayoutResourceKind::MetalBuffer);
                SLANG_RELEASE_ASSERT(offset);
                auto binding = uint32_t(offset->getOffset());
                SLANG_RELEASE_ASSERT(
                    program.info.candidateDescriptorBufferIndex == UINT32_MAX ||
                    program.info.candidateDescriptorBufferIndex == binding);
                program.info.candidateDescriptorBufferIndex = binding;
            }
        }
        auto& table = program.resources[UInt(resource)];
        SLANG_RELEASE_ASSERT(entry.info.functionIndex < table.functionTableSize);
        program.functions.add(_Move(entry));
    }
}

} // namespace Slang
