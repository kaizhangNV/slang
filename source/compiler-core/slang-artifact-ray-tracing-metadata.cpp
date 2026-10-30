#include "slang-artifact-associated-impl.h"

namespace Slang
{

// Retain the caller's advertised size when copying the currently supported prefix. A future
// caller may supply a larger struct; this version must not overwrite its unknown trailing data.
template<typename T>
static SlangResult copyMetalRTInfo(const T& source, T* destination)
{
    if (!destination || destination->structSize < sizeof(T))
        return SLANG_E_INVALID_ARG;
    auto size = destination->structSize;
    *destination = source;
    destination->structSize = size;
    return SLANG_OK;
}

uint32_t ArtifactPostEmitMetadata::getProgramCount()
{
    return uint32_t(m_metalRayTracingPrograms.getCount());
}

SlangResult ArtifactPostEmitMetadata::getProgramInfo(
    uint32_t programIndex,
    slang::MetalRayTracingProgramInfo* outInfo)
{
    if (programIndex >= getProgramCount())
        return SLANG_E_INVALID_ARG;
    auto& program = m_metalRayTracingPrograms[programIndex];
    SLANG_RETURN_ON_FAIL(copyMetalRTInfo(program.info, outInfo));
    outInfo->schemaTypeMangledName = program.schemaTypeMangledName.getBuffer();
    outInfo->resourceCount = uint32_t(program.resources.getCount());
    outInfo->functionCount = uint32_t(program.functions.getCount());
    return SLANG_OK;
}

SlangResult ArtifactPostEmitMetadata::findProgramBySchemaTypeMangledName(
    const char* schemaTypeMangledName,
    uint32_t* outProgramIndex)
{
    if (!schemaTypeMangledName || !outProgramIndex)
        return SLANG_E_INVALID_ARG;
    auto name = UnownedStringSlice(schemaTypeMangledName);
    for (Index i = 0; i < m_metalRayTracingPrograms.getCount(); ++i)
    {
        if (m_metalRayTracingPrograms[i].schemaTypeMangledName.getUnownedSlice() == name)
        {
            *outProgramIndex = uint32_t(i);
            return SLANG_OK;
        }
    }
    return SLANG_E_NOT_FOUND;
}

SlangResult ArtifactPostEmitMetadata::getProgramResourceInfo(
    uint32_t programIndex,
    uint32_t resourceIndex,
    slang::MetalRayTracingResourceInfo* outInfo)
{
    if (programIndex >= getProgramCount())
        return SLANG_E_INVALID_ARG;
    auto& resources = m_metalRayTracingPrograms[programIndex].resources;
    if (resourceIndex >= uint32_t(resources.getCount()))
        return SLANG_E_INVALID_ARG;
    return copyMetalRTInfo(resources[resourceIndex], outInfo);
}

SlangResult ArtifactPostEmitMetadata::getProgramFunctionInfo(
    uint32_t programIndex,
    uint32_t functionIndex,
    slang::MetalRayTracingFunctionInfo* outInfo)
{
    if (programIndex >= getProgramCount())
        return SLANG_E_INVALID_ARG;
    auto& functions = m_metalRayTracingPrograms[programIndex].functions;
    if (functionIndex >= uint32_t(functions.getCount()))
        return SLANG_E_INVALID_ARG;
    auto& function = functions[functionIndex];
    SLANG_RETURN_ON_FAIL(copyMetalRTInfo(function.info, outInfo));
    outInfo->entryPointName = function.entryPointName.getBuffer();
    return SLANG_OK;
}

} // namespace Slang
