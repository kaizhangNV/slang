#include "core/slang-io.h"
#include "gfx-test-util.h"
#include "slang-rhi/shader-cursor.h"

using namespace rhi;
using namespace Slang;

namespace gfx_test
{

// Runs a real native callable SBT dispatch without acceleration-structure construction. This
// isolates portable-stage adaptation, callable-data aliasing, and descriptor erasure from the
// larger traversal tests. Record is void, so this does not depend on local-root-signature support.
static void portableRayTracingCallableTestImpl(IDevice* device, UnitTestContext* context)
{
    if (!device->hasFeature(Feature::RayTracing))
    {
        SLANG_IGNORE_TEST
    }

    // Use a test-local session to enable the experimental module without changing the shared
    // RHI test device or enabling the feature for unrelated tests. Its target matches the device.
    auto globalSession = context->slangGlobalSession;
    const bool isD3D = device->getDeviceType() == DeviceType::D3D12;
    slang::TargetDesc targetDesc = {};
    targetDesc.format = isD3D ? SLANG_DXIL : SLANG_SPIRV;
    targetDesc.profile = globalSession->findProfile(isD3D ? "sm_6_6" : "spirv_1_5");
    slang::CompilerOptionEntry experimental = {};
    experimental.name = slang::CompilerOptionName::ExperimentalFeature;
    experimental.value.kind = slang::CompilerOptionValueKind::Int;
    experimental.value.intValue0 = 1;
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;
    sessionDesc.compilerOptionEntryCount = 1;
    sessionDesc.compilerOptionEntries = &experimental;
    ComPtr<slang::ISession> session;
    GFX_CHECK_CALL_ABORT(globalSession->createSession(sessionDesc, session.writeRef()));

    const char* sourcePath = "tests/ray-tracing-2/integrate/native-callable.slang";
    String source;
    GFX_CHECK_CALL_ABORT(File::readAllText(sourcePath, source));
    ComPtr<slang::IBlob> diagnostics;
    auto module = session->loadModuleFromSourceString(
        "nativeCallable",
        sourcePath,
        source.getBuffer(),
        diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    SLANG_CHECK_ABORT(module != nullptr);

    ComPtr<slang::IEntryPoint> rayGeneration;
    auto result = module->findAndCheckEntryPoint(
        "RayGeneration",
        SLANG_STAGE_RAY_GENERATION,
        rayGeneration.writeRef(),
        diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    GFX_CHECK_CALL_ABORT(result);
    ComPtr<slang::IEntryPoint> callable;
    result = module->findAndCheckEntryPoint(
        "AddThirtyOne",
        SLANG_STAGE_CALLABLE,
        callable.writeRef(),
        diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    GFX_CHECK_CALL_ABORT(result);

    slang::IComponentType* components[] = {module, rayGeneration, callable};
    ComPtr<slang::IComponentType> composite;
    result = session->createCompositeComponentType(
        components,
        SLANG_COUNT_OF(components),
        composite.writeRef(),
        diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    GFX_CHECK_CALL_ABORT(result);
    ComPtr<slang::IComponentType> linked;
    result = composite->link(linked.writeRef(), diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    GFX_CHECK_CALL_ABORT(result);

    auto layout = linked->getLayout(0, diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    SLANG_CHECK_ABORT(layout != nullptr && layout->getEntryPointCount() == 2);

    ShaderProgramDesc programDesc = {};
    programDesc.slangGlobalScope = linked;
    ComPtr<IShaderProgram> program;
    result = device->createShaderProgram(programDesc, program.writeRef(), diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    GFX_CHECK_CALL_ABORT(result);

    RayTracingPipelineDesc pipelineDesc = {};
    pipelineDesc.program = program;
    pipelineDesc.maxRayPayloadSize = sizeof(uint32_t);
    pipelineDesc.maxAttributeSizeInBytes = 8;
    pipelineDesc.maxRecursion = 1;
    ComPtr<IRayTracingPipeline> pipeline;
    GFX_CHECK_CALL_ABORT(device->createRayTracingPipeline(pipelineDesc, pipeline.writeRef()));

    // Selection names identify source declarations; target exports are compiler-generated.
    // Ordinary entry-point reflection supplies the names for the SBT, so the host does not
    // duplicate the compiler's stage suffix convention or need new schema-reflection APIs.
    const char* rayGenerationName = layout->getEntryPointByIndex(0)->getNameOverride();
    const char* callableName = layout->getEntryPointByIndex(1)->getNameOverride();
    SLANG_CHECK_ABORT(rayGenerationName != nullptr && callableName != nullptr);
    ShaderTableDesc tableDesc = {};
    tableDesc.program = program;
    tableDesc.rayGenShaderCount = 1;
    tableDesc.rayGenShaderEntryPointNames = &rayGenerationName;
    tableDesc.callableShaderCount = 1;
    tableDesc.callableShaderEntryPointNames = &callableName;
    ComPtr<IShaderTable> table;
    GFX_CHECK_CALL_ABORT(device->createShaderTable(tableDesc, table.writeRef()));

    uint32_t initialValue = 0;
    BufferDesc outputDesc = {};
    outputDesc.size = sizeof(initialValue);
    outputDesc.elementSize = sizeof(initialValue);
    outputDesc.usage =
        BufferUsage::UnorderedAccess | BufferUsage::CopySource | BufferUsage::CopyDestination;
    outputDesc.defaultState = ResourceState::UnorderedAccess;
    ComPtr<IBuffer> output;
    GFX_CHECK_CALL_ABORT(device->createBuffer(outputDesc, &initialValue, output.writeRef()));

    auto queue = device->getQueue(QueueType::Graphics);
    auto commandEncoder = queue->createCommandEncoder();
    auto pass = commandEncoder->beginRayTracingPass();
    auto rootObject = pass->bindPipeline(pipeline, table);
    GFX_CHECK_CALL_ABORT(ShaderCursor(rootObject)["resultBuffer"].setBinding(output));
    pass->dispatchRays(0, 1, 1, 1);
    pass->end();
    GFX_CHECK_CALL_ABORT(queue->submit(commandEncoder->finish()));
    GFX_CHECK_CALL_ABORT(queue->waitOnHost());

    uint32_t actualValue = 0;
    GFX_CHECK_CALL_ABORT(device->readBuffer(output, 0, sizeof(actualValue), &actualValue));
    SLANG_CHECK(actualValue == 42);
}

SLANG_UNIT_TEST(portableRayTracingCallableVulkan)
{
    runTestImpl(portableRayTracingCallableTestImpl, unitTestContext, DeviceType::Vulkan);
}

SLANG_UNIT_TEST(portableRayTracingCallableD3D12)
{
    runTestImpl(portableRayTracingCallableTestImpl, unitTestContext, DeviceType::D3D12);
}

} // namespace gfx_test
