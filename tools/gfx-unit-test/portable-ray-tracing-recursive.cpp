#include "core/slang-io.h"
#include "gfx-test-util.h"
#include "slang-rhi/acceleration-structure-utils.h"
#include "slang-rhi/shader-cursor.h"

using namespace rhi;
using namespace Slang;

namespace gfx_test
{

// Builds one acceleration structure and waits before releasing its scratch buffer. Both the
// triangle BLAS and single-instance TLAS use this path; no compaction or asynchronous lifetime
// management is needed for a test whose purpose is shader adaptation, not AS construction.
static ComPtr<IAccelerationStructure> buildPortableRecursiveAccelerationStructure(
    IDevice* device,
    ICommandQueue* queue,
    const AccelerationStructureBuildInput& input,
    AccelerationStructureKind kind)
{
    AccelerationStructureBuildDesc buildDesc = {};
    buildDesc.inputs = &input;
    buildDesc.inputCount = 1;
    AccelerationStructureSizes sizes = {};
    GFX_CHECK_CALL_ABORT(device->getAccelerationStructureSizes(buildDesc, &sizes));

    BufferDesc scratchDesc = {};
    scratchDesc.size = sizes.scratchSize;
    scratchDesc.usage = BufferUsage::UnorderedAccess;
    scratchDesc.defaultState = ResourceState::UnorderedAccess;
    ComPtr<IBuffer> scratch;
    GFX_CHECK_CALL_ABORT(device->createBuffer(scratchDesc, nullptr, scratch.writeRef()));

    AccelerationStructureDesc accelerationStructureDesc = {};
    accelerationStructureDesc.kind = kind;
    accelerationStructureDesc.size = sizes.accelerationStructureSize;
    ComPtr<IAccelerationStructure> accelerationStructure;
    GFX_CHECK_CALL_ABORT(device->createAccelerationStructure(
        accelerationStructureDesc,
        accelerationStructure.writeRef()));
    auto encoder = queue->createCommandEncoder();
    encoder->buildAccelerationStructure(
        buildDesc,
        accelerationStructure,
        nullptr,
        scratch,
        0,
        nullptr);
    GFX_CHECK_CALL_ABORT(queue->submit(encoder->finish()));
    GFX_CHECK_CALL_ABORT(queue->waitOnHost());
    return accelerationStructure;
}

// Traces two outer triangle hits, each issuing a secondary trace from ClosestHit into Miss.
// The outer float4 payload and inner uint payload remain distinct, and both invocation levels
// observe the original launch coordinates. Record is void to isolate native recursion behavior.
static void portableRayTracingRecursiveTestImpl(IDevice* device, UnitTestContext* context)
{
    if (!device->hasFeature(Feature::RayTracing))
    {
        SLANG_IGNORE_TEST
    }

    // Enable the experimental module only for this test's compilation session.
    auto globalSession = context->slangGlobalSession;
    const bool isD3D = device->getDeviceType() == DeviceType::D3D12;
    slang::TargetDesc targetDesc = {};
    const bool isOptix = device->getDeviceType() == DeviceType::CUDA;
    targetDesc.format = isOptix ? SLANG_PTX : (isD3D ? SLANG_DXIL : SLANG_SPIRV);
    if (!isOptix)
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
    // Native TraceRay in a retained helper validates as DXIL, but pipeline creation crashes on
    // the Windows runner even with released Slang and a raw D3D12 host (Slang #14077).
    // Keep recursive runtime coverage there with ordinary inlining; Vulkan/OptiX and the DXIL
    // compile-only regression retain the helper. No compiler workaround is introduced.
    slang::PreprocessorMacroDesc inlineTrace = {"TEST_NOINLINE_TRACE", "0"};
    if (isD3D)
    {
        sessionDesc.preprocessorMacroCount = 1;
        sessionDesc.preprocessorMacros = &inlineTrace;
    }
    ComPtr<slang::ISession> session;
    GFX_CHECK_CALL_ABORT(globalSession->createSession(sessionDesc, session.writeRef()));

    const char* sourcePath = "tests/ray-tracing-2/integrate/native-recursive-trace.slang";
    String source;
    GFX_CHECK_CALL_ABORT(File::readAllText(sourcePath, source));
    ComPtr<slang::IBlob> diagnostics;
    auto module = session->loadModuleFromSourceString(
        "nativeRecursive",
        sourcePath,
        source.getBuffer(),
        diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    SLANG_CHECK_ABORT(module != nullptr);

    // The schema supplies stage identities; the host does not duplicate generated export names.
    auto sourceLayout = module->getLayout(0, diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    SLANG_CHECK_ABORT(sourceLayout);
    auto schema = sourceLayout->findTraceProgramSchema("Schema");
    SLANG_CHECK_ABORT(
        schema && schema->getHitGroupCount() == 1 && schema->getMissShaderCount() == 1);
    auto hit = schema->getHitGroup(0);
    auto reflectedMiss = schema->getMissShader(0);
    SLANG_CHECK(hit->getFunctionIndex() == 0 && reflectedMiss->getFunctionIndex() == 0);

    ComPtr<slang::IEntryPoint> entries[3];
    auto result = module->findAndCheckEntryPoint(
        "RayGeneration",
        SLANG_STAGE_RAY_GENERATION,
        entries[0].writeRef(),
        diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    GFX_CHECK_CALL_ABORT(result);
    slang::RayTracingStageReflection* stages[] = {hit->getClosestHit(), reflectedMiss->getMiss()};
    for (Index i = 0; i < SLANG_COUNT_OF(stages); ++i)
    {
        SLANG_CHECK_ABORT(stages[i]);
        ComPtr<slang::IBlob> sourceName;
        GFX_CHECK_CALL_ABORT(stages[i]->getType()->getFullName(sourceName.writeRef()));
        result = module->findAndCheckEntryPoint(
            static_cast<const char*>(sourceName->getBufferPointer()),
            stages[i]->getStage(),
            entries[i + 1].writeRef(),
            diagnostics.writeRef());
        diagnoseIfNeeded(diagnostics);
        GFX_CHECK_CALL_ABORT(result);
    }

    slang::IComponentType* components[] = {module, entries[0], entries[1], entries[2]};
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
    SLANG_CHECK_ABORT(layout != nullptr && layout->getEntryPointCount() == 3);

    ShaderProgramDesc programDesc = {};
    programDesc.slangGlobalScope = linked;
    ComPtr<IShaderProgram> program;
    result = device->createShaderProgram(programDesc, program.writeRef(), diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    GFX_CHECK_CALL_ABORT(result);

    // Source selection names and target export names need not be equal. Reflection, rather
    // than a hardcoded generated suffix, determines every stage name used by the pipeline/SBT.
    const char* rayGenerationName = layout->getEntryPointByIndex(0)->getNameOverride();
    const char* closestHitName = layout->getEntryPointByIndex(1)->getNameOverride();
    const char* missName = layout->getEntryPointByIndex(2)->getNameOverride();
    SLANG_CHECK_ABORT(rayGenerationName && closestHitName && missName);
    const char* hitGroupName = "triangleGroup";
    HitGroupDesc hitGroup = {};
    hitGroup.hitGroupName = hitGroupName;
    hitGroup.closestHitEntryPoint = closestHitName;
    RayTracingPipelineDesc pipelineDesc = {};
    pipelineDesc.program = program;
    pipelineDesc.hitGroups = &hitGroup;
    pipelineDesc.hitGroupCount = 1;
    pipelineDesc.maxRayPayloadSize = sizeof(float) * 4;
    pipelineDesc.maxAttributeSizeInBytes = sizeof(float) * 2;
    pipelineDesc.maxRecursion = 2;
    ComPtr<IRayTracingPipeline> pipeline;
    GFX_CHECK_CALL_ABORT(device->createRayTracingPipeline(pipelineDesc, pipeline.writeRef()));

    ShaderTableDesc tableDesc = {};
    tableDesc.program = program;
    tableDesc.rayGenShaderCount = 1;
    tableDesc.rayGenShaderEntryPointNames = &rayGenerationName;
    tableDesc.hitGroupCount = 1;
    tableDesc.hitGroupNames = &hitGroupName;
    tableDesc.missShaderCount = 1;
    tableDesc.missShaderEntryPointNames = &missName;
    ComPtr<IShaderTable> table;
    GFX_CHECK_CALL_ABORT(device->createShaderTable(tableDesc, table.writeRef()));

    auto queue = device->getQueue(QueueType::Graphics);
    const float vertices[][3] = {{0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 1.0f}};
    BufferDesc vertexDesc = {};
    vertexDesc.size = sizeof(vertices);
    vertexDesc.usage = BufferUsage::AccelerationStructureBuildInput;
    vertexDesc.defaultState = ResourceState::AccelerationStructureBuildInput;
    ComPtr<IBuffer> vertexBuffer;
    GFX_CHECK_CALL_ABORT(device->createBuffer(vertexDesc, vertices, vertexBuffer.writeRef()));
    AccelerationStructureBuildInput triangleInput = {};
    triangleInput.type = AccelerationStructureBuildInputType::Triangles;
    triangleInput.triangles.vertexBuffers[0] = vertexBuffer;
    triangleInput.triangles.vertexBufferCount = 1;
    triangleInput.triangles.vertexFormat = Format::RGB32Float;
    triangleInput.triangles.vertexCount = SLANG_COUNT_OF(vertices);
    triangleInput.triangles.vertexStride = sizeof(vertices[0]);
    triangleInput.triangles.flags = AccelerationStructureGeometryFlags::Opaque;
    auto blas = buildPortableRecursiveAccelerationStructure(
        device,
        queue,
        triangleInput,
        AccelerationStructureKind::BottomLevel);

    AccelerationStructureInstanceDescGeneric instance = {};
    instance.transform[0][0] = 1.0f;
    instance.transform[1][1] = 1.0f;
    instance.transform[2][2] = 1.0f;
    instance.instanceMask = 0xff;
    instance.accelerationStructure = blas->getHandle();
    const auto instanceDescType = getAccelerationStructureInstanceDescType(device);
    const auto instanceDescSize = getAccelerationStructureInstanceDescSize(instanceDescType);
    List<uint8_t> nativeInstance;
    nativeInstance.setCount(instanceDescSize);
    convertAccelerationStructureInstanceDesc(
        instanceDescType,
        nativeInstance.getBuffer(),
        &instance);
    BufferDesc instanceBufferDesc = {};
    instanceBufferDesc.size = instanceDescSize;
    instanceBufferDesc.usage = BufferUsage::AccelerationStructureBuildInput;
    instanceBufferDesc.defaultState = ResourceState::AccelerationStructureBuildInput;
    ComPtr<IBuffer> instanceBuffer;
    GFX_CHECK_CALL_ABORT(device->createBuffer(
        instanceBufferDesc,
        nativeInstance.getBuffer(),
        instanceBuffer.writeRef()));
    AccelerationStructureBuildInput instanceInput = {};
    instanceInput.type = AccelerationStructureBuildInputType::Instances;
    instanceInput.instances.instanceBuffer = instanceBuffer;
    instanceInput.instances.instanceCount = 1;
    instanceInput.instances.instanceStride = instanceDescSize;
    auto tlas = buildPortableRecursiveAccelerationStructure(
        device,
        queue,
        instanceInput,
        AccelerationStructureKind::TopLevel);

    float values[2][4] = {};
    BufferDesc outputDesc = {};
    outputDesc.size = sizeof(values);
    outputDesc.elementSize = sizeof(values[0]);
    outputDesc.usage =
        BufferUsage::UnorderedAccess | BufferUsage::CopySource | BufferUsage::CopyDestination;
    outputDesc.defaultState = ResourceState::UnorderedAccess;
    ComPtr<IBuffer> output;
    GFX_CHECK_CALL_ABORT(device->createBuffer(outputDesc, values, output.writeRef()));

    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginRayTracingPass();
    auto rootObject = pass->bindPipeline(pipeline, table);
    GFX_CHECK_CALL_ABORT(ShaderCursor(rootObject)["sceneBVH"].setBinding(tlas));
    GFX_CHECK_CALL_ABORT(ShaderCursor(rootObject)["resultBuffer"].setBinding(output));
    pass->dispatchRays(0, 2, 1, 1);
    pass->end();
    GFX_CHECK_CALL_ABORT(queue->submit(encoder->finish()));
    GFX_CHECK_CALL_ABORT(queue->waitOnHost());
    GFX_CHECK_CALL_ABORT(device->readBuffer(output, 0, sizeof(values), values));

    // The first component was written before recursion; the other components after it. The
    // inner value includes launch width 2, and the outer builtin reads still describe its hit.
    SLANG_CHECK(values[0][0] == 100.0f);
    SLANG_CHECK(values[0][1] == 215.0f);
    SLANG_CHECK(values[0][2] == 1.0f);
    SLANG_CHECK(values[0][3] == 2.25f);
    SLANG_CHECK(values[1][0] == 101.0f);
    SLANG_CHECK(values[1][1] == 216.0f);
    SLANG_CHECK(values[1][2] == 1.0f);
    SLANG_CHECK(values[1][3] == 2.25f);
}

SLANG_UNIT_TEST(portableRayTracingRecursiveVulkan)
{
    runTestImpl(portableRayTracingRecursiveTestImpl, unitTestContext, DeviceType::Vulkan);
}

SLANG_UNIT_TEST(portableRayTracingRecursiveD3D12)
{
    runTestImpl(portableRayTracingRecursiveTestImpl, unitTestContext, DeviceType::D3D12);
}

SLANG_UNIT_TEST(portableRayTracingRecursiveOptix)
{
    runTestImpl(portableRayTracingRecursiveTestImpl, unitTestContext, DeviceType::CUDA);
}

} // namespace gfx_test
