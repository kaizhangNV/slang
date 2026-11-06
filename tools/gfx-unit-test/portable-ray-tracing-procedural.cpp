#include "core/slang-io.h"
#include "gfx-test-util.h"
#include "slang-rhi/acceleration-structure-utils.h"
#include "slang-rhi/shader-cursor.h"

using namespace rhi;
using namespace Slang;

namespace gfx_test
{

// Completes each AS build before releasing scratch storage. This keeps the fixture focused on
// candidate reporting and stage dispatch rather than asynchronous acceleration-structure lifetime.
static ComPtr<IAccelerationStructure> buildPortableProceduralAccelerationStructure(
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
    AccelerationStructureDesc structureDesc = {};
    structureDesc.kind = kind;
    structureDesc.size = sizes.accelerationStructureSize;
    ComPtr<IAccelerationStructure> structure;
    GFX_CHECK_CALL_ABORT(device->createAccelerationStructure(structureDesc, structure.writeRef()));
    auto encoder = queue->createCommandEncoder();
    encoder->buildAccelerationStructure(buildDesc, structure, nullptr, scratch, 0, nullptr);
    GFX_CHECK_CALL_ABORT(queue->submit(encoder->finish()));
    GFX_CHECK_CALL_ABORT(queue->waitOnHost());
    return structure;
}

// Runs the complete procedural hit group: Intersection reports two candidates, AnyHit rejects
// the near one, and ClosestHit observes the far candidate's attributes and shared payload. A
// second ray misses. Record is void so this does not require the separate RHI local-root update.
static void portableRayTracingProceduralTestImpl(IDevice* device, UnitTestContext* context)
{
    if (!device->hasFeature(Feature::RayTracing))
    {
        SLANG_IGNORE_TEST
    }
    auto globalSession = context->slangGlobalSession;
    const bool isD3D = device->getDeviceType() == DeviceType::D3D12;
    const bool isOptix = device->getDeviceType() == DeviceType::CUDA;
    slang::TargetDesc targetDesc = {};
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
    // Keep the retained AnyHit helper covered on targets that support it. The D3D path uses
    // ordinary inlining because DXC rejects legacy noinline IgnoreHit as well (Slang #14070).
    slang::PreprocessorMacroDesc noinlineFilter = {"TEST_NOINLINE_FILTER", "1"};
    if (!isD3D)
    {
        sessionDesc.preprocessorMacroCount = 1;
        sessionDesc.preprocessorMacros = &noinlineFilter;
    }
    ComPtr<slang::ISession> session;
    GFX_CHECK_CALL_ABORT(globalSession->createSession(sessionDesc, session.writeRef()));

    const char* sourcePath = "tests/ray-tracing-2/integrate/native-procedural.slang";
    String source;
    GFX_CHECK_CALL_ABORT(File::readAllText(sourcePath, source));
    ComPtr<slang::IBlob> diagnostics;
    auto module = session->loadModuleFromSourceString(
        "nativeProcedural",
        sourcePath,
        source.getBuffer(),
        diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    SLANG_CHECK_ABORT(module);
    auto sourceLayout = module->getLayout(0, diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    SLANG_CHECK_ABORT(sourceLayout);
    auto schema = sourceLayout->findTraceProgramSchema("Schema");
    SLANG_CHECK_ABORT(
        schema && schema->getHitGroupCount() == 1 && schema->getMissShaderCount() == 1);
    auto hit = schema->getHitGroup(0);
    auto miss = schema->getMissShader(0);
    SLANG_CHECK(hit->getFunctionIndex() == 0 && miss->getFunctionIndex() == 0);

    ComPtr<slang::IEntryPoint> entries[5];
    auto result = module->findAndCheckEntryPoint(
        "RayGeneration",
        SLANG_STAGE_RAY_GENERATION,
        entries[0].writeRef(),
        diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    GFX_CHECK_CALL_ABORT(result);

    // Discover source stages from the schema rather than duplicating the hit-group definition
    // on the host. Entry reflection below supplies the separately generated native export names.
    slang::RayTracingStageReflection* stages[] =
        {hit->getClosestHit(), hit->getAnyHit(), hit->getIntersection(), miss->getMiss()};
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
    slang::IComponentType* components[] =
        {module, entries[0], entries[1], entries[2], entries[3], entries[4]};
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
    SLANG_CHECK_ABORT(layout && layout->getEntryPointCount() == 5);
    ShaderProgramDesc programDesc = {};
    programDesc.slangGlobalScope = linked;
    ComPtr<IShaderProgram> program;
    result = device->createShaderProgram(programDesc, program.writeRef(), diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    GFX_CHECK_CALL_ABORT(result);

    const char* names[5];
    for (Index i = 0; i < SLANG_COUNT_OF(names); ++i)
    {
        names[i] = layout->getEntryPointByIndex(i)->getNameOverride();
        SLANG_CHECK_ABORT(names[i]);
    }
    const char* hitGroupName = "proceduralGroup";
    HitGroupDesc hitGroup = {};
    hitGroup.hitGroupName = hitGroupName;
    hitGroup.closestHitEntryPoint = names[1];
    hitGroup.anyHitEntryPoint = names[2];
    hitGroup.intersectionEntryPoint = names[3];
    RayTracingPipelineDesc pipelineDesc = {};
    pipelineDesc.program = program;
    pipelineDesc.hitGroups = &hitGroup;
    pipelineDesc.hitGroupCount = 1;
    pipelineDesc.maxRayPayloadSize = sizeof(uint32_t) * 4;
    pipelineDesc.maxAttributeSizeInBytes = sizeof(uint32_t) * 2;
    pipelineDesc.maxRecursion = 1;
    ComPtr<IRayTracingPipeline> pipeline;
    GFX_CHECK_CALL_ABORT(device->createRayTracingPipeline(pipelineDesc, pipeline.writeRef()));
    ShaderTableDesc tableDesc = {};
    tableDesc.program = program;
    tableDesc.rayGenShaderCount = 1;
    tableDesc.rayGenShaderEntryPointNames = &names[0];
    tableDesc.hitGroupCount = 1;
    tableDesc.hitGroupNames = &hitGroupName;
    tableDesc.missShaderCount = 1;
    tableDesc.missShaderEntryPointNames = &names[4];
    ComPtr<IShaderTable> table;
    GFX_CHECK_CALL_ABORT(device->createShaderTable(tableDesc, table.writeRef()));

    auto queue = device->getQueue(QueueType::Graphics);
    const AccelerationStructureAABB bounds = {-1.0f, -1.0f, 0.5f, 1.0f, 1.0f, 3.5f};
    BufferDesc boundsDesc = {};
    boundsDesc.size = sizeof(bounds);
    boundsDesc.usage = BufferUsage::AccelerationStructureBuildInput;
    boundsDesc.defaultState = ResourceState::AccelerationStructureBuildInput;
    ComPtr<IBuffer> boundsBuffer;
    GFX_CHECK_CALL_ABORT(device->createBuffer(boundsDesc, &bounds, boundsBuffer.writeRef()));
    AccelerationStructureBuildInput boxInput = {};
    boxInput.type = AccelerationStructureBuildInputType::ProceduralPrimitives;
    boxInput.proceduralPrimitives.aabbBuffers[0] = boundsBuffer;
    boxInput.proceduralPrimitives.aabbBufferCount = 1;
    boxInput.proceduralPrimitives.aabbStride = sizeof(bounds);
    boxInput.proceduralPrimitives.primitiveCount = 1;
    // Deliberately not opaque: the hardware must invoke AnyHit for each reported candidate.
    auto blas = buildPortableProceduralAccelerationStructure(
        device,
        queue,
        boxInput,
        AccelerationStructureKind::BottomLevel);

    AccelerationStructureInstanceDescGeneric instance = {};
    instance.transform[0][0] = 1.0f;
    instance.transform[1][1] = 1.0f;
    instance.transform[2][2] = 1.0f;
    instance.instanceMask = 0xff;
    instance.accelerationStructure = blas->getHandle();
    auto instanceType = getAccelerationStructureInstanceDescType(device);
    auto instanceSize = getAccelerationStructureInstanceDescSize(instanceType);
    List<uint8_t> instanceData;
    instanceData.setCount(instanceSize);
    convertAccelerationStructureInstanceDesc(instanceType, instanceData.getBuffer(), &instance);
    BufferDesc instanceDesc = {};
    instanceDesc.size = instanceSize;
    instanceDesc.usage = BufferUsage::AccelerationStructureBuildInput;
    instanceDesc.defaultState = ResourceState::AccelerationStructureBuildInput;
    ComPtr<IBuffer> instanceBuffer;
    GFX_CHECK_CALL_ABORT(
        device->createBuffer(instanceDesc, instanceData.getBuffer(), instanceBuffer.writeRef()));
    AccelerationStructureBuildInput instanceInput = {};
    instanceInput.type = AccelerationStructureBuildInputType::Instances;
    instanceInput.instances.instanceBuffer = instanceBuffer;
    instanceInput.instances.instanceCount = 1;
    instanceInput.instances.instanceStride = instanceSize;
    auto tlas = buildPortableProceduralAccelerationStructure(
        device,
        queue,
        instanceInput,
        AccelerationStructureKind::TopLevel);

    uint32_t values[3][4] = {};
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

    const uint32_t expected[3][4] = {
        {3, 29, 3, 9}, // Far distance, far attributes, both AnyHit visits, far hit kind.
        {0, 0, 0, 99}, // Miss.
        {0, 1, 123, 0} // ReportHit returned false for near and true for far.
    };
    for (Index ray = 0; ray < 3; ++ray)
        for (Index component = 0; component < 4; ++component)
            SLANG_CHECK(values[ray][component] == expected[ray][component]);
}

SLANG_UNIT_TEST(portableRayTracingProceduralVulkan)
{
    runTestImpl(portableRayTracingProceduralTestImpl, unitTestContext, DeviceType::Vulkan);
}

SLANG_UNIT_TEST(portableRayTracingProceduralD3D12)
{
    runTestImpl(portableRayTracingProceduralTestImpl, unitTestContext, DeviceType::D3D12);
}

SLANG_UNIT_TEST(portableRayTracingProceduralOptix)
{
    runTestImpl(portableRayTracingProceduralTestImpl, unitTestContext, DeviceType::CUDA);
}

} // namespace gfx_test
