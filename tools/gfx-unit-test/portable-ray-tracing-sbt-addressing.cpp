#include "core/slang-io.h"
#include "gfx-test-util.h"
#include "slang-rhi/acceleration-structure-utils.h"
#include "slang-rhi/shader-cursor.h"

using namespace rhi;
using namespace Slang;

namespace gfx_test
{

// Builds all supplied geometries into one AS and waits before releasing scratch storage. A
// multi-geometry BLAS is essential here: two separate instances would not test geometryIndex.
static ComPtr<IAccelerationStructure> buildPortableAddressingAccelerationStructure(
    IDevice* device,
    ICommandQueue* queue,
    const AccelerationStructureBuildInput* inputs,
    uint32_t inputCount,
    AccelerationStructureKind kind)
{
    AccelerationStructureBuildDesc buildDesc = {};
    buildDesc.inputs = inputs;
    buildDesc.inputCount = inputCount;
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

// Distinguishes schema function indices from host SBT record indices. Sentinel shaders occupy
// unused records, so dropping any addressing term changes the result instead of still selecting
// the same shader. Record is void; this test needs no separate D3D local-root integration.
static void portableRayTracingSbtAddressingTestImpl(IDevice* device, UnitTestContext* context)
{
    if (!device->hasFeature(Feature::RayTracing))
    {
        SLANG_IGNORE_TEST
    }
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

    const char* sourcePath = "tests/ray-tracing-2/integrate/native-sbt-addressing.slang";
    String source;
    GFX_CHECK_CALL_ABORT(File::readAllText(sourcePath, source));
    ComPtr<slang::IBlob> diagnostics;
    auto module = session->loadModuleFromSourceString(
        "nativeSbtAddressing",
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
        schema && schema->getHitGroupCount() == 3 && schema->getMissShaderCount() == 2 &&
        schema->getCallableShaderCount() == 2);

    // Source schema reflection supplies the stage types. Ordinary selected-entry reflection
    // below supplies native exports; the host never guesses generated name suffixes.
    List<slang::RayTracingStageReflection*> stages;
    for (SlangUInt i = 0; i < schema->getHitGroupCount(); ++i)
    {
        auto group = schema->getHitGroup(i);
        SLANG_CHECK_ABORT(group->getFunctionIndex() == i);
        SLANG_CHECK_ABORT(!group->getAnyHit() && !group->getIntersection());
        stages.add(group->getClosestHit());
    }
    for (SlangUInt i = 0; i < schema->getMissShaderCount(); ++i)
        stages.add(schema->getMissShader(i)->getMiss());
    for (SlangUInt i = 0; i < schema->getCallableShaderCount(); ++i)
        stages.add(schema->getCallableShader(i)->getCallable());

    List<ComPtr<slang::IEntryPoint>> entries;
    entries.setCount(stages.getCount() + 1);
    auto result = module->findAndCheckEntryPoint(
        "RayGeneration",
        SLANG_STAGE_RAY_GENERATION,
        entries[0].writeRef(),
        diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    GFX_CHECK_CALL_ABORT(result);
    for (Index i = 0; i < stages.getCount(); ++i)
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
    List<slang::IComponentType*> components;
    components.add(module);
    for (auto& entry : entries)
        components.add(entry);
    ComPtr<slang::IComponentType> composite;
    result = session->createCompositeComponentType(
        components.getBuffer(),
        components.getCount(),
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
    SLANG_CHECK_ABORT(layout && layout->getEntryPointCount() == entries.getCount());
    ShaderProgramDesc programDesc = {};
    programDesc.slangGlobalScope = linked;
    ComPtr<IShaderProgram> program;
    result = device->createShaderProgram(programDesc, program.writeRef(), diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    GFX_CHECK_CALL_ABORT(result);

    const char* names[8];
    for (Index i = 0; i < SLANG_COUNT_OF(names); ++i)
    {
        names[i] = layout->getEntryPointByIndex(i)->getNameOverride();
        SLANG_CHECK_ABORT(names[i]);
    }
    const char* groupNames[] = {"sentinelGroup", "firstGroup", "secondGroup"};
    HitGroupDesc hitGroups[3] = {};
    for (SlangUInt i = 0; i < schema->getHitGroupCount(); ++i)
    {
        const auto functionIndex = schema->getHitGroup(i)->getFunctionIndex();
        hitGroups[functionIndex].hitGroupName = groupNames[functionIndex];
        hitGroups[functionIndex].closestHitEntryPoint = names[1 + i];
    }
    RayTracingPipelineDesc pipelineDesc = {};
    pipelineDesc.program = program;
    pipelineDesc.hitGroups = hitGroups;
    pipelineDesc.hitGroupCount = SLANG_COUNT_OF(hitGroups);
    pipelineDesc.maxRayPayloadSize = sizeof(uint32_t) * 4;
    pipelineDesc.maxAttributeSizeInBytes = sizeof(float) * 2;
    pipelineDesc.maxRecursion = 1;
    ComPtr<IRayTracingPipeline> pipeline;
    GFX_CHECK_CALL_ABORT(device->createRayTracingPipeline(pipelineDesc, pipeline.writeRef()));

    // Five physical hit records refer to only three schema functions. Record positions 0,1,3
    // intentionally contain the sentinel, while the address expression selects records 2/4.
    const char* hitRecordNames[] =
        {groupNames[0], groupNames[0], groupNames[1], groupNames[0], groupNames[2]};
    const char* missRecordNames[] = {names[4], names[5]};
    const char* callableRecordNames[] = {names[6], names[7]};
    ShaderTableDesc tableDesc = {};
    tableDesc.program = program;
    tableDesc.rayGenShaderCount = 1;
    tableDesc.rayGenShaderEntryPointNames = &names[0];
    tableDesc.hitGroupCount = SLANG_COUNT_OF(hitRecordNames);
    tableDesc.hitGroupNames = hitRecordNames;
    tableDesc.missShaderCount = SLANG_COUNT_OF(missRecordNames);
    tableDesc.missShaderEntryPointNames = missRecordNames;
    tableDesc.callableShaderCount = SLANG_COUNT_OF(callableRecordNames);
    tableDesc.callableShaderEntryPointNames = callableRecordNames;
    ComPtr<IShaderTable> table;
    GFX_CHECK_CALL_ABORT(device->createShaderTable(tableDesc, table.writeRef()));

    auto queue = device->getQueue(QueueType::Graphics);
    const float vertices[2][3][3] = {
        {{0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 1.0f}},
        {{2.0f, 0.0f, 1.0f}, {3.0f, 0.0f, 1.0f}, {2.0f, 1.0f, 1.0f}}};
    ComPtr<IBuffer> vertexBuffers[2];
    AccelerationStructureBuildInput triangles[2] = {};
    for (Index i = 0; i < SLANG_COUNT_OF(triangles); ++i)
    {
        BufferDesc vertexDesc = {};
        vertexDesc.size = sizeof(vertices[i]);
        vertexDesc.usage = BufferUsage::AccelerationStructureBuildInput;
        vertexDesc.defaultState = ResourceState::AccelerationStructureBuildInput;
        GFX_CHECK_CALL_ABORT(
            device->createBuffer(vertexDesc, vertices[i], vertexBuffers[i].writeRef()));
        triangles[i].type = AccelerationStructureBuildInputType::Triangles;
        triangles[i].triangles.vertexBuffers[0] = vertexBuffers[i];
        triangles[i].triangles.vertexBufferCount = 1;
        triangles[i].triangles.vertexFormat = Format::RGB32Float;
        triangles[i].triangles.vertexCount = 3;
        triangles[i].triangles.vertexStride = sizeof(vertices[i][0]);
        triangles[i].triangles.flags = AccelerationStructureGeometryFlags::Opaque;
    }
    auto blas = buildPortableAddressingAccelerationStructure(
        device,
        queue,
        triangles,
        SLANG_COUNT_OF(triangles),
        AccelerationStructureKind::BottomLevel);
    AccelerationStructureInstanceDescGeneric instance = {};
    instance.transform[0][0] = 1.0f;
    instance.transform[1][1] = 1.0f;
    instance.transform[2][2] = 1.0f;
    instance.instanceMask = 0xff;
    instance.instanceContributionToHitGroupIndex = 1;
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
    auto tlas = buildPortableAddressingAccelerationStructure(
        device,
        queue,
        &instanceInput,
        1,
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
    pass->dispatchRays(0, 3, 1, 1);
    pass->end();
    GFX_CHECK_CALL_ABORT(queue->submit(encoder->finish()));
    GFX_CHECK_CALL_ABORT(queue->waitOnHost());
    GFX_CHECK_CALL_ABORT(device->readBuffer(output, 0, sizeof(values), values));

    const uint32_t expected[3][4] = {{11, 0, 82, 0}, {22, 1, 82, 1}, {33, 0xffffffff, 82, 2}};
    for (Index ray = 0; ray < 3; ++ray)
        for (Index component = 0; component < 4; ++component)
            SLANG_CHECK(values[ray][component] == expected[ray][component]);
}

SLANG_UNIT_TEST(portableRayTracingSbtAddressingVulkan)
{
    runTestImpl(portableRayTracingSbtAddressingTestImpl, unitTestContext, DeviceType::Vulkan);
}

SLANG_UNIT_TEST(portableRayTracingSbtAddressingD3D12)
{
    runTestImpl(portableRayTracingSbtAddressingTestImpl, unitTestContext, DeviceType::D3D12);
}

} // namespace gfx_test
