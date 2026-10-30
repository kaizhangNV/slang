#include "core/slang-string.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

// Exercise the public host path with two specializations nested under arrays and a parameter
// block. Their source field keys and binding footprints coincide, but emitted table ownership
// must match each checked Schema type's RTTI name, not a specialized IR symbol or debug name.
SLANG_UNIT_TEST(metalPortableRayTracingArtifactMetadata)
{
    const char* source = R"(
        import slang.raytracing.portable;
        struct TraceContext : rt::ITraceContext
        {
            typealias AccelerationStructure = rt::AccelerationStructure;
            typealias Motion = rt::NoMotion;
        }
        struct InnerRecord { uint64_t wide; uint scalar; }
        struct MaterialRecord
        {
            float3 color;
            float weight;
            bool flags[17];
            InnerRecord inner;
            row_major float2x3 rows;
            column_major float2x3 columns;
            float tail;
        }
        struct MissContext : rt::IPayloadContext
        {
            typealias TraceContext = ::TraceContext;
            typealias Payload = uint;
            typealias Record = MaterialRecord;
        }
        StructuredBuffer<uint> background;
        [noinline] uint readBackground() { return background[0]; }
        struct Miss<let N : int> : rt::IMissShader
        {
            typealias Context = MissContext;
            static void invoke(rt::MissInput<Context> input)
            {
                input.payload = uint(N) + readBackground() + uint(input.record.color.x)
                    + uint(input.record.weight) + uint(input.record.flags[16])
                    + uint(input.record.inner.wide) + input.record.inner.scalar
                    + uint(input.record.rows[1][2]) + uint(input.record.columns[1][2])
                    + uint(input.record.tail);
            }
        }
        struct ConstantMiss : rt::IMissShader
        {
            typealias Context = MissContext;
            static void invoke(rt::MissInput<Context> input) { input.payload = 11; }
        }
        struct Schema<let N : int> : rt::ITraceProgramSchema
        {
            typealias TraceContext = ::TraceContext;
            typealias HitGroups = rt::EmptyHitGroupList;
            typealias MissShaders = rt::MissShaderList<Miss<N>, ConstantMiss>;
            typealias CallableShaders = rt::EmptyCallableShaderList;
        }
        struct Holder<let N : int> { rt::TraceProgramDescriptor<Schema<N>> descriptor; }
        struct Resources { Holder<1> first[2]; Holder<2> second; }
        ParameterBlock<Resources> resources;
        rt::AccelerationStructure scene;
        RWStructuredBuffer<uint> output;
        uint selected;
        [shader("raygeneration")]
        void main()
        {
            rt::RayTraversalDesc ray = {};
            ray.ray.Direction = float3(0, 0, 1);
            ray.ray.TMax = 100;
            ray.instanceMask = 255;
            ray.missIndex = selected & 1;
            uint first = 0;
            uint second = 0;
            rt::RayTracer<Schema<1>> firstTracer;
            rt::RayTracer<Schema<2>> secondTracer;
            firstTracer.trace(ray, scene, resources.first[selected & 1].descriptor, first);
            secondTracer.trace(ray, scene, resources.second.descriptor, second);
            output[0] = first + second;
        }
    )";

    slang::TargetDesc target = {};
    target.format = SLANG_METAL;
    target.profile = unitTestContext->slangGlobalSession->findProfile("metallib_2_4");
    slang::CompilerOptionEntry experimental = {};
    experimental.name = slang::CompilerOptionName::ExperimentalFeature;
    experimental.value.kind = slang::CompilerOptionValueKind::Int;
    experimental.value.intValue0 = 1;
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &target;
    sessionDesc.compilerOptionEntryCount = 1;
    sessionDesc.compilerOptionEntries = &experimental;
    ComPtr<slang::ISession> session;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
        unitTestContext->slangGlobalSession->createSession(sessionDesc, session.writeRef())));
    ComPtr<slang::IBlob> diagnostics;
    auto module = session->loadModuleFromSourceString(
        "metalPhysicalMetadata",
        "metalPhysicalMetadata.slang",
        source,
        diagnostics.writeRef());
    if (!module && diagnostics)
        getTestReporter()->message(
            TestMessageType::Info,
            static_cast<const char*>(diagnostics->getBufferPointer()));
    SLANG_CHECK_ABORT(module);
    ComPtr<slang::IEntryPoint> entry;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->findEntryPointByName("main", entry.writeRef())));
    ComPtr<slang::IEntryPoint> selectedMiss;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->findAndCheckEntryPoint(
        "ConstantMiss",
        SLANG_STAGE_MISS,
        selectedMiss.writeRef(),
        diagnostics.writeRef())));
    slang::IComponentType* components[] = {module, entry, selectedMiss};
    ComPtr<slang::IComponentType> composite;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(session->createCompositeComponentType(
        components,
        SLANG_COUNT_OF(components),
        composite.writeRef(),
        diagnostics.writeRef())));
    ComPtr<slang::IComponentType> linked;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(composite->link(linked.writeRef(), diagnostics.writeRef())));
    auto sourceLayout = linked->getLayout(0, diagnostics.writeRef());
    SLANG_CHECK_ABORT(sourceLayout);
    auto firstSchema = sourceLayout->findTraceProgramSchema("Schema<1>");
    auto secondSchema = sourceLayout->findTraceProgramSchema("Schema<2>");
    SLANG_CHECK_ABORT(firstSchema && secondSchema && firstSchema != secondSchema);
    auto recordLayout = firstSchema->getMissShader(0)->getRecordTypeLayout();
    SLANG_CHECK_ABORT(recordLayout && recordLayout->getFieldCount() == 7);
    // Packed float3 leaves the next scalar at byte 12. Metal bool occupies one byte, and the
    // nested uint64/uint aggregate includes its C tail padding. Natural IR layout is not this ABI.
    SLANG_CHECK(recordLayout->getFieldByIndex(0)->getOffset() == 0);
    SLANG_CHECK(recordLayout->getFieldByIndex(1)->getOffset() == 12);
    SLANG_CHECK(recordLayout->getFieldByIndex(2)->getOffset() == 16);
    SLANG_CHECK(recordLayout->getFieldByIndex(3)->getOffset() == 40);
    // Both major orders occupy 24 bytes, represented as two packed float3 rows or three packed
    // float2 columns. The scalar following them must agree with reflection as well.
    SLANG_CHECK(recordLayout->getFieldByIndex(4)->getOffset() == 56);
    SLANG_CHECK(recordLayout->getFieldByIndex(5)->getOffset() == 80);
    SLANG_CHECK(recordLayout->getFieldByIndex(6)->getOffset() == 104);
    SLANG_CHECK(recordLayout->getSize() == 112);
    ComPtr<slang::IBlob> schemaNames[2];
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
        session->getTypeRTTIMangledName(firstSchema->getType(), schemaNames[0].writeRef())));
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
        session->getTypeRTTIMangledName(secondSchema->getType(), schemaNames[1].writeRef())));

    ComPtr<slang::IBlob> code;
    auto result = linked->getEntryPointCode(0, 0, code.writeRef(), diagnostics.writeRef());
    if (SLANG_FAILED(result) && diagnostics)
        getTestReporter()->message(
            TestMessageType::Info,
            static_cast<const char*>(diagnostics->getBufferPointer()));
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(result));
    ComPtr<slang::IMetadata> metadata;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
        linked->getEntryPointMetadata(0, 0, metadata.writeRef(), diagnostics.writeRef())));
    auto physical = static_cast<slang::IMetalRayTracingMetadata*>(
        metadata->castAs(slang::IMetalRayTracingMetadata::getTypeGuid()));
    SLANG_CHECK_ABORT(physical && physical->getProgramCount() == 2);
    auto emitted = UnownedStringSlice(
        static_cast<const char*>(code->getBufferPointer()),
        code->getBufferSize());
    // This artifact contains ray generation only. The schema still describes both Miss slots
    // and their record stride; neither provider implementation is emitted here.
    SLANG_CHECK(emitted.indexOf(UnownedStringSlice("readBackground")) < 0);
    uint32_t programIndices[2] = {};
    for (UInt schemaIndex = 0; schemaIndex < 2; ++schemaIndex)
    {
        auto name = static_cast<const char*>(schemaNames[schemaIndex]->getBufferPointer());
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
            physical->findProgramBySchemaTypeMangledName(name, &programIndices[schemaIndex])));
        slang::MetalRayTracingProgramInfo program;
        SLANG_CHECK_ABORT(
            SLANG_SUCCEEDED(physical->getProgramInfo(programIndices[schemaIndex], &program)));
        SLANG_CHECK(UnownedStringSlice(program.schemaTypeMangledName) == UnownedStringSlice(name));
        SLANG_CHECK(program.resourceCount == 7 && program.functionCount == 0);
        SLANG_CHECK(program.candidateDescriptorBufferIndex == UINT32_MAX);
        SLANG_CHECK(program.tagMask == uint32_t(slang::MetalRayTracingTag::Instancing));
        SLANG_CHECK(program.recordHeaderSize == 16 && program.recordAlignment == 16);
        SLANG_CHECK(program.missRecordStride == 128);
        SLANG_CHECK(
            program.missRecordStride ==
            ((program.recordHeaderSize + recordLayout->getSize() + 15) / 16) * 16);
        for (uint32_t i = 0; i < program.resourceCount; ++i)
        {
            slang::MetalRayTracingResourceInfo resource;
            SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
                physical->getProgramResourceInfo(programIndices[schemaIndex], i, &resource)));
            SLANG_CHECK(resource.argumentBufferID == i);
            SLANG_CHECK(
                resource.functionTableSize ==
                (resource.kind == slang::MetalRayTracingResourceKind::MissVisibleFunctionTable
                     ? 2u
                     : 0u));
        }
    }
    SLANG_CHECK(programIndices[0] != programIndices[1]);
    SLANG_CHECK(sourceLayout->findTraceProgramSchema("Schema<1>") == firstSchema);
    SLANG_CHECK(sourceLayout->findTraceProgramSchema("Schema<2>") == secondSchema);
    uint32_t missing = 123;
    SLANG_CHECK(
        physical->findProgramBySchemaTypeMangledName("absent", &missing) == SLANG_E_NOT_FOUND);
    SLANG_CHECK(missing == 123);
    slang::MetalRayTracingProgramInfo invalid;
    invalid.structSize = 0;
    SLANG_CHECK(physical->getProgramInfo(0, &invalid) == SLANG_E_INVALID_ARG);
    SLANG_CHECK(physical->getProgramInfo(2, nullptr) == SLANG_E_INVALID_ARG);

    // The selected Miss has no trace/call operation and does not read a global. It nevertheless
    // shares this component's full resource layout with ray generation, including both nested
    // descriptor specializations. Their physical fields and metadata must therefore exist here
    // without retaining the ray-generation body or either unselected Miss implementation.
    ComPtr<slang::IBlob> providerCode;
    result = linked->getEntryPointCode(1, 0, providerCode.writeRef(), diagnostics.writeRef());
    if (SLANG_FAILED(result) && diagnostics)
        getTestReporter()->message(
            TestMessageType::Info,
            static_cast<const char*>(diagnostics->getBufferPointer()));
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(result));
    auto providerText = UnownedStringSlice(
        static_cast<const char*>(providerCode->getBufferPointer()),
        providerCode->getBufferSize());
    SLANG_CHECK(providerText.indexOf(UnownedStringSlice("[[visible]]")) >= 0);
    SLANG_CHECK(providerText.indexOf(UnownedStringSlice("PortableRTDescriptorResources")) >= 0);
    SLANG_CHECK(providerText.indexOf(UnownedStringSlice("readBackground")) < 0);
    SLANG_CHECK(providerText.indexOf(UnownedStringSlice("[[kernel]]")) < 0);
    ComPtr<slang::IMetadata> providerMetadata;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
        linked->getEntryPointMetadata(1, 0, providerMetadata.writeRef(), diagnostics.writeRef())));
    auto providerPhysical = static_cast<slang::IMetalRayTracingMetadata*>(
        providerMetadata->castAs(slang::IMetalRayTracingMetadata::getTypeGuid()));
    SLANG_CHECK_ABORT(providerPhysical && providerPhysical->getProgramCount() == 2);
    for (UInt schemaIndex = 0; schemaIndex < 2; ++schemaIndex)
    {
        uint32_t programIndex;
        auto name = static_cast<const char*>(schemaNames[schemaIndex]->getBufferPointer());
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
            providerPhysical->findProgramBySchemaTypeMangledName(name, &programIndex)));
        slang::MetalRayTracingProgramInfo program;
        SLANG_CHECK_ABORT(
            SLANG_SUCCEEDED(providerPhysical->getProgramInfo(programIndex, &program)));
        SLANG_CHECK(program.resourceCount == 7);
        SLANG_CHECK(program.candidateDescriptorBufferIndex == UINT32_MAX);
        SLANG_CHECK(program.missRecordStride == 128);
    }
}
