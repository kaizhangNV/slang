#include "core/slang-string.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

// A host constructs local root signatures from reflection before requesting shader code. The
// compiler-generated record must therefore have an ordinary cbuffer binding at that point, plus
// the ShaderRecord role that distinguishes it from globally bound ConstantBuffer<Material>.
SLANG_UNIT_TEST(portableRayTracingD3DRecordBindingAndReflection)
{
    const char* source = R"(
        import slang.raytracing.portable;

        struct Material
        {
            float3 color;
            float opacity;
            float2 scale;
        }
        struct EmptyRecord {}
        struct ResourceRecord
        {
            StructuredBuffer<float3> values;
        }
        struct MixedRecord
        {
            StructuredBuffer<float3> values;
            float factor;
        }
        ConstantBuffer<Material> globalMaterial : register(b3, space0);
        ConstantBuffer<uint> reservedSpace : register(b5, space1);

        struct TraceContext : rt::ITraceContext
        {
            typealias AccelerationStructure = rt::AccelerationStructure;
            typealias Motion = rt::NoMotion;
        }
        struct RecordContext<RecordType> : rt::IPayloadContext
        {
            typealias TraceContext = ::TraceContext;
            typealias Payload = float3;
            typealias Record = RecordType;
        }
        struct MaterialMiss : rt::IMissShader
        {
            typealias Context = RecordContext<Material>;
            static void invoke(rt::MissInput<Context> input)
            {
                input.payload = input.record.color * input.record.opacity;
            }
        }
        struct OtherMaterialMiss : rt::IMissShader
        {
            typealias Context = RecordContext<Material>;
            static void invoke(rt::MissInput<Context> input)
            {
                // This call is an ordinary helper call, not dispatch to MaterialMiss's SBT
                // record. It must receive this invocation's record through the forwarded view.
                MaterialMiss::invoke(input);
                input.payload *= input.record.scale.x;
            }
        }
        struct NoRecordMiss : rt::IMissShader
        {
            typealias Context = RecordContext<void>;
            static void invoke(rt::MissInput<Context> input)
            {
                input.payload = float3(0);
            }
        }
        [noinline]
        float consumeEmptyRecord(EmptyRecord record)
        {
            return 23;
        }
        struct EmptyRecordMiss : rt::IMissShader
        {
            typealias Context = RecordContext<EmptyRecord>;
            static void invoke(rt::MissInput<Context> input)
            {
                input.payload = float3(consumeEmptyRecord(input.record));
            }
        }
        struct ResourceRecordMiss : rt::IMissShader
        {
            typealias Context = RecordContext<ResourceRecord>;
            static void invoke(rt::MissInput<Context> input)
            {
                input.payload = input.record.values[0];
            }
        }
        struct MixedRecordMiss : rt::IMissShader
        {
            typealias Context = RecordContext<MixedRecord>;
            static void invoke(rt::MissInput<Context> input)
            {
                input.payload = input.record.values[0] * input.record.factor;
            }
        }
    )";

    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(slang_createGlobalSession(0, globalSession.writeRef())));
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_HLSL;
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
    SLANG_CHECK_ABORT(
        SLANG_SUCCEEDED(globalSession->createSession(sessionDesc, session.writeRef())));

    ComPtr<slang::IBlob> diagnostics;
    auto module = session->loadModuleFromSourceString(
        "recordLayout",
        "recordLayout.slang",
        source,
        diagnostics.writeRef());
    SLANG_CHECK_ABORT(module != nullptr);
    const char* entryNames[] = {
        "MaterialMiss",
        "OtherMaterialMiss",
        "NoRecordMiss",
        "EmptyRecordMiss",
        "ResourceRecordMiss",
        "MixedRecordMiss"};
    ComPtr<slang::IEntryPoint> entries[6];
    for (int i = 0; i < 6; ++i)
    {
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->findAndCheckEntryPoint(
            entryNames[i],
            SLANG_STAGE_MISS,
            entries[i].writeRef(),
            diagnostics.writeRef())));
    }
    slang::IComponentType* components[] =
        {module, entries[0], entries[1], entries[2], entries[3], entries[4], entries[5]};
    ComPtr<slang::IComponentType> composite;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(session->createCompositeComponentType(
        components,
        7,
        composite.writeRef(),
        diagnostics.writeRef())));
    ComPtr<slang::IComponentType> linked;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(composite->link(linked.writeRef(), diagnostics.writeRef())));

    // No target code has been requested. This is the same public layout API used by a host.
    auto layout = linked->getLayout(0, diagnostics.writeRef());
    SLANG_CHECK_ABORT(layout != nullptr);
    SLANG_CHECK_ABORT(layout->getParameterCount() == 2);  // globalMaterial and reservedSpace.
    SLANG_CHECK_ABORT(layout->getEntryPointCount() == 6); // The six selected miss shaders.
    const auto cbuffer = slang::ParameterCategory::ConstantBuffer;
    const auto shaderRecord = slang::ParameterCategory::ShaderRecord;
    const auto uniform = slang::ParameterCategory::Uniform;
    // Get: ConstantBuffer<Material> globalMaterial : register(b3, space0);
    auto global = layout->getParameterByIndex(0);
    auto globalType = global->getTypeLayout();
    SLANG_CHECK(global->getOffset(cbuffer) == 3);        // Register b3.
    SLANG_CHECK(global->getBindingSpace(cbuffer) == 0);  // Space 0.
    SLANG_CHECK(globalType->getSize(shaderRecord) == 0); // No shader-record role on this global.
    // Get Material's layout inside ConstantBuffer<Material>.
    auto materialLayout = globalType->getElementTypeLayout();
    SLANG_CHECK_ABORT(materialLayout != nullptr);

    // MaterialMiss and OtherMaterialMiss each need their own Material record.
    slang::VariableLayoutReflection* recordBindings[2];
    for (int i = 0; i < 2; ++i)
    {
        auto entryLayout = layout->getEntryPointByIndex(i);
        SLANG_CHECK_ABORT(entryLayout->getParameterCount() == 1); // One generated record parameter.
        auto record = entryLayout->getParameterByIndex(0);
        recordBindings[i] = record;
        SLANG_CHECK(record->getVariable() == nullptr); // Generated record has no source variable.
        SLANG_CHECK(record->getName() == nullptr);     // The generated parameter is anonymous.
        SLANG_CHECK(record->getBindingSpace(cbuffer) != 0); // Space 0 is used by globalMaterial.
        SLANG_CHECK(record->getBindingSpace(cbuffer) != 1); // Space 1 is used by reservedSpace.
        auto recordType = record->getTypeLayout();
        // D3D stores Material's data in a constant buffer.
        SLANG_CHECK(recordType->getKind() == slang::TypeReflection::Kind::ConstantBuffer);
        SLANG_CHECK(recordType->getSize(cbuffer) == 1);       // One b register.
        SLANG_CHECK(recordType->getSize(shaderRecord) == 1);  // One shader-record role.
        SLANG_CHECK(recordType->getBindingRangeCount() == 1); // One resource binding range.
        // That range describes the constant buffer.
        SLANG_CHECK(recordType->getBindingRangeType(0) == slang::BindingType::ConstantBuffer);
        // The buffer container also carries the shader-record role.
        SLANG_CHECK(
            recordType->getContainerVarLayout()->getTypeLayout()->getSize(shaderRecord) == 1);
        auto element = recordType->getElementTypeLayout();
        SLANG_CHECK_ABORT(element != nullptr);
        // Material has the same byte size, fields, and field offsets as in globalMaterial.
        SLANG_CHECK(element->getSize(uniform) == materialLayout->getSize(uniform));
        SLANG_CHECK_ABORT(element->getFieldCount() == materialLayout->getFieldCount());
        for (unsigned fieldIndex = 0; fieldIndex < element->getFieldCount(); ++fieldIndex)
        {
            SLANG_CHECK(
                element->getFieldByIndex(fieldIndex)->getOffset(uniform) ==
                materialLayout->getFieldByIndex(fieldIndex)->getOffset(uniform));
        }
    }
    // Both records share the generated space, with a different b register for each entry.
    SLANG_CHECK(
        recordBindings[0]->getBindingSpace(cbuffer) == recordBindings[1]->getBindingSpace(cbuffer));
    SLANG_CHECK(recordBindings[0]->getOffset(cbuffer) != recordBindings[1]->getOffset(cbuffer));
    // NoRecordMiss uses void; EmptyRecordMiss uses an empty struct. Neither needs a record binding.
    SLANG_CHECK(layout->getEntryPointByIndex(2)->getParameterCount() == 0);
    SLANG_CHECK(layout->getEntryPointByIndex(3)->getParameterCount() == 0);
    const auto shaderResource = slang::ParameterCategory::ShaderResource;
    // ResourceRecordMiss and MixedRecordMiss both have a StructuredBuffer named values.
    slang::VariableLayoutReflection* resourceBindings[2];
    for (int i = 0; i < 2; ++i)
    {
        auto entry = layout->getEntryPointByIndex(4 + i);
        SLANG_CHECK_ABORT(entry->getParameterCount() == 1); // One generated record parameter.
        auto record = entry->getParameterByIndex(0);
        resourceBindings[i] = record;
        auto type = record->getTypeLayout();
        SLANG_CHECK(type->getSize(shaderRecord) == 1);   // One shader-record role.
        SLANG_CHECK(type->getSize(shaderResource) == 1); // One t register for values.
        // Only MixedRecord needs a constant buffer, for its factor field.
        SLANG_CHECK(type->getSize(cbuffer) == size_t(i));
        // Resource fields use the same generated space as the Material record buffers.
        SLANG_CHECK(
            record->getBindingSpace(shaderResource) == recordBindings[0]->getBindingSpace(cbuffer));
    }
    // Each entry's values field gets a different t register.
    SLANG_CHECK(
        resourceBindings[0]->getOffset(shaderResource) !=
        resourceBindings[1]->getOffset(shaderResource));

    // Adapter generation must reuse those exact allocations. A second independent allocation
    // could produce valid HLSL while disagreeing with the host's already-created root signature.
    for (int i = 0; i < 2; ++i)
    {
        ComPtr<slang::IBlob> code;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
            linked->getEntryPointCode(i, 0, code.writeRef(), diagnostics.writeRef())));
        SLANG_CHECK_ABORT(code != nullptr);
        String codeText(
            UnownedStringSlice((const char*)code->getBufferPointer(), code->getBufferSize()));
        StringBuilder expectedBinding;
        expectedBinding << "register(b" << UInt64(recordBindings[i]->getOffset(cbuffer))
                        << ", space" << UInt64(recordBindings[i]->getBindingSpace(cbuffer)) << ")";
        // Emitted HLSL uses this entry's reflected b register and space.
        SLANG_CHECK(codeText.indexOf(expectedBinding.getUnownedSlice()) >= 0);
        SLANG_CHECK(codeText.indexOf("cbuffer") >= 0); // The record buffer is emitted.

        // Calling another selected stage as a static helper must not read that stage's own
        // SBT record. In particular, OtherMaterialMiss must use its binding for both calls.
        auto otherRecord = recordBindings[1 - i];
        StringBuilder unexpectedBinding;
        unexpectedBinding << "register(b" << UInt64(otherRecord->getOffset(cbuffer)) << ", space"
                          << UInt64(otherRecord->getBindingSpace(cbuffer)) << ")";
        SLANG_CHECK(codeText.indexOf(unexpectedBinding.getUnownedSlice()) < 0);
    }
    // EmptyRecordMiss must still compile when it passes its empty record to a helper.
    ComPtr<slang::IBlob> emptyRecordCode;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
        linked->getEntryPointCode(3, 0, emptyRecordCode.writeRef(), diagnostics.writeRef())));
    SLANG_CHECK_ABORT(emptyRecordCode != nullptr);
    for (int i = 0; i < 2; ++i)
    {
        ComPtr<slang::IBlob> code;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
            linked->getEntryPointCode(4 + i, 0, code.writeRef(), diagnostics.writeRef())));
        SLANG_CHECK_ABORT(code != nullptr);
        String text(
            UnownedStringSlice((const char*)code->getBufferPointer(), code->getBufferSize()));
        StringBuilder expectedBinding;
        expectedBinding << "register(t" << UInt64(resourceBindings[i]->getOffset(shaderResource))
                        << ", space" << UInt64(resourceBindings[i]->getBindingSpace(shaderResource))
                        << ")";
        // Emitted HLSL uses this entry's reflected t register and space for values.
        SLANG_CHECK(text.indexOf(expectedBinding.getUnownedSlice()) >= 0);
    }
}

// A Vulkan host also needs the record's byte layout before code generation, but its SBT bytes
// are read directly through ShaderRecordBufferKHR and must not consume a descriptor binding.
SLANG_UNIT_TEST(portableRayTracingSPIRVRecordBindingAndReflection)
{
    const char* source = R"(
        import slang.raytracing.portable;

        struct Material
        {
            float weights[2];
            float2 scale;
            float3 color;
            float opacity;
        }
        struct EmptyRecord {}
        ConstantBuffer<Material> globalMaterial;
        [[vk::shader_record]] ConstantBuffer<Material> nativeRecord;

        struct TraceContext : rt::ITraceContext
        {
            typealias AccelerationStructure = rt::AccelerationStructure;
            typealias Motion = rt::NoMotion;
        }
        struct RecordContext<RecordType> : rt::IPayloadContext
        {
            typealias TraceContext = ::TraceContext;
            typealias Payload = float3;
            typealias Record = RecordType;
        }
        struct MaterialMiss : rt::IMissShader
        {
            typealias Context = RecordContext<Material>;
            static void invoke(rt::MissInput<Context> input)
            {
                float2 weights = float2(input.record.weights[0], input.record.weights[1]);
                input.payload = input.record.color * input.record.opacity
                    * dot(input.record.scale, weights);
            }
        }
        struct OtherMaterialMiss : rt::IMissShader
        {
            typealias Context = RecordContext<Material>;
            static void invoke(rt::MissInput<Context> input)
            {
                MaterialMiss::invoke(input);
            }
        }
        struct NoRecordMiss : rt::IMissShader
        {
            typealias Context = RecordContext<void>;
            static void invoke(rt::MissInput<Context> input)
            {
                input.payload = float3(0);
            }
        }
        [noinline]
        float consumeEmptyRecord(EmptyRecord record)
        {
            return 23;
        }
        struct EmptyRecordMiss : rt::IMissShader
        {
            typealias Context = RecordContext<EmptyRecord>;
            static void invoke(rt::MissInput<Context> input)
            {
                input.payload = float3(consumeEmptyRecord(input.record));
            }
        }
    )";

    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(slang_createGlobalSession(0, globalSession.writeRef())));
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV_ASM;
    targetDesc.profile = globalSession->findProfile("spirv_1_5");
    slang::CompilerOptionEntry options[2] = {};
    options[0].name = slang::CompilerOptionName::ExperimentalFeature;
    options[0].value.kind = slang::CompilerOptionValueKind::Int;
    options[0].value.intValue0 = 1;
    options[1].name = slang::CompilerOptionName::EmitSpirvDirectly;
    options[1].value.kind = slang::CompilerOptionValueKind::Int;
    options[1].value.intValue0 = 1;
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;
    sessionDesc.compilerOptionEntryCount = 2;
    sessionDesc.compilerOptionEntries = options;
    ComPtr<slang::ISession> session;
    SLANG_CHECK_ABORT(
        SLANG_SUCCEEDED(globalSession->createSession(sessionDesc, session.writeRef())));

    ComPtr<slang::IBlob> diagnostics;
    auto module = session->loadModuleFromSourceString(
        "spirvRecordLayout",
        "spirvRecordLayout.slang",
        source,
        diagnostics.writeRef());
    SLANG_CHECK_ABORT(module != nullptr);
    const char* entryNames[] =
        {"MaterialMiss", "OtherMaterialMiss", "NoRecordMiss", "EmptyRecordMiss"};
    ComPtr<slang::IEntryPoint> entries[4];
    for (int i = 0; i < 4; ++i)
    {
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->findAndCheckEntryPoint(
            entryNames[i],
            SLANG_STAGE_MISS,
            entries[i].writeRef(),
            diagnostics.writeRef())));
    }
    slang::IComponentType* components[] = {module, entries[0], entries[1], entries[2], entries[3]};
    ComPtr<slang::IComponentType> composite;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(session->createCompositeComponentType(
        components,
        5,
        composite.writeRef(),
        diagnostics.writeRef())));
    ComPtr<slang::IComponentType> linked;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(composite->link(linked.writeRef(), diagnostics.writeRef())));

    // Reflection describes the future record resource, not the source MissInput parameter.
    // No adapter or target code has been requested when the host obtains these layouts.
    auto layout = linked->getLayout(0, diagnostics.writeRef());
    SLANG_CHECK_ABORT(layout != nullptr);
    SLANG_CHECK_ABORT(layout->getParameterCount() == 2);  // globalMaterial and nativeRecord.
    SLANG_CHECK_ABORT(layout->getEntryPointCount() == 4); // The four selected miss shaders.
    const auto shaderRecord = slang::ParameterCategory::ShaderRecord;
    const auto descriptor = slang::ParameterCategory::DescriptorTableSlot;
    const auto uniform = slang::ParameterCategory::Uniform;
    // Get: ConstantBuffer<Material> globalMaterial;
    auto globalType = layout->getParameterByIndex(0)->getTypeLayout();
    SLANG_CHECK(globalType->getSize(shaderRecord) == 0); // No shader-record role on this global.
    SLANG_CHECK(globalType->getSize(descriptor) == 1);   // One descriptor for the ordinary buffer.
    // Get: [[vk::shader_record]] ConstantBuffer<Material> nativeRecord;
    auto nativeRecordType = layout->getParameterByIndex(1)->getTypeLayout();
    SLANG_CHECK(nativeRecordType->getSize(shaderRecord) == 1); // One shader-record role.
    SLANG_CHECK(nativeRecordType->getSize(descriptor) == 0);   // SBT data needs no descriptor.
    // Get Material's layout inside nativeRecord.
    auto nativeRecordElement = nativeRecordType->getElementTypeLayout();
    SLANG_CHECK_ABORT(nativeRecordElement != nullptr);
    // Material has weights, scale, color, and opacity.
    SLANG_CHECK_ABORT(nativeRecordElement->getFieldCount() == 4);

    // MaterialMiss and OtherMaterialMiss each expose a generated Material record.
    slang::TypeLayoutReflection* recordElements[2];
    for (int i = 0; i < 2; ++i)
    {
        auto entry = layout->getEntryPointByIndex(i);
        SLANG_CHECK_ABORT(entry->getParameterCount() == 1); // One generated record parameter.
        auto record = entry->getParameterByIndex(0);
        SLANG_CHECK(record->getVariable() == nullptr); // Generated record has no source variable.
        SLANG_CHECK(record->getName() == nullptr);     // The generated parameter is anonymous.
        auto type = record->getTypeLayout();
        // The record is reflected as ConstantBuffer<Material>.
        SLANG_CHECK(type->getKind() == slang::TypeReflection::Kind::ConstantBuffer);
        SLANG_CHECK(type->getSize(shaderRecord) == 1); // One shader-record role.
        SLANG_CHECK(type->getSize(descriptor) == 0);   // SBT data needs no descriptor.
        // No constant-buffer register is allocated.
        SLANG_CHECK(type->getSize(slang::ParameterCategory::ConstantBuffer) == 0);
        // The buffer container also carries the shader-record role.
        SLANG_CHECK(type->getContainerVarLayout()->getTypeLayout()->getSize(shaderRecord) == 1);
        auto element = type->getElementTypeLayout();
        SLANG_CHECK_ABORT(element != nullptr);
        recordElements[i] = element;

        // Material has the same byte size as in nativeRecord.
        SLANG_CHECK(element->getSize(uniform) == nativeRecordElement->getSize(uniform));
        SLANG_CHECK_ABORT(element->getFieldCount() == 4); // weights, scale, color, and opacity.
        // Each Material field has the same byte offset as in nativeRecord.
        for (unsigned field = 0; field < 4; ++field)
        {
            SLANG_CHECK(
                element->getFieldByIndex(field)->getOffset(uniform) ==
                nativeRecordElement->getFieldByIndex(field)->getOffset(uniform));
        }
        // The byte stride between weights[0] and weights[1] also matches nativeRecord.
        SLANG_CHECK(
            element->getFieldByIndex(0)->getTypeLayout()->getElementStride(
                SLANG_PARAMETER_CATEGORY_UNIFORM) ==
            nativeRecordElement->getFieldByIndex(0)->getTypeLayout()->getElementStride(
                SLANG_PARAMETER_CATEGORY_UNIFORM));
    }
    // NoRecordMiss uses void; EmptyRecordMiss uses an empty struct. Neither needs a record binding.
    SLANG_CHECK(layout->getEntryPointByIndex(2)->getParameterCount() == 0);
    SLANG_CHECK(layout->getEntryPointByIndex(3)->getParameterCount() == 0);

    for (int i = 0; i < 4; ++i)
    {
        ComPtr<slang::IBlob> code;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
            linked->getEntryPointCode(i, 0, code.writeRef(), diagnostics.writeRef())));
        SLANG_CHECK_ABORT(code != nullptr);
        String text(
            UnownedStringSlice((const char*)code->getBufferPointer(), code->getBufferSize()));
        // Unused globals are removed; SBT bytes need no descriptor.
        SLANG_CHECK(text.indexOf(" DescriptorSet ") < 0); // No descriptor set decoration.
        SLANG_CHECK(text.indexOf(" Binding ") < 0);       // No descriptor binding decoration.
        if (i < 2)
        {
            // MaterialMiss and OtherMaterialMiss read their Material data from the SBT.
            SLANG_CHECK(text.indexOf("ShaderRecordBufferKHR") >= 0);
            // Emitted Material field offsets match reflection.
            for (unsigned field = 0; field < 4; ++field)
            {
                StringBuilder decoration;
                decoration << "OpMemberDecorate %Material_std140 " << field << " Offset "
                           << UInt64(recordElements[i]->getFieldByIndex(field)->getOffset(uniform));
                SLANG_CHECK(text.indexOf(decoration.getUnownedSlice()) >= 0);
            }
            StringBuilder arrayStride;
            arrayStride
                << "ArrayStride "
                << UInt64(recordElements[i]->getFieldByIndex(0)->getTypeLayout()->getElementStride(
                       SLANG_PARAMETER_CATEGORY_UNIFORM));
            // Emitted weights array stride matches reflection.
            SLANG_CHECK(text.indexOf(arrayStride.getUnownedSlice()) >= 0);
        }
        else
        {
            // NoRecordMiss and EmptyRecordMiss need no shader-record storage.
            SLANG_CHECK(text.indexOf("ShaderRecordBufferKHR") < 0);
        }
    }
}

// Before specialization, the record has a generic resource requirement rather than a concrete
// byte layout. Entry-point totals must preserve that requirement without allocating a descriptor.
SLANG_UNIT_TEST(portableRayTracingSPIRVGenericRecordReflection)
{
    const char* source = R"(
        import slang.raytracing.portable;
        type_param T;

        struct TraceContext : rt::ITraceContext
        {
            typealias AccelerationStructure = rt::AccelerationStructure;
            typealias Motion = rt::NoMotion;
        }
        struct TestContext : rt::IPayloadContext
        {
            typealias TraceContext = ::TraceContext;
            typealias Payload = uint;
            typealias Record = T;
        }
        struct GenericRecordMiss : rt::IMissShader
        {
            typealias Context = TestContext;
            static void invoke(rt::MissInput<Context> input) {}
        }
    )";

    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(slang_createGlobalSession(0, globalSession.writeRef())));
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = globalSession->findProfile("spirv_1_5");
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
    SLANG_CHECK_ABORT(
        SLANG_SUCCEEDED(globalSession->createSession(sessionDesc, session.writeRef())));

    ComPtr<slang::IBlob> diagnostics;
    auto module = session->loadModuleFromSourceString(
        "genericRecordLayout",
        "genericRecordLayout.slang",
        source,
        diagnostics.writeRef());
    SLANG_CHECK_ABORT(module != nullptr);
    ComPtr<slang::IEntryPoint> entryPoint;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->findAndCheckEntryPoint(
        "GenericRecordMiss",
        SLANG_STAGE_MISS,
        entryPoint.writeRef(),
        diagnostics.writeRef())));
    slang::IComponentType* components[] = {module, entryPoint};
    ComPtr<slang::IComponentType> composite;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(session->createCompositeComponentType(
        components,
        2,
        composite.writeRef(),
        diagnostics.writeRef())));

    // T remains unresolved. Query the partial layout without linking or requesting target code.
    auto layout = composite->getLayout(0, diagnostics.writeRef());
    SLANG_CHECK_ABORT(layout != nullptr);
    SLANG_CHECK_ABORT(layout->getEntryPointCount() == 1); // GenericRecordMiss.
    auto entry = layout->getEntryPointByIndex(0);
    SLANG_CHECK_ABORT(entry->getParameterCount() == 1); // One generated record parameter.
    auto recordType = entry->getParameterByIndex(0)->getTypeLayout();
    // The generated parameter retains its shader-record role.
    SLANG_CHECK(recordType->getSize(slang::ParameterCategory::ShaderRecord) == 1);
    // Record is T, so its resource needs remain unresolved.
    SLANG_CHECK(recordType->getSize(slang::ParameterCategory::GenericResource) == 1);
    // No descriptor binding is allocated before T is known.
    SLANG_CHECK(recordType->getSize(slang::ParameterCategory::DescriptorTableSlot) == 0);
    // Entry totals preserve the same unresolved requirement and no descriptor binding.
    SLANG_CHECK(entry->getTypeLayout()->getSize(slang::ParameterCategory::GenericResource) == 1);
    SLANG_CHECK(
        entry->getTypeLayout()->getSize(slang::ParameterCategory::DescriptorTableSlot) == 0);
}
