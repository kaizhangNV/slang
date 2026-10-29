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
    SLANG_CHECK_ABORT(layout->getParameterCount() == 2);
    SLANG_CHECK_ABORT(layout->getEntryPointCount() == 6);
    const auto cbuffer = slang::ParameterCategory::ConstantBuffer;
    const auto shaderRecord = slang::ParameterCategory::ShaderRecord;
    const auto uniform = slang::ParameterCategory::Uniform;
    auto global = layout->getParameterByIndex(0);
    auto globalType = global->getTypeLayout();
    SLANG_CHECK(global->getOffset(cbuffer) == 3);
    SLANG_CHECK(global->getBindingSpace(cbuffer) == 0);
    SLANG_CHECK(globalType->getSize(shaderRecord) == 0);
    auto materialLayout = globalType->getElementTypeLayout();
    SLANG_CHECK_ABORT(materialLayout != nullptr);

    slang::VariableLayoutReflection* recordBindings[2];
    for (int i = 0; i < 2; ++i)
    {
        auto entryLayout = layout->getEntryPointByIndex(i);
        SLANG_CHECK_ABORT(entryLayout->getParameterCount() == 1);
        auto record = entryLayout->getParameterByIndex(0);
        recordBindings[i] = record;
        SLANG_CHECK(record->getVariable() == nullptr);
        SLANG_CHECK(record->getName() == nullptr);
        SLANG_CHECK(record->getBindingSpace(cbuffer) != 0);
        SLANG_CHECK(record->getBindingSpace(cbuffer) != 1);
        auto recordType = record->getTypeLayout();
        SLANG_CHECK(recordType->getKind() == slang::TypeReflection::Kind::ConstantBuffer);
        SLANG_CHECK(recordType->getSize(cbuffer) == 1);
        SLANG_CHECK(recordType->getSize(shaderRecord) == 1);
        SLANG_CHECK(recordType->getBindingRangeCount() == 1);
        SLANG_CHECK(recordType->getBindingRangeType(0) == slang::BindingType::ConstantBuffer);
        SLANG_CHECK(
            recordType->getContainerVarLayout()->getTypeLayout()->getSize(shaderRecord) == 1);
        auto element = recordType->getElementTypeLayout();
        SLANG_CHECK_ABORT(element != nullptr);
        SLANG_CHECK(element->getSize(uniform) == materialLayout->getSize(uniform));
        SLANG_CHECK_ABORT(element->getFieldCount() == materialLayout->getFieldCount());
        for (unsigned fieldIndex = 0; fieldIndex < element->getFieldCount(); ++fieldIndex)
        {
            SLANG_CHECK(
                element->getFieldByIndex(fieldIndex)->getOffset(uniform) ==
                materialLayout->getFieldByIndex(fieldIndex)->getOffset(uniform));
        }
    }
    SLANG_CHECK(
        recordBindings[0]->getBindingSpace(cbuffer) == recordBindings[1]->getBindingSpace(cbuffer));
    SLANG_CHECK(recordBindings[0]->getOffset(cbuffer) != recordBindings[1]->getOffset(cbuffer));
    SLANG_CHECK(layout->getEntryPointByIndex(2)->getParameterCount() == 0);
    SLANG_CHECK(layout->getEntryPointByIndex(3)->getParameterCount() == 0);
    const auto shaderResource = slang::ParameterCategory::ShaderResource;
    slang::VariableLayoutReflection* resourceBindings[2];
    for (int i = 0; i < 2; ++i)
    {
        auto entry = layout->getEntryPointByIndex(4 + i);
        SLANG_CHECK_ABORT(entry->getParameterCount() == 1);
        auto record = entry->getParameterByIndex(0);
        resourceBindings[i] = record;
        auto type = record->getTypeLayout();
        SLANG_CHECK(type->getSize(shaderRecord) == 1);
        SLANG_CHECK(type->getSize(shaderResource) == 1);
        SLANG_CHECK(type->getSize(cbuffer) == size_t(i));
        SLANG_CHECK(
            record->getBindingSpace(shaderResource) == recordBindings[0]->getBindingSpace(cbuffer));
    }
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
        SLANG_CHECK(codeText.indexOf(expectedBinding.getUnownedSlice()) >= 0);
        SLANG_CHECK(codeText.indexOf("cbuffer") >= 0);

        // Calling another selected stage as a static helper must not read that stage's own
        // SBT record. In particular, OtherMaterialMiss must use its binding for both calls.
        auto otherRecord = recordBindings[1 - i];
        StringBuilder unexpectedBinding;
        unexpectedBinding << "register(b" << UInt64(otherRecord->getOffset(cbuffer)) << ", space"
                          << UInt64(otherRecord->getBindingSpace(cbuffer)) << ")";
        SLANG_CHECK(codeText.indexOf(unexpectedBinding.getUnownedSlice()) < 0);
    }
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
        SLANG_CHECK(text.indexOf(expectedBinding.getUnownedSlice()) >= 0);
    }
}
