#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

// The host requests bindings before asking for target code. Descriptor layout must therefore
// agree with target lowering without requiring the latter to run as a side effect of reflection.
SLANG_UNIT_TEST(portableRayTracingDescriptorLayoutBeforeCodegen)
{
    const char* source = R"(
        import slang.raytracing.portable;

        struct TraceContext : rt::ITraceContext
        {
            typealias AccelerationStructure = rt::AccelerationStructure;
            typealias Motion = rt::NoMotion;
        }
        struct Schema : rt::ITraceProgramSchema
        {
            typealias TraceContext = ::TraceContext;
            typealias HitGroups = rt::EmptyHitGroupList;
            typealias MissShaders = rt::EmptyMissShaderList;
            typealias CallableShaders = rt::EmptyCallableShaderList;
        }

        rt::TraceProgramDescriptor<Schema> first;
        rt::TraceProgramDescriptor<Schema> second;
        ConstantBuffer<uint> ordinary;
    )";

    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(slang_createGlobalSession(0, globalSession.writeRef())));

    const SlangCompileTarget targets[] = {SLANG_HLSL, SLANG_SPIRV, SLANG_CUDA_SOURCE, SLANG_METAL};
    for (auto target : targets)
    {
        slang::TargetDesc targetDesc = {};
        targetDesc.format = target;
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
            "descriptorLayout",
            "descriptorLayout.slang",
            source,
            diagnostics.writeRef());
        SLANG_CHECK_ABORT(module != nullptr);
        auto layout = module->getLayout(0, diagnostics.writeRef());
        SLANG_CHECK_ABORT(layout != nullptr);
        SLANG_CHECK_ABORT(layout->getParameterCount() == 3);

        auto first = layout->getParameterByIndex(0);
        auto second = layout->getParameterByIndex(1);
        const auto buffer = slang::ParameterCategory::ConstantBuffer;
        const auto uniform = slang::ParameterCategory::Uniform;
        if (target == SLANG_METAL)
        {
            SLANG_CHECK(first->getTypeLayout()->getSize(buffer) == 1);
            SLANG_CHECK(second->getTypeLayout()->getSize(buffer) == 1);
            SLANG_CHECK(first->getOffset(buffer) != second->getOffset(buffer));
            SLANG_CHECK(first->getTypeLayout()->getSize(uniform) == 0);
        }
        else
        {
            // Erasure must not consume even a dummy uniform field or a native resource slot.
            SLANG_CHECK(first->getTypeLayout()->getCategoryCount() == 0);
            SLANG_CHECK(second->getTypeLayout()->getCategoryCount() == 0);
            SLANG_CHECK(first->getTypeLayout()->getSize(uniform) == 0);
        }
    }
}

// A provider can declare an acceleration structure without tracing it. Its bindings must already
// describe one Metal argument-buffer pointer; native ray queries keep their native resource type.
SLANG_UNIT_TEST(portableRayTracingAccelerationStructureLayoutBeforeCodegen)
{
    const char* source = R"(
        import slang.raytracing.portable;
        rt::AccelerationStructure first;
        rt::AccelerationStructure second;
        RaytracingAccelerationStructure nativeScene;
        rt::AccelerationStructure scenes[2];
        RaytracingAccelerationStructure nativeScenes[2];
    )";

    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(slang_createGlobalSession(0, globalSession.writeRef())));
    const SlangCompileTarget targets[] = {SLANG_HLSL, SLANG_SPIRV, SLANG_CUDA_SOURCE, SLANG_METAL};
    for (auto target : targets)
    {
        slang::TargetDesc targetDesc = {};
        targetDesc.format = target;
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
            "accelerationStructureLayout",
            "accelerationStructureLayout.slang",
            source,
            diagnostics.writeRef());
        SLANG_CHECK_ABORT(module != nullptr);
        auto layout = module->getLayout(0, diagnostics.writeRef());
        SLANG_CHECK_ABORT(layout != nullptr);
        SLANG_CHECK_ABORT(layout->getParameterCount() == 5);
        auto first = layout->getParameterByIndex(0);
        auto second = layout->getParameterByIndex(1);
        auto nativeScene = layout->getParameterByIndex(2);
        auto scenes = layout->getParameterByIndex(3);
        auto nativeScenes = layout->getParameterByIndex(4);

        // Equal resource footprints do not make these source types interchangeable. In Metal
        // one binds an argument buffer, while the native type binds the AS directly.
        SLANG_CHECK(first->getType() != nativeScene->getType());
        SLANG_CHECK(first->getType() == second->getType());
        SLANG_CHECK(first->getType()->getResourceShape() == SLANG_ACCELERATION_STRUCTURE);
        SLANG_CHECK(nativeScene->getType()->getResourceShape() == SLANG_ACCELERATION_STRUCTURE);
        auto nativeLayout = nativeScene->getTypeLayout();
        SLANG_CHECK(first->getTypeLayout()->getCategoryCount() == nativeLayout->getCategoryCount());
        for (unsigned i = 0; i < nativeLayout->getCategoryCount(); ++i)
        {
            auto category = nativeLayout->getCategoryByIndex(i);
            auto size = nativeLayout->getSize(category);
            SLANG_CHECK(first->getTypeLayout()->getSize(category) == size);
            // A Vulkan descriptor array consumes one binding, while other targets can reserve
            // a slot per element. Compare with the native array instead of assuming two slots.
            SLANG_CHECK(
                scenes->getTypeLayout()->getSize(category) ==
                nativeScenes->getTypeLayout()->getSize(category));
        }
        if (target == SLANG_METAL)
        {
            const auto buffer = slang::ParameterCategory::ConstantBuffer;
            SLANG_CHECK(first->getTypeLayout()->getSize(buffer) == 1);
            SLANG_CHECK(scenes->getTypeLayout()->getSize(buffer) == 2);
            SLANG_CHECK(first->getOffset(buffer) != second->getOffset(buffer));
            SLANG_CHECK(second->getOffset(buffer) != nativeScene->getOffset(buffer));
        }
    }
}
