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
