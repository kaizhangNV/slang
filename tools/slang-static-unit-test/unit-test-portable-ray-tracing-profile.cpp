#include "slang-com-ptr.h"
#include "slang.h"
#include "slang/slang-entry-point.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

// Selecting Miss or Miss.invoke must produce a portable_miss profile, even when invoke never
// reads its input. Checking only successful compilation would miss a native profile that the
// capability checker repairs later by inspecting the input type.
SLANG_UNIT_TEST(portableRayTracingEntryPointProfiles)
{
    const char* source = R"(
        import slang.raytracing.portable;

        struct TraceContext : rt::ITraceContext
        {
            typealias AccelerationStructure = rt::AccelerationStructure;
            typealias Motion = rt::NoMotion;
        }
        struct HitContext : rt::IHitContext
        {
            typealias TraceContext = ::TraceContext;
            typealias Payload = uint;
            typealias Record = void;
            typealias Primitive = rt::BoundingBoxPrimitive<float2>;
        }
        struct MissContext : rt::IPayloadContext
        {
            typealias TraceContext = ::TraceContext;
            typealias Payload = uint;
            typealias Record = void;
        }
        struct CallableContext : rt::ICallableContext
        {
            typealias TraceContext = ::TraceContext;
            typealias CallableData = uint;
            typealias Record = void;
        }
        struct ClosestHit : rt::IClosestHitShader
        {
            typealias Context = HitContext;
            static void invoke(rt::ClosestHitInput<Context> input) {}
        }
        struct AnyHit : rt::IAnyHitShader
        {
            typealias Context = HitContext;
            static void invoke(rt::AnyHitInput<Context> input) {}
        }
        struct Intersection : rt::IIntersectionShader
        {
            typealias Context = HitContext;
            static void invoke(rt::IntersectionInput<Context> input) {}
        }
        struct Miss : rt::IMissShader
        {
            typealias Context = MissContext;
            static void invoke(rt::MissInput<Context> input) {}
        }
        struct Callable : rt::ICallableShader
        {
            typealias Context = CallableContext;
            static void invoke(rt::CallableInput<Context> input) {}
        }

        [shader("closesthit")] void NativeClosestHit() {}
        [shader("anyhit")] void NativeAnyHit() {}
        [shader("intersection")] void NativeIntersection() {}
        [shader("miss")] void NativeMiss() {}
        [shader("callable")] void NativeCallable() {}
    )";

    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_HLSL;
    targetDesc.profile = unitTestContext->slangGlobalSession->findProfile("lib_6_6");
    slang::CompilerOptionEntry experimental = {};
    experimental.name = slang::CompilerOptionName::ExperimentalFeature;
    experimental.value.kind = slang::CompilerOptionValueKind::Int;
    experimental.value.intValue0 = 1;
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;
    sessionDesc.compilerOptionEntryCount = 1;
    sessionDesc.compilerOptionEntries = &experimental;
    const char* standardModulePath = SLANG_TEST_STANDARD_MODULE_PATH;
    sessionDesc.searchPathCount = 1;
    sessionDesc.searchPaths = &standardModulePath;
    ComPtr<slang::ISession> session;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
        unitTestContext->slangGlobalSession->createSession(sessionDesc, session.writeRef())));

    ComPtr<slang::IBlob> diagnostics;
    auto module = session->loadModuleFromSourceString(
        "portableProfiles",
        "portableProfiles.slang",
        source,
        diagnostics.writeRef());
    if (!module && diagnostics)
        getTestReporter()->message(
            TestMessageType::Info,
            static_cast<const char*>(diagnostics->getBufferPointer()));
    SLANG_CHECK_ABORT(module);

    struct TestCase
    {
        const char* name;
        Stage stage;
        Slang::CapabilityName portableCapability;
        Slang::CapabilityName nativeCapability;
    };
    const TestCase cases[] = {
        {"ClosestHit",
         Stage::ClosestHit,
         Slang::CapabilityName::portable_closesthit,
         Slang::CapabilityName::closesthit},
        {"AnyHit",
         Stage::AnyHit,
         Slang::CapabilityName::portable_anyhit,
         Slang::CapabilityName::anyhit},
        {"Intersection",
         Stage::Intersection,
         Slang::CapabilityName::portable_intersection,
         Slang::CapabilityName::intersection},
        {"Miss", Stage::Miss, Slang::CapabilityName::portable_miss, Slang::CapabilityName::miss},
        {"Callable",
         Stage::Callable,
         Slang::CapabilityName::portable_callable,
         Slang::CapabilityName::callable},
    };
    for (const auto& testCase : cases)
    {
        ComPtr<slang::IEntryPoint> nativeEntryPoint;
        auto nativeName = String("Native") + testCase.name;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->findAndCheckEntryPoint(
            nativeName.getBuffer(),
            SLANG_STAGE_NONE,
            nativeEntryPoint.writeRef(),
            diagnostics.writeRef())));
        auto nativeProfile = static_cast<EntryPoint*>(nativeEntryPoint.get())->getProfile();
        SLANG_CHECK(nativeProfile.getStage() == testCase.stage);
        SLANG_CHECK(nativeProfile.getCapabilityName() == CapabilitySet{testCase.nativeCapability});
        SLANG_CHECK(nativeProfile.getCapabilityName().isIncompatibleWith(
            CapabilitySet{Slang::CapabilityName::metal}));

        // Both source spellings support inferred or explicit stage selection. None should
        // convert the portable profile back into the native profile for the same Stage value.
        for (const char* suffix : {"", ".invoke"})
        {
            auto name = String(testCase.name) + suffix;
            for (SlangStage requestedStage : {SLANG_STAGE_NONE, SlangStage(testCase.stage)})
            {
                ComPtr<slang::IEntryPoint> entryPoint;
                auto result = module->findAndCheckEntryPoint(
                    name.getBuffer(),
                    requestedStage,
                    entryPoint.writeRef(),
                    diagnostics.writeRef());
                if (SLANG_FAILED(result) && diagnostics)
                    getTestReporter()->message(
                        TestMessageType::Info,
                        static_cast<const char*>(diagnostics->getBufferPointer()));
                SLANG_CHECK_ABORT(SLANG_SUCCEEDED(result));
                auto selected = static_cast<EntryPoint*>(entryPoint.get());
                auto profile = selected->getProfile();
                SLANG_CHECK(selected->isStructuralRayTracingEntryPoint());
                SLANG_CHECK(profile.getStage() == testCase.stage);
                SLANG_CHECK(
                    profile.getCapabilityName() == CapabilitySet{testCase.portableCapability});
                SLANG_CHECK(profile != nativeProfile);
                SLANG_CHECK(!profile.getCapabilityName().isIncompatibleWith(
                    CapabilitySet{Slang::CapabilityName::metal}));

                // Serialized entry-point decorations store the raw profile value. Reconstructing
                // that value must preserve its portable capability as well as its logical stage.
                Profile reconstructed(profile.raw);
                SLANG_CHECK(reconstructed.getStage() == testCase.stage);
                SLANG_CHECK(reconstructed.getCapabilityName() == profile.getCapabilityName());

                // Effective target profiles replace the version; lowering to a different stage
                // replaces the stage contract. Neither operation may corrupt the other field.
                reconstructed.setVersion(ProfileVersion::METAL_2_4);
                auto expectedCaps = CapabilitySet{testCase.portableCapability};
                expectedCaps.join(CapabilitySet{Slang::CapabilityName::METAL_2_4});
                SLANG_CHECK(reconstructed.getStage() == testCase.stage);
                SLANG_CHECK(reconstructed.isPortableRayTracingStage());
                SLANG_CHECK(reconstructed.getCapabilityName() == expectedCaps);
                reconstructed.setStage(Stage::Compute);
                expectedCaps = CapabilitySet{Slang::CapabilityName::compute};
                expectedCaps.join(CapabilitySet{Slang::CapabilityName::METAL_2_4});
                SLANG_CHECK(reconstructed.getVersion() == ProfileVersion::METAL_2_4);
                SLANG_CHECK(!reconstructed.isPortableRayTracingStage());
                SLANG_CHECK(reconstructed.getCapabilityName() == expectedCaps);
            }
        }
    }
}
