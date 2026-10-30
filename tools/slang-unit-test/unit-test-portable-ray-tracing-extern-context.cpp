#include "core/slang-string.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <string.h>

using namespace Slang;

// An entry can be selected before its Context's associated types are known. Serialize that
// module, then supply the Context definition from another module: native adapters must derive
// their concrete payload/attribute ABI after linking, not while first lowering the declaration.
SLANG_UNIT_TEST(portableRayTracingExternContextNativeLink)
{
    const char* stageSource = R"(
        import slang.raytracing.portable;
        extern struct ExternalContext : rt::IHitContext;
        RWStructuredBuffer<float> output;
        RWStructuredBuffer<ExternalContext.Payload> payloadOutput;
        RWStructuredBuffer<ExternalContext.Primitive.Attributes> attributeOutput;

        struct ExternalHit : rt::IClosestHitShader
        {
            typealias Context = ExternalContext;
            static void invoke(in rt::ClosestHitInput<Context> input)
            {
                output[0] = input.distance;
                payloadOutput[0] = input.payload;
                attributeOutput[0] = input.attributes;
            }
        }
        struct GenericHit<C : rt::IHitContext> : rt::IClosestHitShader
        {
            typealias Context = C;
            static void invoke(in rt::ClosestHitInput<Context> input)
            {
                output[1] = input.distance;
            }
        }
    )";
    const char* contextSource = R"(
        import portableExternalStages;
        import slang.raytracing.portable;

        struct TraceContext : rt::ITraceContext
        {
            typealias AccelerationStructure = rt::AccelerationStructure;
            typealias Motion = rt::NoMotion;
        }
        struct ConcreteContext : rt::IHitContext
        {
            typealias TraceContext = ::TraceContext;
            typealias Payload = uint4;
            typealias Record = void;
            typealias Primitive = rt::BoundingBoxPrimitive<Attributes>;
        }
        struct Attributes { uint value; }
        export struct ExternalContext : rt::IHitContext = ConcreteContext;
    )";
    const SlangCompileTarget targets[] = {SLANG_HLSL, SLANG_SPIRV_ASM, SLANG_CUDA_SOURCE};
    for (auto target : targets)
    {
        slang::TargetDesc targetDesc = {};
        targetDesc.format = target;
        targetDesc.profile = unitTestContext->slangGlobalSession->findProfile(
            target == SLANG_SPIRV_ASM ? "spirv_1_5" : "sm_6_6");
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
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
            unitTestContext->slangGlobalSession->createSession(sessionDesc, session.writeRef())));
        ComPtr<slang::IBlob> diagnostics;
        auto stages = session->loadModuleFromSourceString(
            "portableExternalStages",
            "portableExternalStages.slang",
            stageSource,
            diagnostics.writeRef());
        SLANG_CHECK_ABORT(stages);
        ComPtr<slang::IBlob> serializedStages;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(stages->serialize(serializedStages.writeRef())));

        // A fresh session ensures that the imported module's serialized semantic/IR witnesses
        // carry the unresolved associated-type identities; no source AST cache can supply them.
        ComPtr<slang::ISession> importingSession;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(unitTestContext->slangGlobalSession->createSession(
            sessionDesc,
            importingSession.writeRef())));
        stages = importingSession->loadModuleFromIRBlob(
            "portableExternalStages",
            "portableExternalStages.slang-module",
            serializedStages,
            diagnostics.writeRef());
        SLANG_CHECK_ABORT(stages);
        auto context = importingSession->loadModuleFromSourceString(
            "portableExternalContext",
            "portableExternalContext.slang",
            contextSource,
            diagnostics.writeRef());
        SLANG_CHECK_ABORT(context);

        const char* selections[] = {"ExternalHit", "GenericHit<ExternalContext>"};
        for (auto selection : selections)
        {
            ComPtr<slang::IEntryPoint> entry;
            SLANG_CHECK_ABORT(SLANG_SUCCEEDED(stages->findAndCheckEntryPoint(
                selection,
                SLANG_STAGE_CLOSEST_HIT,
                entry.writeRef(),
                diagnostics.writeRef())));
            slang::IComponentType* components[] = {stages, context, entry};
            ComPtr<slang::IComponentType> composite;
            SLANG_CHECK_ABORT(SLANG_SUCCEEDED(importingSession->createCompositeComponentType(
                components,
                SLANG_COUNT_OF(components),
                composite.writeRef(),
                diagnostics.writeRef())));
            ComPtr<slang::IComponentType> linked;
            SLANG_CHECK_ABORT(
                SLANG_SUCCEEDED(composite->link(linked.writeRef(), diagnostics.writeRef())));
            ComPtr<slang::IBlob> code;
            auto result = linked->getEntryPointCode(0, 0, code.writeRef(), diagnostics.writeRef());
            if (SLANG_FAILED(result) && diagnostics)
                getTestReporter()->message(
                    TestMessageType::Info,
                    static_cast<const char*>(diagnostics->getBufferPointer()));
            SLANG_CHECK_ABORT(SLANG_SUCCEEDED(result));
            SLANG_CHECK_ABORT(code && code->getBufferSize());
            auto text = UnownedStringSlice(
                static_cast<const char*>(code->getBufferPointer()),
                code->getBufferSize());
            if (target == SLANG_HLSL)
            {
                SLANG_CHECK(text.indexOf(toSlice("SV_RayPayload")) >= 0);
                SLANG_CHECK(text.indexOf(toSlice("uint4")) >= 0);
                SLANG_CHECK(text.indexOf(toSlice("SV_IntersectionAttributes")) >= 0);
            }
            else if (target == SLANG_SPIRV_ASM)
            {
                SLANG_CHECK(text.indexOf(toSlice("OpEntryPoint ClosestHitKHR")) >= 0);
                // The direct stage consumes both views, so its storage must remain live.
                // GenericHit only reads distance; SPIR-V may remove its unused ABI variables.
                if (strcmp(selection, "ExternalHit") == 0)
                {
                    SLANG_CHECK(text.indexOf(toSlice("IncomingRayPayloadKHR")) >= 0);
                    SLANG_CHECK(text.indexOf(toSlice("HitAttributeKHR")) >= 0);
                }
            }
            else
            {
                SLANG_CHECK(text.indexOf(toSlice("__closesthit__")) >= 0);
                SLANG_CHECK(text.indexOf(toSlice("optixGetRayTmax")) >= 0);
            }
        }
    }
}
