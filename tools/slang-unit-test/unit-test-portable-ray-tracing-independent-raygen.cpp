#include "core/slang-string.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

SLANG_UNIT_TEST(portableRayTracingIndependentRayGeneration)
{
    // Consider compiling RayGeneration while another library supplies the schema's stages.
    // Changing stage bodies must not change the ray-generation code or its resource arguments.
    // Keep every declaration fixed, including the stage-only resource, and change only the
    // operations inside Miss.invoke and Intersection.invoke. Both compilations derive their ABI
    // from the schema, including any native intersection function used only for traversal glue.
    const char* source = R"(
        import slang.raytracing.portable;

        struct TraceContext : rt::ITraceContext
        {
            typealias AccelerationStructure = rt::AccelerationStructure;
            typealias Motion = rt::NoMotion;
        }

        struct Record
        {
            uint value;
        }

        struct MissContext : rt::IPayloadContext
        {
            typealias TraceContext = ::TraceContext;
            typealias Payload = uint;
            typealias Record = ::Record;
        }

        StructuredBuffer<uint> stageOnlyBuffer;

        struct Miss : rt::IMissShader
        {
            typealias Context = MissContext;

            static void invoke(in rt::MissInput<Context> input)
            {
                #if BODY_VARIANT == 0
                input.payload = 7;
                #else
                input.payload = input.dispatchRaysIndex.x + input.dispatchRaysDimensions.x
                    + input.record.value + stageOnlyBuffer[0];
                #endif
            }
        }

        struct Attributes
        {
            uint value;
        }

        struct HitContext : rt::IHitContext
        {
            typealias TraceContext = ::TraceContext;
            typealias Payload = uint;
            typealias Record = void;
            typealias Primitive = rt::BoundingBoxPrimitive<Attributes>;
        }

        struct Intersection : rt::IIntersectionShader
        {
            typealias Context = HitContext;

            static void invoke(in rt::IntersectionInput<Context> input)
            {
                #if BODY_VARIANT == 0
                input.reportHit(1, 0, Attributes(1));
                #else
                input.reportHit(2, 0, Attributes(stageOnlyBuffer[1]));
                #endif
            }
        }

        struct Group : rt::IHitGroup
        {
            typealias Context = HitContext;
            typealias ClosestHit = rt::NoClosestHit<Context>;
            typealias AnyHit = rt::NoAnyHit<Context>;
            typealias Intersection = ::Intersection;
        }

        struct Schema : rt::ITraceProgramSchema
        {
            typealias TraceContext = ::TraceContext;
            typealias HitGroups = rt::HitGroupList<Group>;
            typealias MissShaders = rt::MissShaderList<Miss>;
            typealias CallableShaders = rt::EmptyCallableShaderList;
        }

        rt::AccelerationStructure scene;
        rt::TraceProgramDescriptor<Schema> program;
        RWStructuredBuffer<uint> result;

        [shader("raygeneration")]
        void RayGeneration()
        {
            rt::RayTraversalDesc desc = {};
            desc.ray.Direction = float3(0, 0, 1);
            desc.ray.TMax = 10;
            desc.instanceMask = 255;
            desc.sbtStride = 1;
            uint payload = 0;
            rt::RayTracer<Schema> tracer;
            tracer.trace(desc, scene, program, payload);
            result[0] = payload;
        }
    )";

    const struct
    {
        SlangCompileTarget target;
        const char* profile;
        const char* name;
    } targets[] = {
        {SLANG_METAL, "metallib_2_4", "Metal ray-generation body independence"},
        {SLANG_HLSL, "sm_6_6", "HLSL ray-generation body independence"},
        {SLANG_SPIRV_ASM, "spirv_1_5", "SPIR-V ray-generation body independence"},
        {SLANG_CUDA_SOURCE, "sm_6_6", "CUDA ray-generation body independence"},
    };

    for (const auto& target : targets)
    {
        String baseline;
        for (int variant = 0; variant < 2; ++variant)
        {
            slang::TargetDesc targetDesc = {};
            targetDesc.format = target.target;
            targetDesc.profile = unitTestContext->slangGlobalSession->findProfile(target.profile);

            slang::CompilerOptionEntry options[2] = {};
            options[0].name = slang::CompilerOptionName::ExperimentalFeature;
            options[0].value.kind = slang::CompilerOptionValueKind::Int;
            options[0].value.intValue0 = 1;
            options[1].name = slang::CompilerOptionName::EmitSpirvDirectly;
            options[1].value.kind = slang::CompilerOptionValueKind::Int;
            options[1].value.intValue0 = 1;

            slang::PreprocessorMacroDesc bodyVariant = {};
            bodyVariant.name = "BODY_VARIANT";
            bodyVariant.value = variant == 0 ? "0" : "1";

            slang::SessionDesc sessionDesc = {};
            sessionDesc.targetCount = 1;
            sessionDesc.targets = &targetDesc;
            sessionDesc.compilerOptionEntryCount = target.target == SLANG_SPIRV_ASM ? 2 : 1;
            sessionDesc.compilerOptionEntries = options;
            sessionDesc.preprocessorMacroCount = 1;
            sessionDesc.preprocessorMacros = &bodyVariant;

            // Fresh sessions prevent a cached source module from hiding the body change. Keep
            // the module name and path identical so emitted names and source locations match.
            ComPtr<slang::ISession> session;
            SLANG_CHECK_ABORT(SLANG_SUCCEEDED(unitTestContext->slangGlobalSession->createSession(
                sessionDesc,
                session.writeRef())));
            ComPtr<slang::IBlob> diagnostics;
            auto module = session->loadModuleFromSourceString(
                "independentRayGeneration",
                "independentRayGeneration.slang",
                source,
                diagnostics.writeRef());
            SLANG_CHECK_ABORT(module);

            ComPtr<slang::IEntryPoint> entry;
            SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->findAndCheckEntryPoint(
                "RayGeneration",
                SLANG_STAGE_RAY_GENERATION,
                entry.writeRef(),
                diagnostics.writeRef())));

            // The schema contains miss and intersection slots, but neither implementation is
            // selected for this compilation. Their function tables remain part of the ABI.
            slang::IComponentType* components[] = {module, entry};
            ComPtr<slang::IComponentType> composite;
            SLANG_CHECK_ABORT(SLANG_SUCCEEDED(session->createCompositeComponentType(
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
            String output(UnownedStringSlice(
                static_cast<const char*>(code->getBufferPointer()),
                code->getBufferSize()));

            if (variant == 0)
                baseline = output;
            else
            {
                if (output != baseline)
                    getTestReporter()->message(TestMessageType::Info, target.name);
                SLANG_CHECK(output == baseline);
            }

            if (target.target == SLANG_METAL)
            {
                SLANG_CHECK(output.indexOf("[[kernel]]") >= 0);
                SLANG_CHECK(output.indexOf("visible_function_table<") >= 0);
                SLANG_CHECK(output.indexOf("intersection_function_table<") >= 0);
                SLANG_CHECK(output.indexOf("[[visible]]") < 0);
                // Native intersection functions may contain schema-owned traversal glue.
                // The application stage bodies are separately selected VFT functions; body
                // independence above forbids their requirements from changing this glue.
            }
        }
    }
}
