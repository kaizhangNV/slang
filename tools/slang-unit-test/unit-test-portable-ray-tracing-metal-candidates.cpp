#include "core/slang-string.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <string.h>

using namespace Slang;

namespace
{

// Count only the selected entry's outer parameter list; the intersection adapter also has a
// visible-function-table parameter whose function signature contains its own comma.
int countCandidateParameters(const char* declaration)
{
    auto cursor = strchr(declaration, '(');
    if (!cursor)
        return -1;
    int depth = 1;
    int count = 1;
    for (++cursor; *cursor; ++cursor)
    {
        if (*cursor == '(')
            ++depth;
        else if (*cursor == ')' && --depth == 0)
            return count;
        else if (*cursor == ',' && depth == 1)
            ++count;
    }
    return -1;
}

} // namespace

// Compile each candidate shader from serialized IR without defining any schema. The two filters
// deliberately use different inputs; their ABI must stay identical. The intersection shader must
// call an externally supplied filter handle rather than retaining either filter's implementation.
SLANG_UNIT_TEST(portableRayTracingMetalCandidatesCompileIndependently)
{
    const char* source = R"(
        import slang.raytracing.portable;
        struct Attributes { uint candidate; }
        struct TraceContext : rt::ITraceContext
        {
            typealias AccelerationStructure = rt::AccelerationStructure;
            typealias Motion = rt::NoMotion;
        }
        struct HitContext : rt::IHitContext
        {
            typealias TraceContext = ::TraceContext;
            typealias Payload = uint;
            typealias Record = uint;
            typealias Primitive = rt::BoundingBoxPrimitive<Attributes>;
        }
        struct Resources
        {
            RWStructuredBuffer<uint> observed;
            uint threshold;
        }
        ParameterBlock<Resources> resources;

        [noinline]
        uint firstFilter(in rt::AnyHitInput<HitContext> input)
        {
            input.payload += input.record;
            if (input.payload < resources.threshold)
                input.ignoreHit();
            return input.payload;
        }
        struct FirstAnyHit : rt::IAnyHitShader
        {
            typealias Context = HitContext;
            static void invoke(in rt::AnyHitInput<Context> input)
            {
                resources.observed[0] = firstFilter(input);
            }
        }
        [noinline]
        uint secondFilter(in rt::AnyHitInput<HitContext> input)
        {
            input.payload += input.attributes.candidate + input.hitKind;
            if (input.distance < input.objectSpaceRay.TMax + resources.threshold)
                input.acceptHitAndEndSearch();
            return uint(input.worldSpaceOrigin.x);
        }
        struct SecondAnyHit : rt::IAnyHitShader
        {
            typealias Context = HitContext;
            static void invoke(in rt::AnyHitInput<Context> input)
            {
                resources.observed[1] = secondFilter(input);
            }
        }
        [noinline]
        bool candidateReport(in rt::IntersectionInput<HitContext> input, float distance)
        {
            return input.reportHit(distance, 7, Attributes(input.record));
        }
        struct IntersectionProvider : rt::IIntersectionShader
        {
            typealias Context = HitContext;
            static void invoke(in rt::IntersectionInput<Context> input)
            {
                float before = input.objectSpaceRay.TMax;
                bool accepted = candidateReport(input, 2);
                float after = input.objectSpaceRay.TMax;
                resources.observed[2] = asuint(before - after) + uint(accepted);
            }
        }
    )";
    slang::TargetDesc target = {};
    target.format = SLANG_METAL;
    target.profile = unitTestContext->slangGlobalSession->findProfile("metallib_2_4");
    slang::CompilerOptionEntry options[2] = {};
    options[0].name = slang::CompilerOptionName::ExperimentalFeature;
    options[0].value.kind = slang::CompilerOptionValueKind::Int;
    options[0].value.intValue0 = 1;
    options[1].name = slang::CompilerOptionName::ValidateIr;
    options[1].value.kind = slang::CompilerOptionValueKind::Int;
    options[1].value.intValue0 = 1;
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &target;
    sessionDesc.compilerOptionEntryCount = SLANG_COUNT_OF(options);
    sessionDesc.compilerOptionEntries = options;
    ComPtr<slang::ISession> producingSession;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(unitTestContext->slangGlobalSession->createSession(
        sessionDesc,
        producingSession.writeRef())));
    ComPtr<slang::IBlob> diagnostics;
    auto module = producingSession->loadModuleFromSourceString(
        "independentCandidates",
        "independentCandidates.slang",
        source,
        diagnostics.writeRef());
    SLANG_CHECK_ABORT(module);
    ComPtr<slang::IBlob> serialized;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->serialize(serialized.writeRef())));

    struct Selection
    {
        const char* name;
        SlangStage stage;
        const char* helper;
        int parameterCount;
    };
    const Selection selections[] = {
        {"FirstAnyHit", SLANG_STAGE_ANY_HIT, "firstFilter", 2},
        {"SecondAnyHit", SLANG_STAGE_ANY_HIT, "secondFilter", 2},
        {"IntersectionProvider", SLANG_STAGE_INTERSECTION, "candidateReport", 4},
    };
    for (const auto& selection : selections)
    {
        // A separate session and one selected entry ensure no earlier provider compilation or
        // source AST can supply another shader's implementation to this compilation.
        ComPtr<slang::ISession> session;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(
            unitTestContext->slangGlobalSession->createSession(sessionDesc, session.writeRef())));
        module = session->loadModuleFromIRBlob(
            "independentCandidates",
            "independentCandidates.slang-module",
            serialized,
            diagnostics.writeRef());
        SLANG_CHECK_ABORT(module);
        ComPtr<slang::IEntryPoint> entry;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(module->findAndCheckEntryPoint(
            selection.name,
            selection.stage,
            entry.writeRef(),
            diagnostics.writeRef())));
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
        String text(UnownedStringSlice(
            static_cast<const char*>(code->getBufferPointer()),
            code->getBufferSize()));
        String declaration("uint ");
        declaration.append(selection.name);
        auto signature = strstr(text.getBuffer(), declaration.getBuffer());
        SLANG_CHECK_ABORT(signature);
        SLANG_CHECK(countCandidateParameters(signature) == selection.parameterCount);
        auto visible = strstr(text.getBuffer(), "[[visible]]");
        SLANG_CHECK_ABORT(visible);
        SLANG_CHECK(!strstr(visible + 11, "[[visible]]"));
        SLANG_CHECK(!strstr(text.getBuffer(), "[[intersection("));
        SLANG_CHECK(strstr(text.getBuffer(), "globalResourceContext"));
        SLANG_CHECK(strstr(text.getBuffer(), "observed"));
        SLANG_CHECK(!strstr(text.getBuffer(), "metalRTIgnoreHit"));
        SLANG_CHECK(!strstr(text.getBuffer(), "metalRTAcceptHitAndEndSearch"));
        SLANG_CHECK(!strstr(text.getBuffer(), "metalRTReportHit"));
        for (const auto& other : selections)
            SLANG_CHECK(
                bool(strstr(text.getBuffer(), other.helper)) ==
                (strcmp(other.name, selection.name) == 0));
        if (selection.stage == SLANG_STAGE_INTERSECTION)
            SLANG_CHECK(strstr(
                text.getBuffer(),
                "visible_function_table<uint(void thread*, uchar thread*)>"));
    }
}
