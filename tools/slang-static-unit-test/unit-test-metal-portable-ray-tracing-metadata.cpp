#include "compiler-core/slang-artifact-associated-impl.h"
#include "slang/slang-ir-metal-portable-ray-tracing-metadata.h"
#include "slang/slang-ir-metal-portable-ray-tracing-target.h"
#include "slang/slang-profile.h"
#include "static-unit-test-env.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

// Collection reads the surviving exports and their role-keyed metadata. It does not require a
// source schema registry, and an IFT's primitive index is independent of the source hit-group list.
SLANG_UNIT_TEST(metalPortableRayTracingMetadataCollection)
{
    StaticUnitTestEnv env(unitTestContext);
    IRFixtureBuilder fixture(env.getSessionImpl());
    auto module = fixture.getModule();
    IRBuilder builder(module);
    builder.setInsertInto(module);
    auto resources = builder.createStructType();
    builder.addDecoration(
        resources,
        kIROp_PortableRTSchemaNameDecoration,
        builder.getStringValue(UnownedStringSlice("SourceSchemaMangle")));
    addMetalRTProgramMetadata(module, resources, UInt(MetalRTTag::Instancing), 0, 48, 32, 16);

    auto functionType = builder.getFuncType(0, nullptr, builder.getVoidType());
    auto visible = builder.createFunc();
    visible->setFullType(functionType);
    builder.addExternCppDecoration(visible, UnownedStringSlice("physical_miss"));
    auto stageKind = builder.getIntValue(builder.getUIntType(), UInt(Stage::Miss));
    IRInst* tableOperands[] = {functionType, stageKind};
    auto table = builder.getType(kIROp_MetalVisibleFunctionTable, 2, tableOperands);
    IRInst* visibleOperands[] = {stageKind, table};
    builder.addDecoration(visible, kIROp_MetalVisibleFunctionDecoration, visibleOperands, 2);
    addMetalRTFunctionMetadata(
        module,
        visible,
        resources,
        3,
        UInt(MetalRTStageRequirement::Record));

    auto candidate = builder.createFunc();
    candidate->setFullType(functionType);
    builder.addExternCppDecoration(candidate, UnownedStringSlice("physical_curve"));
    IRInst* candidateOperands[] = {
        builder.getIntValue(builder.getUIntType(), UInt(MetalRTGeometryKind::Curve)),
        builder.getIntValue(builder.getUIntType(), UInt(MetalRTTag::CurveData)),
        builder.getIntValue(builder.getUIntType(), 0)};
    builder
        .addDecoration(candidate, kIROp_MetalIntersectionFunctionDecoration, candidateOperands, 3);
    addMetalRTFunctionMetadata(
        module,
        candidate,
        resources,
        UInt(MetalRTCandidateKind::Curve),
        UInt(MetalRTStageRequirement::CurveParameter));

    // Resource legalization moves descriptor type decorations to its ordinary-data part, which
    // may then disappear. Artifact program identity must not depend on that temporary struct.
    auto ordinaryPart = builder.createStructType();
    resources->transferDecorationsTo(ordinaryPart);
    ordinaryPart->removeAndDeallocate();

    auto implementation = new ArtifactPostEmitMetadata;
    ComPtr<IArtifactPostEmitMetadata> owner(implementation);
    auto metadata = static_cast<slang::IMetalRayTracingMetadata*>(
        owner->castAs(slang::IMetalRayTracingMetadata::getTypeGuid()));
    SLANG_CHECK_ABORT(metadata);
    SLANG_CHECK(metadata->getProgramCount() == 0);
    collectMetalRTMetadata(module, *implementation);
    SLANG_CHECK_ABORT(metadata->getProgramCount() == 1);
    slang::MetalRayTracingProgramInfo program;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(metadata->getProgramInfo(0, &program)));
    SLANG_CHECK(program.resourceCount == 5 && program.functionCount == 2);
    SLANG_CHECK(program.hitRecordStride == 48 && program.missRecordStride == 32);
    SLANG_CHECK(program.callableRecordStride == 16);
    SLANG_CHECK(program.recordHeaderSize == 16 && program.recordAlignment == 16);
    slang::MetalRayTracingResourceInfo ift;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(metadata->getProgramResourceInfo(0, 0, &ift)));
    SLANG_CHECK(ift.kind == slang::MetalRayTracingResourceKind::IntersectionFunctionTable);
    SLANG_CHECK(ift.functionTableSize == 3 && ift.argumentBufferID == 0);
    slang::MetalRayTracingResourceInfo miss;
    SLANG_CHECK_ABORT(SLANG_SUCCEEDED(metadata->getProgramResourceInfo(0, 1, &miss)));
    SLANG_CHECK(miss.functionTableSize == 4 && miss.argumentBufferID == 1);
    bool sawMiss = false;
    bool sawCurve = false;
    for (uint32_t i = 0; i < program.functionCount; ++i)
    {
        slang::MetalRayTracingFunctionInfo function;
        SLANG_CHECK_ABORT(SLANG_SUCCEEDED(metadata->getProgramFunctionInfo(0, i, &function)));
        if (function.tableKind == slang::MetalRayTracingResourceKind::MissVisibleFunctionTable)
        {
            sawMiss = true;
            SLANG_CHECK(
                UnownedStringSlice(function.entryPointName) == UnownedStringSlice("physical_miss"));
            SLANG_CHECK(function.functionIndex == 3);
            SLANG_CHECK(function.requirements == uint32_t(MetalRTStageRequirement::Record));
        }
        else
        {
            sawCurve = true;
            SLANG_CHECK(function.primitive == slang::MetalRayTracingPrimitive::Curve);
            SLANG_CHECK(function.functionIndex == 2);
            SLANG_CHECK(
                UnownedStringSlice(function.entryPointName) ==
                UnownedStringSlice("physical_curve"));
        }
    }
    SLANG_CHECK(sawMiss && sawCurve);
    uint32_t index = 9;
    SLANG_CHECK(SLANG_SUCCEEDED(
        metadata->findProgramBySchemaTypeMangledName("SourceSchemaMangle", &index)));
    SLANG_CHECK(index == 0);
    SLANG_CHECK(metadata->getProgramResourceInfo(0, 5, &ift) == SLANG_E_INVALID_ARG);
    SLANG_CHECK(metadata->getProgramFunctionInfo(0, 2, nullptr) == SLANG_E_INVALID_ARG);
}
