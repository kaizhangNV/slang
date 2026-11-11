#include "slang/slang-ir-collect-global-uniforms.h"
#include "slang/slang-ir-dce.h"
#include "slang/slang-ir-explicit-global-context.h"
#include "slang/slang-ir-metal-portable-ray-tracing-context.h"
#include "slang/slang-ir-metal-portable-ray-tracing-resources.h"
#include "slang/slang-ir-util.h"
#include "static-unit-test-env.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

// Resource physicalization only consumes descriptor identity. Use an empty conformance and void
// placeholders for the five schema types; checked-source integration tests cover those types.
static IRTraceProgramDescriptorType* createDescriptorType(IRBuilder& builder)
{
    auto schema = builder.createStructType();
    auto contract = builder.createInterfaceType(0, nullptr);
    auto witness = builder.createWitnessTable(contract, schema);
    auto unusedSchemaType = builder.getVoidType();
    IRInst* operands[] = {
        schema,
        witness,
        unusedSchemaType,
        unusedSchemaType,
        unusedSchemaType,
        unusedSchemaType,
        unusedSchemaType};
    return cast<IRTraceProgramDescriptorType>(
        builder.getType(kIROp_TraceProgramDescriptorType, SLANG_COUNT_OF(operands), operands));
}

// Creates one normal binding with a nonzero space so rebuilding cannot accidentally preserve
// only its register index. The same helper represents fields inside existing aggregate layouts.
static IRVarLayout* createBinding(
    IRBuilder& builder,
    IRTypeLayout* typeLayout,
    LayoutResourceKind kind,
    UInt offset,
    UInt space = 0)
{
    IRVarLayout::Builder result(&builder, typeLayout);
    auto info = result.findOrAddResourceInfo(kind);
    info->offset = LayoutOffset(offset);
    info->space = space;
    return result.build();
}

// Checks the private descriptor contents as seen by normal target layout consumers. The generated
// field keys must belong to this descriptor, even when its enclosing generic shares source keys.
static void checkDescriptorLayout(IRTypeLayout* layout, const MetalRTDescriptorResources& resources)
{
    auto group = as<IRParameterGroupTypeLayout>(layout);
    SLANG_CHECK_ABORT(group != nullptr);
    auto fields = as<IRStructTypeLayout>(group->getElementVarLayout()->getTypeLayout());
    SLANG_CHECK_ABORT(fields && fields->getFieldCount() == UInt(MetalRTDescriptorResource::Count));
    for (UInt i = 0; i < fields->getFieldCount(); ++i)
    {
        SLANG_CHECK(
            fields->getFieldLayoutAttrs()[i]->getFieldKey() == resources.fields[i]->getKey());
        auto offset = fields->getFieldLayout(i)->findOffsetAttr(
            LayoutResourceKind::MetalArgumentBufferElement);
        SLANG_CHECK_ABORT(offset != nullptr);
        SLANG_CHECK(offset->getOffset() == i);
    }
}

SLANG_UNIT_TEST(metalPortableDescriptorNestedLayouts)
{
    StaticUnitTestEnv env(unitTestContext);
    IRFixtureBuilder fixture(env.getSessionImpl());
    auto module = fixture.getModule();
    IRBuilder builder(module);
    builder.setInsertInto(module);

    auto first = createDescriptorType(builder);
    auto second = createDescriptorType(builder);
    auto signature = builder.getFuncType(0, nullptr, builder.getVoidType());
    List<MetalRTDescriptorResources> resources;
    resources.add(createMetalRTDescriptorResources(
        module,
        first,
        UInt(MetalRTTag::Instancing),
        0,
        signature,
        signature,
        signature,
        signature,
        signature));
    resources.add(createMetalRTDescriptorResources(
        module,
        second,
        UInt(MetalRTTag::Instancing),
        0,
        signature,
        signature,
        signature,
        signature,
        signature));

    // Model two specializations of Holder<S> that share one declared field key and source layout.
    // Each descriptor's concrete type must still select its own generated resource fields.
    auto sharedKey = builder.createStructKey();
    auto firstHolder = builder.createStructType();
    auto secondHolder = builder.createStructType();
    builder.createStructField(firstHolder, sharedKey, first);
    builder.createStructField(secondHolder, sharedKey, second);
    IRTypeLayout::Builder opaqueBuilder(&builder);
    opaqueBuilder.addResourceUsage(LayoutResourceKind::MetalArgumentBufferElement, LayoutSize(1));
    auto opaqueLayout = opaqueBuilder.build();
    IRStructTypeLayout::Builder holderBuilder(&builder);
    holderBuilder.addResourceUsage(LayoutResourceKind::MetalArgumentBufferElement, LayoutSize(1));
    holderBuilder.addField(
        sharedKey,
        createBinding(builder, opaqueLayout, LayoutResourceKind::MetalArgumentBufferElement, 0));
    auto holderLayout = holderBuilder.build();

    auto count = builder.getIntValue(builder.getIntType(), 2);
    auto arrayType = builder.getArrayType(firstHolder, count);
    IRArrayTypeLayout::Builder arrayBuilder(&builder, holderLayout);
    arrayBuilder.addResourceUsage(LayoutResourceKind::MetalArgumentBufferElement, LayoutSize(2));
    arrayBuilder.addResourceUsage(LayoutResourceKind::Uniform, LayoutSize(16));
    arrayBuilder.addAlignment(LayoutResourceKind::Uniform, 8);
    auto arrayLayout = arrayBuilder.build();
    auto tupleType = builder.getTupleType(arrayType, secondHolder);
    IRStructTypeLayout::Builder tupleBuilder(&builder);
    tupleBuilder.addField(
        builder.getIndexedFieldKey(tupleType, 0),
        createBinding(builder, arrayLayout, LayoutResourceKind::MetalArgumentBufferElement, 0));
    tupleBuilder.addField(
        builder.getIndexedFieldKey(tupleType, 1),
        createBinding(builder, holderLayout, LayoutResourceKind::MetalArgumentBufferElement, 2));
    tupleBuilder.addResourceUsage(LayoutResourceKind::MetalArgumentBufferElement, LayoutSize(3));
    auto tupleLayout = tupleBuilder.build();

    IRTypeLayout::Builder containerBuilder(&builder);
    containerBuilder.addResourceUsage(LayoutResourceKind::MetalBuffer, LayoutSize(1));
    auto containerLayout = containerBuilder.build();
    IRParameterGroupTypeLayout::Builder groupBuilder(&builder);
    groupBuilder.addResourceUsageFrom(containerLayout);
    groupBuilder.setContainerVarLayout(
        createBinding(builder, containerLayout, LayoutResourceKind::MetalBuffer, 0));
    groupBuilder.setElementVarLayout(
        createBinding(builder, tupleLayout, LayoutResourceKind::MetalArgumentBufferElement, 0));
    groupBuilder.setOffsetElementTypeLayout(tupleLayout);
    IRInst* groupOperands[] = {tupleType};
    auto groupType = builder.getType(kIROp_ParameterBlockType, 1, groupOperands);
    auto global = builder.createGlobalParam(groupType);
    builder.addLayoutDecoration(
        global,
        createBinding(builder, groupBuilder.build(), LayoutResourceKind::MetalBuffer, 7, 3));

    physicalizeMetalRTDescriptorResources(module, resources);

    auto varLayout = cast<IRVarLayout>(global->findDecoration<IRLayoutDecoration>()->getLayout());
    auto binding = varLayout->findOffsetAttr(LayoutResourceKind::MetalBuffer);
    SLANG_CHECK_ABORT(binding);
    SLANG_CHECK(binding->getOffset() == 7 && binding->getSpace() == 3);
    auto groupLayout = cast<IRParameterGroupTypeLayout>(varLayout->getTypeLayout());
    auto tuple = cast<IRStructTypeLayout>(groupLayout->getElementVarLayout()->getTypeLayout());
    auto array = cast<IRArrayTypeLayout>(tuple->getFieldLayout(0)->getTypeLayout());
    SLANG_CHECK(array->getSizeInBytes().getFiniteValue().getValidValue() == 16);
    SLANG_CHECK(array->getAlignmentInBytes() == 8);
    auto firstLayout = cast<IRStructTypeLayout>(array->getElementTypeLayout());
    auto secondLayout = cast<IRStructTypeLayout>(tuple->getFieldLayout(1)->getTypeLayout());
    checkDescriptorLayout(firstLayout->getFieldLayout(0)->getTypeLayout(), resources[0]);
    checkDescriptorLayout(secondLayout->getFieldLayout(0)->getTypeLayout(), resources[1]);
    SLANG_CHECK(
        findStructField(firstHolder, sharedKey)->getFieldType() ==
        resources[0].physicalParameterBlockType);
    SLANG_CHECK(
        findStructField(secondHolder, sharedKey)->getFieldType() ==
        resources[1].physicalParameterBlockType);
    SLANG_CHECK(firstLayout != secondLayout);
    SLANG_CHECK(groupLayout->getOffsetElementTypeLayout() == tuple);
}

SLANG_UNIT_TEST(metalPortableDescriptorGlobalLayoutOnlyKeys)
{
    StaticUnitTestEnv env(unitTestContext);
    IRFixtureBuilder fixture(env.getSessionImpl());
    auto module = fixture.getModule();
    IRBuilder builder(module);
    builder.setInsertInto(module);
    auto descriptorType = createDescriptorType(builder);
    auto signature = builder.getFuncType(0, nullptr, builder.getVoidType());
    List<MetalRTDescriptorResources> resources;
    resources.add(createMetalRTDescriptorResources(
        module,
        descriptorType,
        UInt(MetalRTTag::Instancing),
        0,
        signature,
        signature,
        signature,
        signature,
        signature));

    IRTypeLayout::Builder opaqueBuilder(&builder);
    opaqueBuilder.addResourceUsage(LayoutResourceKind::MetalBuffer, LayoutSize(1));
    auto opaqueLayout = opaqueBuilder.build();
    auto descriptor = builder.createGlobalParam(descriptorType);
    builder.addLayoutDecoration(
        descriptor,
        createBinding(builder, opaqueLayout, LayoutResourceKind::MetalBuffer, 3, 2));
    IRTypeLayout::Builder scalarBuilder(&builder);
    scalarBuilder.addResourceUsage(LayoutResourceKind::Uniform, LayoutSize(4));
    auto scalarLayout = scalarBuilder.build();
    auto scalar = builder.createGlobalParam(builder.getUIntType());

    // Element and offset-element layouts deliberately use different field-layout attributes.
    // The descriptor remains a standalone global, so neither wrapper layout owns its storage.
    IRStructTypeLayout::Builder elementBuilder(&builder);
    elementBuilder.addResourceUsage(LayoutResourceKind::Uniform, LayoutSize(4));
    elementBuilder.addField(
        scalar,
        createBinding(builder, scalarLayout, LayoutResourceKind::Uniform, 0));
    elementBuilder.addField(
        descriptor,
        createBinding(builder, opaqueLayout, LayoutResourceKind::MetalBuffer, 3, 2));
    auto element = elementBuilder.build();
    IRStructTypeLayout::Builder offsetBuilder(&builder);
    offsetBuilder.addResourceUsageFrom(element);
    offsetBuilder.addField(
        scalar,
        createBinding(builder, scalarLayout, LayoutResourceKind::Uniform, 0));
    offsetBuilder.addField(
        descriptor,
        createBinding(builder, opaqueLayout, LayoutResourceKind::MetalBuffer, 9, 2));
    IRParameterGroupTypeLayout::Builder groupBuilder(&builder);
    groupBuilder.addResourceUsage(LayoutResourceKind::MetalBuffer, LayoutSize(1));
    groupBuilder.setContainerVarLayout(
        createBinding(builder, opaqueLayout, LayoutResourceKind::MetalBuffer, 0));
    groupBuilder.setElementVarLayout(
        createBinding(builder, element, LayoutResourceKind::Uniform, 0));
    groupBuilder.setOffsetElementTypeLayout(offsetBuilder.build());
    auto globalLayout =
        createBinding(builder, groupBuilder.build(), LayoutResourceKind::MetalBuffer, 1);
    collectGlobalUniformParameters(module, globalLayout, CodeGenTarget::Metal);

    IRGlobalParam* collected = nullptr;
    for (auto inst : module->getGlobalInsts())
        if (auto parameter = as<IRGlobalParam>(inst))
            if (as<IRUniformParameterGroupType>(parameter->getDataType()))
                collected = parameter;
    SLANG_CHECK_ABORT(collected);
    builder.addKeepAliveDecoration(collected);

    // Keep a real descriptor use through DCE so explicit-global-context conversion must
    // replace it, after the obsolete wrapper layout attributes have been removed.
    builder.setInsertInto(module);
    auto descriptorGetter = builder.createFunc();
    descriptorGetter->setFullType(builder.getFuncType(0, nullptr, descriptorType));
    builder.addKeepAliveDecoration(descriptorGetter);
    builder.setInsertInto(descriptorGetter);
    builder.emitBlock();
    builder.emitReturn(descriptor);
    physicalizeMetalRTDescriptorResources(module, resources);

    auto layout = cast<IRVarLayout>(collected->findDecoration<IRLayoutDecoration>()->getLayout());
    auto group = cast<IRParameterGroupTypeLayout>(layout->getTypeLayout());
    auto fields = cast<IRStructTypeLayout>(group->getElementVarLayout()->getTypeLayout());
    auto offsetFields = cast<IRStructTypeLayout>(group->getOffsetElementTypeLayout());
    SLANG_CHECK_ABORT(fields->getFieldCount() == 1);
    SLANG_CHECK_ABORT(offsetFields->getFieldCount() == 1);
    auto key = as<IRStructKey>(fields->getFieldLayoutAttrs()[0]->getFieldKey());
    SLANG_CHECK_ABORT(key);
    auto collectedType = cast<IRStructType>(
        cast<IRUniformParameterGroupType>(collected->getDataType())->getElementType());
    SLANG_CHECK_ABORT(findStructField(collectedType, key));
    SLANG_CHECK(key == offsetFields->getFieldLayoutAttrs()[0]->getFieldKey());
    SLANG_CHECK(
        fields->getFieldLayout(0)->findOffsetAttr(LayoutResourceKind::Uniform)->getOffset() == 0);
    SLANG_CHECK(
        offsetFields->getFieldLayout(0)->findOffsetAttr(LayoutResourceKind::Uniform)->getOffset() ==
        0);
    SLANG_CHECK(layout->findOffsetAttr(LayoutResourceKind::MetalBuffer)->getOffset() == 1);

    auto descriptorLayout = findVarLayout(descriptor);
    auto descriptorBinding = descriptorLayout->findOffsetAttr(LayoutResourceKind::MetalBuffer);
    SLANG_CHECK_ABORT(descriptorBinding);
    SLANG_CHECK(descriptorBinding->getOffset() == 3 && descriptorBinding->getSpace() == 2);
    SLANG_CHECK(descriptor->getDataType() == resources[0].physicalParameterBlockType);
    checkDescriptorLayout(descriptorLayout->getTypeLayout(), resources[0]);

    eliminateDeadCode(module, IRDeadCodeEliminationOptions());
    SLANG_CHECK_ABORT(descriptor->hasUses());
    for (auto use = descriptor->firstUse; use; use = use->nextUse)
        SLANG_CHECK(!as<IRStructFieldLayoutAttr>(use->getUser()));
    DiagnosticSink sink;
    introduceExplicitGlobalContext(module, CodeGenTarget::Metal, &sink);
    SLANG_CHECK(sink.getErrorCount() == 0);
    SLANG_CHECK(!descriptor->hasUses());
}

SLANG_UNIT_TEST(metalPortableCollectedGlobalResources)
{
    StaticUnitTestEnv env(unitTestContext);
    for (int variant = 0; variant < 4; ++variant)
    {
        bool hasUniforms = (variant & 1) != 0;
        bool reverseDiscovery = (variant & 2) != 0;
        IRFixtureBuilder fixture(env.getSessionImpl());
        auto module = fixture.getModule();
        IRBuilder builder(module);
        builder.setInsertInto(module);

        // Linking different entries can discover bufferA and bufferB in either order. Their
        // bindings alias, but their linkage names still give both compilations the same order.
        // An ordinary uniform adds a wrapper that must be preserved before both source globals.
        IRType* bufferTypes[] = {
            builder.getPtrType(builder.getUIntType(), AddressSpace::Global),
            builder.getPtrType(builder.getFloatType(), AddressSpace::Global)};
        IRGlobalParam* buffers[2];
        for (UInt i = 0; i < 2; ++i)
        {
            UInt index = reverseDiscovery ? 1 - i : i;
            buffers[index] = builder.createGlobalParam(bufferTypes[index]);
        }
        auto firstBuffer = buffers[0];
        auto secondBuffer = buffers[1];
        builder.addExportDecoration(firstBuffer, UnownedStringSlice("_S_bufferA"));
        builder.addExportDecoration(secondBuffer, UnownedStringSlice("_S_bufferB"));
        IRTypeLayout::Builder bufferBuilder(&builder);
        bufferBuilder.addResourceUsage(LayoutResourceKind::MetalBuffer, LayoutSize(1));
        auto bufferLayout = bufferBuilder.build();
        auto bufferBinding =
            createBinding(builder, bufferLayout, LayoutResourceKind::MetalBuffer, 3);
        builder.addLayoutDecoration(firstBuffer, bufferBinding);
        builder.addLayoutDecoration(secondBuffer, bufferBinding);

        // Coverage instrumentation can create a laid-out global in only one compilation.
        // It has no source linkage and must not change the shared resource structure.
        IRGlobalParam* coverageBuffer = nullptr;
        if (reverseDiscovery)
        {
            coverageBuffer = builder.createGlobalParam(bufferTypes[0]);
            builder.addNameHintDecoration(coverageBuffer, UnownedStringSlice("coverageBuffer"));
            builder.addLayoutDecoration(
                coverageBuffer,
                createBinding(builder, bufferLayout, LayoutResourceKind::MetalBuffer, 9));
        }

        // A specialization constant has source linkage and layout, but is not a runtime resource.
        auto specialization = builder.createGlobalParam(builder.getUIntType());
        builder.addExportDecoration(specialization, UnownedStringSlice("_S_A_specialization"));
        IRTypeLayout::Builder specializationBuilder(&builder);
        specializationBuilder.addResourceUsage(
            LayoutResourceKind::SpecializationConstant,
            LayoutSize(1));
        auto specializationBinding = createBinding(
            builder,
            specializationBuilder.build(),
            LayoutResourceKind::SpecializationConstant,
            4);
        builder.addLayoutDecoration(specialization, specializationBinding);

        IRStructTypeLayout::Builder fieldsBuilder(&builder);
        fieldsBuilder.addResourceUsage(LayoutResourceKind::MetalBuffer, LayoutSize(2));
        fieldsBuilder.addResourceUsage(LayoutResourceKind::SpecializationConstant, LayoutSize(1));
        fieldsBuilder.addField(specialization, specializationBinding);
        fieldsBuilder.addField(firstBuffer, bufferBinding);
        IRGlobalParam* secondUniform = nullptr;
        IRTypeLayout* scalarLayout = nullptr;
        if (hasUniforms)
        {
            IRTypeLayout::Builder scalarBuilder(&builder);
            scalarBuilder.addResourceUsage(LayoutResourceKind::Uniform, LayoutSize(4));
            scalarLayout = scalarBuilder.build();
            auto firstUniform = builder.createGlobalParam(builder.getUIntType());
            secondUniform = builder.createGlobalParam(builder.getUIntType());
            fieldsBuilder.addResourceUsage(LayoutResourceKind::Uniform, LayoutSize(8));
            fieldsBuilder.addField(
                firstUniform,
                createBinding(builder, scalarLayout, LayoutResourceKind::Uniform, 0));
        }
        fieldsBuilder.addField(secondBuffer, bufferBinding);
        if (hasUniforms)
            fieldsBuilder.addField(
                secondUniform,
                createBinding(builder, scalarLayout, LayoutResourceKind::Uniform, 4));
        auto fields = fieldsBuilder.build();
        IRVarLayout* globalLayout = nullptr;
        if (hasUniforms)
        {
            IRParameterGroupTypeLayout::Builder groupBuilder(&builder);
            groupBuilder.addResourceUsage(LayoutResourceKind::MetalBuffer, LayoutSize(1));
            groupBuilder.setContainerVarLayout(
                createBinding(builder, bufferLayout, LayoutResourceKind::MetalBuffer, 0));
            groupBuilder.setElementVarLayout(
                createBinding(builder, fields, LayoutResourceKind::Uniform, 0));
            groupBuilder.setOffsetElementTypeLayout(fields);
            globalLayout =
                createBinding(builder, groupBuilder.build(), LayoutResourceKind::MetalBuffer, 0);
        }
        else
        {
            IRVarLayout::Builder layoutBuilder(&builder, fields);
            globalLayout = layoutBuilder.build();
        }

        collectGlobalUniformParameters(module, globalLayout, CodeGenTarget::Metal);
        List<IRGlobalParam*> globalParams;
        if (hasUniforms)
        {
            for (auto inst : module->getGlobalInsts())
                if (auto parameter = as<IRGlobalParam>(inst))
                    if (findVarLayout(parameter) == globalLayout)
                        globalParams.add(parameter);
            SLANG_CHECK_ABORT(globalParams.getCount() == 1);
            SLANG_CHECK(as<IRUniformParameterGroupType>(globalParams[0]->getDataType()));
        }
        globalParams.add(firstBuffer);
        globalParams.add(secondBuffer);

        // No shader body reads these globals yet. Preservation must keep the source resources
        // and wrapper through DCE, while excluding generated buffers and specialization constants.
        preserveMetalRTGlobalResources(module, globalLayout);
        for (Index i = 0; i < globalParams.getCount(); ++i)
        {
            auto decoration = globalParams[i]->findDecoration<IRMetalRTGlobalResourceDecoration>();
            SLANG_CHECK_ABORT(decoration);
            SLANG_CHECK(getIntVal(decoration->getOperand(0)) == i);
        }
        SLANG_CHECK(!specialization->findDecoration<IRMetalRTGlobalResourceDecoration>());
        if (coverageBuffer)
            SLANG_CHECK(!coverageBuffer->findDecoration<IRMetalRTGlobalResourceDecoration>());
        IRDeadCodeEliminationOptions dceOptions;
        dceOptions.keepGlobalParamsAlive = false;
        eliminateDeadCode(module, dceOptions);
        List<IRGlobalParam*> survivingParams;
        for (auto inst : module->getGlobalInsts())
            if (auto parameter = as<IRGlobalParam>(inst))
                survivingParams.add(parameter);
        SLANG_CHECK_ABORT(survivingParams.getCount() == globalParams.getCount());
        for (auto parameter : globalParams)
            SLANG_CHECK_ABORT(survivingParams.contains(parameter));

        // A separately compiled visible shader must interpret its resource pointer using
        // the same order, including the wrapper even though this shader only reads bufferB.
        builder.setInsertInto(module);
        auto erasedPointer = builder.getPtrType(builder.getUInt8Type(), AddressSpace::ThreadLocal);
        IRType* parameterTypes[] = {erasedPointer};
        auto signature = builder.getFuncType(1, parameterTypes, builder.getVoidType());
        auto stage = builder.getIntValue(builder.getIntType(), IRIntegerValue(Stage::Miss));
        IRInst* tableOperands[] = {signature, stage};
        auto table = builder.getType(kIROp_MetalVisibleFunctionTable, 2, tableOperands);
        auto provider = builder.createFunc();
        provider->setFullType(signature);
        IRInst* providerOperands[] = {stage, table};
        builder.addDecoration(provider, kIROp_MetalVisibleFunctionDecoration, providerOperands, 2);
        builder.setInsertInto(provider);
        builder.emitBlock();
        auto resourceParameter = builder.emitParam(erasedPointer);
        builder.addDecoration(resourceParameter, kIROp_MetalRTGlobalResourceContextDecoration);
        builder.emitLoad(secondBuffer);
        builder.emitReturn();

        DiagnosticSink sink;
        introduceExplicitGlobalContext(module, CodeGenTarget::Metal, &sink);
        IRStructType* sharedType = nullptr;
        for (auto inst : provider->getFirstBlock()->getChildren())
            if (auto cast = as<IRBitCast>(inst))
                if (cast->getOperand(0) == resourceParameter)
                    sharedType =
                        as<IRStructType>(as<IRPtrType>(cast->getDataType())->getValueType());
        SLANG_CHECK_ABORT(sharedType);
        Index fieldIndex = 0;
        for (auto field : sharedType->getFields())
        {
            SLANG_CHECK_ABORT(fieldIndex < globalParams.getCount());
            SLANG_CHECK(field->getFieldType() == globalParams[fieldIndex]->getFullType());
            ++fieldIndex;
        }
        SLANG_CHECK(fieldIndex == globalParams.getCount());
        SLANG_CHECK(sink.getErrorCount() == 0);
    }
}
