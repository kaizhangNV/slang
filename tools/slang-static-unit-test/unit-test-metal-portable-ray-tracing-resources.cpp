#include "slang/slang-ir-collect-global-uniforms.h"
#include "slang/slang-ir-metal-portable-ray-tracing-resources.h"
#include "slang/slang-ir-util.h"
#include "static-unit-test-env.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

// Match the real layout producer: equal binding footprints retain distinct schema identities as
// layout operands, not decorations added after uniquing otherwise identical type layouts.
struct DescriptorLayoutBuilder : IRTypeLayout::Builder
{
    IRStringLit* name;
    DescriptorLayoutBuilder(IRBuilder* builder, const char* schemaName)
        : IRTypeLayout::Builder(builder)
        , name(builder->getStringValue(UnownedStringSlice(schemaName)))
    {
    }
    void addAttrsImpl(List<IRInst*>& operands) SLANG_OVERRIDE
    {
        IRInst* args[] = {name};
        operands.add(getIRBuilder()->getAttr(kIROp_PortableRTSchemaNameAttr, 1, args));
    }
};

// Resource physicalization consumes descriptor identity, not schema requirements. An empty
// conformance is sufficient here: schema discovery has its own checked-source integration tests.
static IRTraceProgramDescriptorType* createDescriptorType(IRBuilder& builder)
{
    auto schema = builder.createStructType();
    auto contract = builder.createInterfaceType(0, nullptr);
    auto witness = builder.createWitnessTable(contract, schema);
    IRInst* operands[] = {schema, witness};
    return cast<IRTraceProgramDescriptorType>(
        builder.getType(kIROp_TraceProgramDescriptorType, 2, operands));
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

// Checks the private descriptor contents as seen by normal target layout consumers. The five
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

    // Model two specializations of Holder<S> that share one declared field key and binding
    // footprint. Their checked descriptor layouts retain the corresponding schema identities.
    auto sharedKey = builder.createStructKey();
    auto firstHolder = builder.createStructType();
    auto secondHolder = builder.createStructType();
    builder.createStructField(firstHolder, sharedKey, first);
    builder.createStructField(secondHolder, sharedKey, second);
    DescriptorLayoutBuilder opaqueBuilder(&builder, "FirstSchema");
    opaqueBuilder.addResourceUsage(LayoutResourceKind::MetalArgumentBufferElement, LayoutSize(1));
    auto opaqueLayout = opaqueBuilder.build();
    IRStructTypeLayout::Builder holderBuilder(&builder);
    holderBuilder.addResourceUsage(LayoutResourceKind::MetalArgumentBufferElement, LayoutSize(1));
    holderBuilder.addField(
        sharedKey,
        createBinding(builder, opaqueLayout, LayoutResourceKind::MetalArgumentBufferElement, 0));
    auto holderLayout = holderBuilder.build();
    DescriptorLayoutBuilder secondOpaqueBuilder(&builder, "SecondSchema");
    secondOpaqueBuilder.addResourceUsage(
        LayoutResourceKind::MetalArgumentBufferElement,
        LayoutSize(1));
    IRStructTypeLayout::Builder secondHolderBuilder(&builder);
    secondHolderBuilder.addResourceUsage(
        LayoutResourceKind::MetalArgumentBufferElement,
        LayoutSize(1));
    secondHolderBuilder.addField(
        sharedKey,
        createBinding(
            builder,
            secondOpaqueBuilder.build(),
            LayoutResourceKind::MetalArgumentBufferElement,
            0));

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
        createBinding(
            builder,
            secondHolderBuilder.build(),
            LayoutResourceKind::MetalArgumentBufferElement,
            2));
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

    SLANG_CHECK(
        resources[0]
            .resourcesStruct->findDecoration<IRPortableRTSchemaNameDecoration>()
            ->getSchemaTypeMangledName()
            ->getStringSlice() == UnownedStringSlice("FirstSchema"));
    SLANG_CHECK(
        resources[1]
            .resourcesStruct->findDecoration<IRPortableRTSchemaNameDecoration>()
            ->getSchemaTypeMangledName()
            ->getStringSlice() == UnownedStringSlice("SecondSchema"));

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

    DescriptorLayoutBuilder opaqueBuilder(&builder, "GlobalSchema");
    opaqueBuilder.addResourceUsage(LayoutResourceKind::MetalBuffer, LayoutSize(1));
    auto opaqueLayout = opaqueBuilder.build();
    auto descriptor = builder.createGlobalParam(descriptorType);
    builder.addLayoutDecoration(
        descriptor,
        createBinding(builder, opaqueLayout, LayoutResourceKind::MetalBuffer, 3));
    IRTypeLayout::Builder scalarBuilder(&builder);
    scalarBuilder.addResourceUsage(LayoutResourceKind::Uniform, LayoutSize(4));
    auto scalarLayout = scalarBuilder.build();
    auto scalar = builder.createGlobalParam(builder.getUIntType());

    // Element and offset-element layouts deliberately use different field-layout attributes.
    // Both keys must be replaced by collection even though the resource is not a nominal field.
    IRStructTypeLayout::Builder elementBuilder(&builder);
    elementBuilder.addResourceUsage(LayoutResourceKind::Uniform, LayoutSize(4));
    elementBuilder.addResourceUsage(LayoutResourceKind::MetalBuffer, LayoutSize(1));
    elementBuilder.addField(
        scalar,
        createBinding(builder, scalarLayout, LayoutResourceKind::Uniform, 0));
    elementBuilder.addField(
        descriptor,
        createBinding(builder, opaqueLayout, LayoutResourceKind::MetalBuffer, 3));
    auto element = elementBuilder.build();
    IRStructTypeLayout::Builder offsetBuilder(&builder);
    offsetBuilder.addResourceUsageFrom(element);
    offsetBuilder.addField(
        scalar,
        createBinding(builder, scalarLayout, LayoutResourceKind::Uniform, 0));
    offsetBuilder.addField(
        descriptor,
        createBinding(builder, opaqueLayout, LayoutResourceKind::MetalBuffer, 9));
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
    physicalizeMetalRTDescriptorResources(module, resources);

    auto layout = cast<IRVarLayout>(collected->findDecoration<IRLayoutDecoration>()->getLayout());
    auto group = cast<IRParameterGroupTypeLayout>(layout->getTypeLayout());
    auto fields = cast<IRStructTypeLayout>(group->getElementVarLayout()->getTypeLayout());
    auto offsetFields = cast<IRStructTypeLayout>(group->getOffsetElementTypeLayout());
    auto key = fields->getFieldLayoutAttrs()[1]->getFieldKey();
    SLANG_CHECK(as<IRStructKey>(key));
    SLANG_CHECK(key == offsetFields->getFieldLayoutAttrs()[1]->getFieldKey());
    auto typeDecoration = key->findDecoration<IRLayoutFieldTypeDecoration>();
    SLANG_CHECK_ABORT(typeDecoration);
    SLANG_CHECK(typeDecoration->getFieldType() == resources[0].physicalParameterBlockType);
    checkDescriptorLayout(fields->getFieldLayout(1)->getTypeLayout(), resources[0]);
    checkDescriptorLayout(offsetFields->getFieldLayout(1)->getTypeLayout(), resources[0]);
    SLANG_CHECK(
        fields->getFieldLayout(1)->findOffsetAttr(LayoutResourceKind::MetalBuffer)->getOffset() ==
        3);
    SLANG_CHECK(
        offsetFields->getFieldLayout(1)
            ->findOffsetAttr(LayoutResourceKind::MetalBuffer)
            ->getOffset() == 9);
}
