#include "slang-ir-metal-portable-ray-tracing-resources.h"

#include "slang-ir-util.h"
#include "slang-profile.h"

namespace Slang
{

MetalRTDescriptorResources createMetalRTDescriptorResources(
    IRModule* module,
    IRTraceProgramDescriptorType* sourceType,
    UInt tagMask,
    IRIntegerValue maxLevels,
    IRFuncType* missSignature,
    IRFuncType* closestHitSignature,
    IRFuncType* callableSignature)
{
    SLANG_RELEASE_ASSERT(sourceType && missSignature && closestHitSignature && callableSignature);
    IRBuilder builder(module);
    builder.setInsertInto(module);
    MetalRTDescriptorResources result;
    result.sourceType = sourceType;
    result.resourcesStruct = builder.createStructType();
    builder.addNameHintDecoration(result.resourcesStruct, toSlice("PortableRTDescriptorResources"));

    auto addField = [&](MetalRTDescriptorResource role, IRType* type, const char* name)
    {
        auto key = builder.createStructKey();
        builder.addNameHintDecoration(key, UnownedTerminatedStringSlice(name));
        result.fields[UInt(role)] = builder.createStructField(result.resourcesStruct, key, type);
    };
    IRInst* intersectionOperands[] = {
        builder.getIntValue(builder.getIntType(), tagMask),
        builder.getIntValue(builder.getIntType(), maxLevels),
    };
    addField(
        MetalRTDescriptorResource::IntersectionFunctionTable,
        builder.getType(kIROp_MetalIntersectionFunctionTable, 2, intersectionOperands),
        "intersectionFunctions");
    auto addVisibleTable =
        [&](MetalRTDescriptorResource role, IRFuncType* signature, Stage stage, const char* name)
    {
        IRInst* operands[] = {signature, builder.getIntValue(builder.getIntType(), Int(stage))};
        addField(role, builder.getType(kIROp_MetalVisibleFunctionTable, 2, operands), name);
    };
    addVisibleTable(
        MetalRTDescriptorResource::MissVisibleFunctionTable,
        missSignature,
        Stage::Miss,
        "missFunctions");
    addVisibleTable(
        MetalRTDescriptorResource::ClosestHitVisibleFunctionTable,
        closestHitSignature,
        Stage::ClosestHit,
        "closestHitFunctions");
    addVisibleTable(
        MetalRTDescriptorResource::CallableVisibleFunctionTable,
        callableSignature,
        Stage::Callable,
        "callableFunctions");
    addField(
        MetalRTDescriptorResource::Records,
        builder.getPtrType(builder.getUIntType(), AddressSpace::Global),
        "records");
    IRInst* parameterBlockOperands[] = {result.resourcesStruct};
    result.physicalParameterBlockType = cast<IRUniformParameterGroupType>(
        builder.getType(kIROp_ParameterBlockType, 1, parameterBlockOperands));
    return result;
}

namespace
{

// Preserves binding offsets, spaces, and variable attributes when only a resource's contents
// change. For example, an opaque descriptor at buffer(4) remains at buffer(4) after synthesis.
IRVarLayout* cloneVarLayout(IRBuilder& builder, IRVarLayout* source, IRTypeLayout* typeLayout)
{
    IRVarLayout::Builder result(&builder, typeLayout);
    result.cloneEverythingButOffsetsFrom(source);
    for (auto offsetAttr : source->getOffsetAttrs())
    {
        auto info = result.findOrAddResourceInfo(offsetAttr->getResourceKind());
        info->offset = offsetAttr->getOffset();
        info->space = offsetAttr->getSpace();
    }
    return result.build();
}

// Adds the private argument-buffer IDs without changing the source binding footprint. Source
// reflection already allocated one parameter-block handle; this creates its five internal fields.
IRTypeLayout* createPhysicalLayout(
    IRBuilder& builder,
    const MetalRTDescriptorResources& resources,
    IRTypeLayout* sourceLayout)
{
    // The AST layout producer owns the checked Schema type. Transfer its public identity while
    // this exact descriptor type/layout pair is available; field names or shared generic keys
    // cannot identify which schema a nested descriptor belongs to.
    auto schemaName = sourceLayout->findAttr<IRPortableRTSchemaNameAttr>();
    SLANG_RELEASE_ASSERT(schemaName);
    if (auto owner =
            resources.resourcesStruct->findDecoration<IRMetalRTProgramMetadataDecoration>())
    {
        // Keep this identity on the artifact record rather than on a particular lowered struct.
        // Later legalization may split the descriptor, but every exported table function still
        // refers to the same program record.
        owner->getProgram()->setOperand(0, schemaName->getSchemaTypeMangledName());
    }
    if (auto existing =
            resources.resourcesStruct->findDecoration<IRPortableRTSchemaNameDecoration>())
    {
        SLANG_RELEASE_ASSERT(
            existing->getSchemaTypeMangledName() == schemaName->getSchemaTypeMangledName());
    }
    else
    {
        builder.addDecoration(
            resources.resourcesStruct,
            kIROp_PortableRTSchemaNameDecoration,
            schemaName->getSchemaTypeMangledName());
    }
    IRTypeLayout::Builder fieldTypeBuilder(&builder);
    fieldTypeBuilder.addResourceUsage(
        LayoutResourceKind::MetalArgumentBufferElement,
        LayoutSize(1));
    auto fieldTypeLayout = fieldTypeBuilder.build();
    IRStructTypeLayout::Builder structBuilder(&builder);
    structBuilder.addResourceUsage(
        LayoutResourceKind::MetalArgumentBufferElement,
        LayoutSize(UInt(MetalRTDescriptorResource::Count)));
    for (UInt i = 0; i < UInt(MetalRTDescriptorResource::Count); ++i)
    {
        IRVarLayout::Builder fieldBuilder(&builder, fieldTypeLayout);
        fieldBuilder.findOrAddResourceInfo(LayoutResourceKind::MetalArgumentBufferElement)->offset =
            LayoutOffset(i);
        structBuilder.addField(resources.fields[i]->getKey(), fieldBuilder.build());
    }
    auto structLayout = structBuilder.build();
    IRVarLayout::Builder containerBuilder(&builder, sourceLayout);
    IRVarLayout::Builder elementBuilder(&builder, structLayout);
    IRParameterGroupTypeLayout::Builder groupBuilder(&builder);
    groupBuilder.addResourceUsageFrom(sourceLayout);
    groupBuilder.setContainerVarLayout(containerBuilder.build());
    groupBuilder.setElementVarLayout(elementBuilder.build());
    groupBuilder.setOffsetElementTypeLayout(structLayout);
    return groupBuilder.build();
}

// Rewrites a concrete type and its layout together. Consider Holder<FirstSchema> and
// Holder<SecondSchema>: both specialized structs may reuse the same source field key, but their
// descriptor fields must retain different physical types. The nominal owner resolves that key.
// Layout-only global fields instead carry their checked type from global-parameter collection.
struct LayoutPhysicalizationContext
{
    IRBuilder builder;
    Dictionary<IRType*, const MetalRTDescriptorResources*> resourcesByPhysicalType;
    Dictionary<KeyValuePair<IRType*, IRTypeLayout*>, IRTypeLayout*> rewrittenLayouts;

    LayoutPhysicalizationContext(
        IRModule* module,
        const List<MetalRTDescriptorResources>& resources)
        : builder(module)
    {
        for (const auto& resource : resources)
            resourcesByPhysicalType.add(resource.physicalParameterBlockType, &resource);
    }

    IRVarLayout* rewriteVarLayout(IRType* type, IRVarLayout* layout)
    {
        auto typeLayout = rewriteTypeLayout(type, layout->getTypeLayout());
        return typeLayout == layout->getTypeLayout() ? layout
                                                     : cloneVarLayout(builder, layout, typeLayout);
    }

    // Reads only the producer's canonical representations: an actual nominal field, or an
    // explicitly typed layout-only key. Following arbitrary key uses would confuse generic owners.
    IRType* getFieldType(IRStructType* type, IRInst* key)
    {
        if (auto structKey = as<IRStructKey>(key))
            if (auto field = findStructField(type, structKey))
                return field->getFieldType();
        auto decoration = key->findDecoration<IRLayoutFieldTypeDecoration>();
        SLANG_RELEASE_ASSERT(decoration);
        return decoration->getFieldType();
    }

    IRTypeLayout* rewriteTypeLayout(IRType* type, IRTypeLayout* layout)
    {
        SLANG_RELEASE_ASSERT(type && layout);
        KeyValuePair<IRType*, IRTypeLayout*> cacheKey(type, layout);
        if (auto existing = rewrittenLayouts.tryGetValue(cacheKey))
            return *existing;
        IRTypeLayout* result = layout;
        if (auto resources = resourcesByPhysicalType.tryGetValue(type))
        {
            result = createPhysicalLayout(builder, **resources, layout);
        }
        else if (auto structType = as<IRStructType>(type))
        {
            if (auto structLayout = as<IRStructTypeLayout>(layout))
            {
                IRStructTypeLayout::Builder replacement(&builder);
                replacement.addResourceUsageFrom(layout);
                bool changed = false;
                for (auto field : structLayout->getFieldLayoutAttrs())
                {
                    auto oldFieldLayout = field->getLayout();
                    auto newFieldLayout = rewriteVarLayout(
                        getFieldType(structType, field->getFieldKey()),
                        oldFieldLayout);
                    changed |= oldFieldLayout != newFieldLayout;
                    replacement.addField(field->getFieldKey(), newFieldLayout);
                }
                if (changed)
                    result = replacement.build();
            }
        }
        else if (auto arrayType = as<IRArrayTypeBase>(type))
        {
            auto arrayLayout = as<IRArrayTypeLayout>(layout);
            SLANG_RELEASE_ASSERT(arrayLayout);
            auto elementLayout =
                rewriteTypeLayout(arrayType->getElementType(), arrayLayout->getElementTypeLayout());
            if (elementLayout != arrayLayout->getElementTypeLayout())
            {
                IRArrayTypeLayout::Builder replacement(&builder, elementLayout);
                replacement.addResourceUsageFrom(layout);
                result = replacement.build();
            }
        }
        else if (auto tupleType = as<IRTupleType>(type))
        {
            // Tuple layout is intentionally positional: lowerTypeLayout creates one struct-layout
            // field per value element, excluding the optional trailing tuple-name operand.
            auto structLayout = as<IRStructTypeLayout>(layout);
            SLANG_RELEASE_ASSERT(structLayout);
            IRStructTypeLayout::Builder replacement(&builder);
            replacement.addResourceUsageFrom(layout);
            UInt index = 0;
            bool changed = false;
            for (auto field : structLayout->getFieldLayoutAttrs())
            {
                SLANG_RELEASE_ASSERT(index < tupleType->getOperandCount());
                auto fieldType = cast<IRType>(tupleType->getOperand(index++));
                auto newFieldLayout = rewriteVarLayout(fieldType, field->getLayout());
                changed |= newFieldLayout != field->getLayout();
                replacement.addField(field->getFieldKey(), newFieldLayout);
            }
            if (changed)
                result = replacement.build();
        }
        else if (auto groupType = as<IRUniformParameterGroupType>(type))
        {
            if (auto groupLayout = as<IRParameterGroupTypeLayout>(layout))
            {
                auto newElement = rewriteVarLayout(
                    groupType->getElementType(),
                    groupLayout->getElementVarLayout());
                auto newOffsetElement = rewriteTypeLayout(
                    groupType->getElementType(),
                    groupLayout->getOffsetElementTypeLayout());
                if (newElement != groupLayout->getElementVarLayout() ||
                    newOffsetElement != groupLayout->getOffsetElementTypeLayout())
                {
                    IRParameterGroupTypeLayout::Builder replacement(&builder);
                    replacement.addResourceUsageFrom(layout);
                    replacement.setContainerVarLayout(groupLayout->getContainerVarLayout());
                    replacement.setElementVarLayout(newElement);
                    replacement.setOffsetElementTypeLayout(newOffsetElement);
                    result = replacement.build();
                }
            }
        }
        rewrittenLayouts.add(cacheKey, result);
        return result;
    }
};

// Collects decorations before rewriting because canonical layout rebuilding can deduplicate
// hoistable instructions. Mutating the same instruction list during discovery would skip owners.
void collectVarLayoutDecorations(IRInst* parent, List<IRLayoutDecoration*>& result)
{
    for (auto child : parent->getDecorationsAndChildren())
    {
        if (auto decoration = as<IRLayoutDecoration>(child))
        {
            if (as<IRVarLayout>(decoration->getLayout()))
                result.add(decoration);
        }
        else if (!as<IRDecoration>(child))
            collectVarLayoutDecorations(child, result);
    }
}
} // namespace

void physicalizeMetalRTDescriptorResources(
    IRModule* module,
    const List<MetalRTDescriptorResources>& resources)
{
    // Ordinary replacement updates and deduplicates all dependent types, including pointers,
    // tuples, helper signatures, and fields. No separate SSA propagation graph is necessary.
    for (const auto& resource : resources)
    {
        resource.sourceType->replaceUsesWith(resource.physicalParameterBlockType);
        resource.sourceType->removeAndDeallocate();
    }
    List<IRLayoutDecoration*> decorations;
    collectVarLayoutDecorations(module->getModuleInst(), decorations);
    LayoutPhysicalizationContext context(module, resources);
    for (auto decoration : decorations)
    {
        auto type = decoration->getParent()->getDataType();
        if (!type)
            continue;
        auto oldLayout = cast<IRVarLayout>(decoration->getLayout());
        auto newLayout = context.rewriteVarLayout(type, oldLayout);
        if (newLayout != oldLayout)
            decoration->setOperand(0, newLayout);
    }
}

} // namespace Slang
