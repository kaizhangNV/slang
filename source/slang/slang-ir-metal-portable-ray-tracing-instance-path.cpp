#include "slang-ir-metal-portable-ray-tracing-instance-path.h"

#include "slang-ir-util.h"

namespace Slang
{
namespace PortableRT
{
namespace Metal
{

/// Builds one path operation against the canonical native array-view type. This type is not an
/// ordinary Slang aggregate: Metal passes it by value at an intersection-function boundary.
IRFunc* createInstancePathOperation(IRModule* module, InstancePathOperation operation)
{
    IRBuilder builder(module);
    builder.setInsertInto(module->getModuleInst());
    auto pathType = builder.getType(kIROp_MetalRTInstancePath, 0, nullptr);
    auto function = builder.createFunc();
    IRType* parameters[] = {pathType, builder.getUIntType()};
    switch (operation)
    {
    case InstancePathOperation::Count:
        function->setFullType(builder.getFuncType(1, parameters, builder.getUIntType()));
        builder.addTargetIntrinsicDecoration(
            function,
            CapabilitySet::makeEmpty(),
            toSlice("uint(($0).size())"));
        break;
    case InstancePathOperation::Element:
        function->setFullType(builder.getFuncType(2, parameters, builder.getUIntType()));
        builder.addTargetIntrinsicDecoration(
            function,
            CapabilitySet::makeEmpty(),
            toSlice("($0)[$1]"));
        break;
    case InstancePathOperation::Empty:
        function->setFullType(builder.getFuncType(0, nullptr, pathType));
        builder.addTargetIntrinsicDecoration(
            function,
            CapabilitySet::makeEmpty(),
            toSlice("metal::array_ref<uint>()"));
        break;
    default:
        SLANG_UNEXPECTED("invalid instance path operation");
    }
    return function;
}

IRFunc* createInstanceContribution(IRModule* module, IRType* instancePathType)
{
    IRBuilder builder(module);
    builder.setInsertInto(module->getModuleInst());
    auto result = builder.createFunc();
    auto uintType = builder.getUIntType();
    auto metadataType = builder.getPtrType(uintType, AddressSpace::Global);
    IRType* types[] = {metadataType, instancePathType};
    result->setFullType(builder.getFuncType(2, types, uintType));
    builder.addNameHintDecoration(result, toSlice("portableRTInstanceContribution"));
    bool isMultiLevel = instancePathType->getOp() == kIROp_MetalRTInstancePath;
    auto getCount =
        isMultiLevel ? createInstancePathOperation(module, InstancePathOperation::Count) : nullptr;
    auto getElement = isMultiLevel
                          ? createInstancePathOperation(module, InstancePathOperation::Element)
                          : nullptr;

    builder.setInsertInto(result);
    builder.emitBlock();
    auto metadata = builder.emitParam(metadataType);
    builder.addNameHintDecoration(metadata, toSlice("instanceContributionMetadata"));
    IRInst* instancePath = builder.emitParam(instancePathType);
    if (!isMultiLevel)
    {
        SLANG_RELEASE_ASSERT(instancePathType == uintType);
        builder.emitReturn(builder.emitLoad(builder.emitGetOffsetPtr(metadata, instancePath)));
        return result;
    }

    // Consider paths [0, 0] and [1, 0]. Their leaf instance indices match, but their SBT
    // contributions may differ. The host therefore stores a trie for multilevel traversal.
    // Each intermediate value is a word offset from the metadata buffer's start; the final
    // value is the contribution. This pointer belongs to the AS passed to trace, so two ASes
    // can use the same program records but supply different values for path [0, 0]. Following
    // the whole path keeps candidate and committed-hit selection equal. An empty path returns
    // zero without reading metadata, as required for a directly traced primitive AS.
    auto count = builder.emitCallInst(uintType, getCount, 1, &instancePath);
    auto nodeOrContribution = builder.emitVar(uintType);
    builder.addNameHintDecoration(nodeOrContribution, toSlice("nodeOrContribution"));
    builder.emitStore(nodeOrContribution, builder.getIntValue(uintType, 0));
    IRBlock* body = nullptr;
    IRBlock* exit = nullptr;
    auto depth = emitLoopBlocks(&builder, builder.getIntValue(uintType, 0), count, body, exit);
    builder.setInsertBefore(body->getFirstOrdinaryInst());
    IRInst* arguments[] = {instancePath, depth};
    auto instanceIndex = builder.emitCallInst(uintType, getElement, 2, arguments);
    auto index = builder.emitAdd(uintType, builder.emitLoad(nodeOrContribution), instanceIndex);
    builder.emitStore(
        nodeOrContribution,
        builder.emitLoad(builder.emitGetOffsetPtr(metadata, index)));
    builder.setInsertInto(exit);
    builder.emitReturn(builder.emitLoad(nodeOrContribution));
    return result;
}

} // namespace Metal
} // namespace PortableRT
} // namespace Slang
