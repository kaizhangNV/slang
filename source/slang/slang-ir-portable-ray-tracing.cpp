#include "slang-ir-portable-ray-tracing.h"

#include "slang-ir-insts.h"
#include "slang-ir-portable-ray-tracing-util.h"
#include "slang-ir-util.h"
#include "slang-rich-diagnostics.h"
#include "slang-target-program.h"
#include "slang-target.h"
#include "slang-type-layout.h"

namespace Slang
{
namespace
{

/// Reads the checked input contract from an ordinary input struct, including an input which an
/// earlier parameter legalization has changed from by-value to borrow-in.
IRPortableRTStageInputDecoration* getInputInfo(IRType* type)
{
    if (auto pointerType = as<IRPtrTypeBase>(type))
        type = pointerType->getValueType();
    return getResolvedInstForDecorations(type)->findDecoration<IRPortableRTStageInputDecoration>();
}

KnownBuiltinDeclName getPrimitiveKind(IRPortableRTStageInputDecoration* info)
{
    auto decoration = info->getPrimitiveType()->findDecoration<IRKnownBuiltinDecoration>();
    return decoration ? decoration->getName() : KnownBuiltinDeclName::COUNT;
}

struct NativeStage
{
    IRFunc* adapter;
    IRPortableRTStageInputDecoration* input;
    IRVarLayout* recordLayout;
};

/// Gives a generated native parameter the same semantic and resource layout that ordinary
/// front-end entry-point parameter checking would produce. Its logical source Input<C> had no
/// native layout: only this adapter owns the target ABI.
void addParameterLayout(
    IRBuilder& builder,
    IRParam* parameter,
    Stage stage,
    LayoutResourceKind resourceKind,
    const char* semantic,
    bool isOutput)
{
    builder.addSimpleDecoration<IRGlobalInputDecoration>(parameter);
    if (isOutput)
        builder.addSimpleDecoration<IRGlobalOutputDecoration>(parameter);
    if (semantic)
        builder.addSemanticDecoration(parameter, UnownedTerminatedStringSlice(semantic));

    IRTypeLayout::Builder typeLayout(&builder);
    typeLayout.addResourceUsage(resourceKind, LayoutSize(1));
    IRVarLayout::Builder layout(&builder, typeLayout.build());
    layout.findOrAddResourceInfo(resourceKind);
    layout.setStage(stage);
    builder.addLayoutDecoration(parameter, layout.build());
}

/// Exposes an invocation value through a native entry-point parameter. Ordinary helper-call
/// threading is target-independent; only this adapter boundary knows the native ABI. For example,
/// triangle barycentrics are a float2 in helpers but come from a borrowed native attribute struct.
struct NativeInvocationParameter : PortableRTInvocationParameter
{
    NativeInvocationParameter(
        IRModule* inModule,
        IRType* inHelperType,
        IRType* inNativeType,
        LayoutResourceKind inResourceKind,
        const char* inName,
        const char* inSemantic,
        bool inIsOutput,
        IRStructKey* inNativeField = nullptr)
        : PortableRTInvocationParameter(inModule, inHelperType, inName)
        , nativeType(inNativeType)
        , resourceKind(inResourceKind)
        , semantic(inSemantic)
        , isOutput(inIsOutput)
        , nativeField(inNativeField)
    {
    }

    IRType* nativeType;
    LayoutResourceKind resourceKind;
    const char* semantic;
    bool isOutput;
    IRStructKey* nativeField = nullptr;
    IRInst* createParameter(IRFunc* function) override
    {
        auto entryPoint = function->findDecoration<IREntryPointDecoration>();
        if (!entryPoint)
            return PortableRTInvocationParameter::createParameter(function);

        auto block = function->getFirstBlock();
        SLANG_RELEASE_ASSERT(block);
        IRBuilder builder(m_module);
        auto parameter = builder.createParam(nativeType);
        parameter->insertBefore(block->getFirstOrdinaryInst());
        builder.addNameHintDecoration(parameter, UnownedTerminatedStringSlice(m_name));
        addParameterLayout(
            builder,
            parameter,
            entryPoint->getProfile().getStage(),
            resourceKind,
            semantic,
            isOutput);

        IRInst* value = parameter;
        if (nativeType != m_valueType)
        {
            builder.setInsertBefore(block->getFirstOrdinaryInst());
            value = builder.emitLoad(parameter);
            if (nativeField)
                value = builder.emitFieldExtract(m_valueType, value, nativeField);
        }
        return value;
    }
};

/// Separates the native entry signature from the callable static implementation. Moving only
/// entry-related decorations preserves the implementation's ordinary symbol and direct callers.
/// For example, selecting SharedMiss while CallingMiss invokes it produces two adapters and one
/// shared callable implementation, not a native entry point called with a logical input.
IRFunc* createAdapter(IRModule* module, IRFunc* invoke)
{
    SLANG_RELEASE_ASSERT(invoke->getParamCount() == 1);
    IRBuilder builder(module);
    builder.setInsertBefore(invoke);
    auto adapter = builder.createFunc();
    adapter->setFullType(builder.getFuncType(List<IRType*>(), builder.getVoidType()));
    auto sourceLayoutDecoration = invoke->findDecoration<IRLayoutDecoration>();
    SLANG_RELEASE_ASSERT(sourceLayoutDecoration);
    auto sourceLayout = cast<IREntryPointLayout>(sourceLayoutDecoration->getLayout());
    sourceLayoutDecoration->removeAndDeallocate();
    for (auto decoration = invoke->getFirstDecoration(); decoration;)
    {
        auto next = decoration->getNextDecoration();
        switch (decoration->getOp())
        {
        case kIROp_EntryPointDecoration:
        case kIROp_KeepAliveDecoration:
            decoration->insertAtStart(adapter);
            break;
        default:
            break;
        }
        decoration = next;
    }
    auto entryPoint = adapter->findDecoration<IREntryPointDecoration>();
    SLANG_RELEASE_ASSERT(entryPoint);
    builder.addNameHintDecoration(adapter, entryPoint->getName()->getStringSlice());
    if (!adapter->findDecoration<IRKeepAliveDecoration>())
        builder.addKeepAliveDecoration(adapter);
    // The reflection layout describes the source contract and its shader-record binding. It is
    // not the adapter's parameter layout: native passes must see the actual (initially empty)
    // ABI rather than interpret a logical input or reflected record as a varying parameter.
    IRStructTypeLayout::Builder paramsTypeLayout(&builder);
    IRVarLayout::Builder paramsLayout(&builder, paramsTypeLayout.build());
    builder.addLayoutDecoration(
        adapter,
        builder.getEntryPointLayout(paramsLayout.build(), sourceLayout->getResultLayout()));

    builder.setInsertInto(adapter);
    builder.emitBlock();
    auto inputType = invoke->getParamType(0);
    auto valueType = inputType;
    if (auto pointerType = as<IRPtrTypeBase>(inputType))
        valueType = pointerType->getValueType();
    builder.addDecoration(
        adapter,
        kIROp_PortableRTStageAdapterDecoration,
        valueType,
        invoke,
        sourceLayout);
    IRInst* input;
    if (auto pointerType = as<IRPtrTypeBase>(inputType))
    {
        input = builder.emitVar(pointerType->getValueType());
        builder.emitStore(input, builder.emitDefaultConstruct(pointerType->getValueType()));
    }
    else
    {
        input = builder.emitDefaultConstruct(inputType);
    }
    builder.emitCallInst(builder.getVoidType(), invoke, 1, &input);
    builder.emitReturn();
    return adapter;
}

bool isInputOperation(IROp op)
{
    switch (op)
    {
    case kIROp_PortableRTGetPayload:
    case kIROp_PortableRTGetCallableData:
    case kIROp_PortableRTGetRecord:
    case kIROp_PortableRTGetHitAttributes:
    case kIROp_PortableRTGetTriangleBarycentricCoord:
    case kIROp_PortableRTGetCurveParameter:
        return true;
    default:
        return false;
    }
}

void collectInputOperations(IRInst* parent, List<IRInst*>& result)
{
    for (auto child : parent->getChildren())
    {
        if (isInputOperation(child->getOp()))
            result.add(child);
        collectInputOperations(child, result);
    }
}

/// Removes native-stage identity from portable implementations not selected in this output.
///
/// Consider compiling both CallingMiss and SharedMiss into separate outputs, where CallingMiss
/// directly invokes SharedMiss. The shared module marks both methods as selected entries, but
/// the CallingMiss output links SharedMiss only as a helper. Only the selected output receives
/// an adapter; the remaining logical implementation must not be interpreted as a native entry
/// by parameter legalization. This operates on the linked module, not the reusable source module.
void demoteUnselectedImplementations(IRInst* parent)
{
    for (auto child : parent->getChildren())
    {
        if (auto function = as<IRFunc>(child))
        {
            auto entryPoint = function->findDecoration<IREntryPointDecoration>();
            if (entryPoint && function->getParamCount() == 1 &&
                getInputInfo(function->getParamType(0)))
                entryPoint->removeAndDeallocate();
        }
        else if (as<IRGeneric>(child) || as<IRBlock>(child))
        {
            demoteUnselectedImplementations(child);
        }
    }
}

/// Lowers mutable payload/data access through ordinary references. The native ABI still requires
/// a payload parameter when a stage does not read it; empty and scalar payloads are left intact
/// for the existing native payload legalization to adapt later.
void lowerPayloads(
    IRModule* module,
    const List<NativeStage>& stages,
    List<IRInst*>& operations,
    bool callable)
{
    List<IRType*> types;
    HashSet<IRType*> seen;
    auto operationKind = callable ? kIROp_PortableRTGetCallableData : kIROp_PortableRTGetPayload;
    for (auto stage : stages)
    {
        auto kind = Stage(stage.input->getStageKind()->getValue());
        if ((kind == Stage::Callable) != callable || kind == Stage::Intersection)
            continue;
        if (seen.add(stage.input->getPayloadType()))
            types.add(stage.input->getPayloadType());
    }
    for (auto operation : operations)
    {
        if (!operation || operation->getOp() != operationKind)
            continue;
        auto type = cast<IRPtrTypeBase>(operation->getDataType())->getValueType();
        if (seen.add(type))
            types.add(type);
    }
    for (auto type : types)
    {
        IRBuilder builder(module);
        auto parameterType = builder.getBorrowInOutParamType(type);
        NativeInvocationParameter parameter = {
            module,
            parameterType,
            parameterType,
            callable ? LayoutResourceKind::CallablePayload : LayoutResourceKind::RayPayload,
            callable ? "data" : "payload",
            callable ? nullptr : "SV_RayPayload",
            true};
        for (auto stage : stages)
        {
            auto kind = Stage(stage.input->getStageKind()->getValue());
            if ((kind == Stage::Callable) == callable && kind != Stage::Intersection &&
                stage.input->getPayloadType() == type)
                parameter.get(stage.adapter);
        }
        for (auto& operation : operations)
        {
            if (!operation || operation->getOp() != operationKind)
                continue;
            if (cast<IRPtrTypeBase>(operation->getDataType())->getValueType() == type)
            {
                parameter.lower(operation);
                operation = nullptr;
            }
        }
    }
}

/// Selects the native triangle/custom attribute ABI from the primitive, not the attribute type's
/// spelling. Built-in TriangleData is a property view; a procedural attribute is application data.
void lowerAttributes(IRModule* module, const List<NativeStage>& stages, List<IRInst*>& operations)
{
    List<IRType*> customTypes;
    HashSet<IRType*> seen;
    bool needsTriangles = false;
    for (auto stage : stages)
    {
        auto kind = Stage(stage.input->getStageKind()->getValue());
        if (kind != Stage::ClosestHit && kind != Stage::AnyHit)
            continue;
        if (getPrimitiveKind(stage.input) == KnownBuiltinDeclName::RayTracingTrianglePrimitive)
            needsTriangles = true;
        else if (seen.add(stage.input->getAttributesType()))
            customTypes.add(stage.input->getAttributesType());
    }
    IRBuilder builder(module);
    for (auto& operation : operations)
    {
        if (!operation)
            continue;
        if (operation->getOp() == kIROp_PortableRTGetTriangleBarycentricCoord)
            needsTriangles = true;
        if (operation->getOp() != kIROp_PortableRTGetHitAttributes)
            continue;
        auto input = getInputInfo(operation->getOperand(0)->getDataType());
        SLANG_RELEASE_ASSERT(input);
        if (getPrimitiveKind(input) == KnownBuiltinDeclName::RayTracingTrianglePrimitive)
        {
            // The aggregate is only a token for builtin properties. Each getter, including one
            // called through an ordinary helper or a copied TriangleData value, reads native state.
            builder.setInsertBefore(operation);
            operation->replaceUsesWith(builder.emitDefaultConstruct(operation->getDataType()));
            operation->removeAndDeallocate();
            operation = nullptr;
        }
        else if (seen.add(operation->getDataType()))
        {
            customTypes.add(operation->getDataType());
        }
    }
    for (auto type : customTypes)
    {
        NativeInvocationParameter parameter = {
            module,
            type,
            builder.getBorrowInParamType(type, AddressSpace::Generic),
            LayoutResourceKind::HitAttributes,
            "attributes",
            "SV_IntersectionAttributes",
            false};
        for (auto stage : stages)
        {
            auto kind = Stage(stage.input->getStageKind()->getValue());
            if ((kind == Stage::ClosestHit || kind == Stage::AnyHit) &&
                getPrimitiveKind(stage.input) !=
                    KnownBuiltinDeclName::RayTracingTrianglePrimitive &&
                stage.input->getAttributesType() == type)
                parameter.get(stage.adapter);
        }
        for (auto& operation : operations)
        {
            if (operation && operation->getOp() == kIROp_PortableRTGetHitAttributes &&
                operation->getDataType() == type)
            {
                parameter.lower(operation);
                operation = nullptr;
            }
        }
    }
    if (!needsTriangles)
        return;

    builder.setInsertInto(module->getModuleInst());
    auto nativeType = builder.createStructType();
    builder.addNameHintDecoration(
        nativeType,
        UnownedTerminatedStringSlice("TriangleHitAttributes"));
    auto barycentrics = builder.createStructKey();
    builder.addNameHintDecoration(barycentrics, UnownedTerminatedStringSlice("barycentrics"));
    auto barycentricType = builder.getVectorType(builder.getFloatType(), 2);
    builder.createStructField(nativeType, barycentrics, barycentricType);
    NativeInvocationParameter parameter = {
        module,
        barycentricType,
        builder.getBorrowInParamType(nativeType, AddressSpace::Generic),
        LayoutResourceKind::HitAttributes,
        "attributes",
        "SV_IntersectionAttributes",
        false,
        barycentrics};
    for (auto stage : stages)
    {
        auto kind = Stage(stage.input->getStageKind()->getValue());
        if ((kind == Stage::ClosestHit || kind == Stage::AnyHit) &&
            getPrimitiveKind(stage.input) == KnownBuiltinDeclName::RayTracingTrianglePrimitive)
            parameter.get(stage.adapter);
    }
    for (auto& operation : operations)
    {
        if (operation && operation->getOp() == kIROp_PortableRTGetTriangleBarycentricCoord)
        {
            parameter.lower(operation);
            operation = nullptr;
        }
    }
}

/// Gives each native invocation its selected SBT record while allowing helpers to share code.
/// The resource is created at an adapter, not in the helper which reads it: two selected stages
/// can call the same helper yet have different records. OptiX already has a native SBT pointer
/// operation and needs no resource parameter or helper argument for that case.
void lowerRecords(
    IRModule* module,
    TargetRequest* target,
    const List<NativeStage>& stages,
    List<IRInst*>& operations)
{
    IRBuilder builder(module);
    List<IRType*> types;
    HashSet<IRType*> seen;
    for (auto& operation : operations)
    {
        if (!operation || operation->getOp() != kIROp_PortableRTGetRecord)
            continue;
        auto recordType = operation->getDataType();
        builder.setInsertBefore(operation);
        if (isEmptyType(recordType))
        {
            // Like an empty ConstantBuffer<T>, an empty record has no physical binding. Its
            // logical value still exists and can flow through ordinary source helpers.
            operation->replaceUsesWith(
                as<IRVoidType>(recordType) ? builder.getVoidValue()
                                           : builder.emitDefaultConstruct(recordType));
            operation->removeAndDeallocate();
            operation = nullptr;
        }
        else if (isCUDATarget(target))
        {
            auto bufferType = builder.getConstantBufferType(
                recordType,
                builder.getType(kIROp_DefaultBufferLayoutType));
            auto recordPointer =
                builder.emitIntrinsicInst(bufferType, kIROp_GetOptiXSbtDataPtr, 0, nullptr);
            operation->replaceUsesWith(builder.emitLoad(recordPointer));
            operation->removeAndDeallocate();
            operation = nullptr;
        }
        else if (seen.add(recordType))
        {
            types.add(recordType);
        }
    }
    if (!isKhronosTarget(target) && !isD3DTarget(target))
        return;

    for (auto recordType : types)
    {
        PortableRTInvocationParameter parameter(module, recordType, "record");
        for (auto stage : stages)
        {
            if (stage.input->getRecordType() != recordType)
                continue;
            builder.setInsertBefore(stage.adapter);
            auto bufferType = builder.getConstantBufferType(
                recordType,
                builder.getType(kIROp_DefaultBufferLayoutType));
            auto record = builder.createGlobalParam(bufferType);
            builder.addNameHintDecoration(record, UnownedTerminatedStringSlice("record"));
            builder.addEntryPointParamDecoration(record, stage.adapter);
            if (isD3DTarget(target))
            {
                // Binding allocation already assigned this constant buffer its ordinary b/space
                // register and its independent ShaderRecord role. Preserve that exact reflected
                // layout rather than reconstruct packing or allocate a second binding here.
                SLANG_RELEASE_ASSERT(stage.recordLayout);
                builder.addLayoutDecoration(record, stage.recordLayout);
            }
            else
            {
                IRTypeLayout::Builder typeLayout(&builder);
                typeLayout.addResourceUsage(LayoutResourceKind::ShaderRecord, LayoutSize(1));
                IRVarLayout::Builder layout(&builder, typeLayout.build());
                layout.findOrAddResourceInfo(LayoutResourceKind::ShaderRecord);
                layout.setStage(Stage(stage.input->getStageKind()->getValue()));
                builder.addLayoutDecoration(record, layout.build());
            }
            builder.setInsertBefore(stage.adapter->getFirstBlock()->getFirstOrdinaryInst());
            parameter.bind(stage.adapter, builder.emitLoad(record));
        }
        for (auto& operation : operations)
        {
            if (operation && operation->getOp() == kIROp_PortableRTGetRecord &&
                operation->getDataType() == recordType)
            {
                parameter.lower(operation);
                operation = nullptr;
            }
        }
    }
}

} // namespace

void preparePortableRayTracingEntryPoints(IRModule* module, List<IRFunc*>& entryPoints)
{
    for (auto& entryPoint : entryPoints)
    {
        if (entryPoint->getParamCount() == 1 && getInputInfo(entryPoint->getParamType(0)))
            entryPoint = createAdapter(module, entryPoint);
    }
    demoteUnselectedImplementations(module->getModuleInst());
}

void lowerPortableRayTracingEntryPoints(
    IRModule* module,
    TargetProgram* targetProgram,
    List<IRFunc*>& entryPoints,
    DiagnosticSink* sink)
{
    auto target = targetProgram->getTargetReq();
    if (!isD3DTarget(target) && !isKhronosTarget(target) && !isCUDATarget(target))
        return;

    List<NativeStage> stages;
    for (auto& entryPoint : entryPoints)
    {
        auto adapter = entryPoint->findDecoration<IRPortableRTStageAdapterDecoration>();
        if (!adapter)
            continue;
        auto input = getInputInfo(adapter->getInputType());
        SLANG_RELEASE_ASSERT(input);
        SLANG_RELEASE_ASSERT(
            Stage(input->getStageKind()->getValue()) ==
            entryPoint->findDecoration<IREntryPointDecoration>()->getProfile().getStage());
        IRVarLayout* recordLayout = nullptr;
        for (auto field : getScopeStructLayout(adapter->getSourceLayout())->getFieldLayoutAttrs())
        {
            if (field->getLayout()->findOffsetAttr(LayoutResourceKind::ShaderRecord))
            {
                SLANG_RELEASE_ASSERT(!recordLayout);
                recordLayout = field->getLayout();
            }
        }
        stages.add({entryPoint, input, recordLayout});
    }

    List<IRInst*> operations;
    collectInputOperations(module->getModuleInst(), operations);
    lowerPayloads(module, stages, operations, false);
    lowerPayloads(module, stages, operations, true);
    lowerAttributes(module, stages, operations);
    lowerRecords(module, target, stages, operations);

    for (auto stage : stages)
    {
        IRBuilder builder(module);
        IRStructTypeLayout::Builder paramsTypeLayout(&builder);
        for (auto param : stage.adapter->getParams())
        {
            auto layout =
                cast<IRVarLayout>(param->findDecoration<IRLayoutDecoration>()->getLayout());
            paramsTypeLayout.addField(builder.createStructKey(), layout);
        }
        IRVarLayout::Builder paramsLayout(&builder, paramsTypeLayout.build());
        auto layoutDecoration = stage.adapter->findDecoration<IRLayoutDecoration>();
        auto oldLayout = cast<IREntryPointLayout>(layoutDecoration->getLayout());
        auto layout =
            builder.getEntryPointLayout(paramsLayout.build(), oldLayout->getResultLayout());
        layoutDecoration->removeAndDeallocate();
        builder.addLayoutDecoration(stage.adapter, layout);

        // The checked contract is needed until this adapter has its native ABI. After that the
        // actual call and record resource own its executable dependencies. Keeping the temporary
        // metadata would unnecessarily retain dead logical input types and inlined invoke bodies.
        stage.adapter->findDecoration<IRPortableRTStageAdapterDecoration>()->removeAndDeallocate();
    }

    for (auto operation : operations)
    {
        if (operation)
        {
            sink->diagnose(Diagnostics::Unimplemented{
                .feature = operation->getOp() == kIROp_PortableRTGetRecord
                               ? "portable ray-tracing shader-record binding"
                               : "portable ray-tracing curve input on a native SBT target",
                .location = operation->sourceLoc});
        }
    }
}
} // namespace Slang
