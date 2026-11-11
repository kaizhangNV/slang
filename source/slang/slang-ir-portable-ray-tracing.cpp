#include "slang-ir-portable-ray-tracing.h"

#include "slang-ir-insts.h"
#include "slang-ir-metal-portable-ray-tracing.h"
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

/// Creates a new entry-point wrapper that calls the original function. Consider a selected
/// Miss::invoke, shown here as pseudocode:
///     Before: [entry point] Miss::invoke(MissInput<C> input) { ... }
///     After:  [entry point] entry() { Miss::invoke(default(MissInput<C>)); }
///             Miss::invoke(MissInput<C> input) { ... }
/// Run after specialization. The entry-point marker moves to the wrapper, while existing calls
/// still target invoke. Native lowering adds the wrapper's payload parameters and record access
/// in this same pass, so an ordinary call keeps using its caller's invocation.
/// Denormal execution modes move with the entry marker because they apply to the wrapper.
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
        case kIROp_FpDenormalPreserveDecoration:
        case kIROp_FpDenormalFlushToZeroDecoration:
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

/// Groups the selected entries and getters that use one kind and type of invocation value.
/// The stage pointers refer to the completed stages list, which stays unchanged during lowering.
struct NativeInputUses
{
    List<const NativeStage*> stages;
    List<IRInst*> getters;
};

using NativeInputGroups = OrderedDictionary<IRType*, NativeInputUses>;

/// Returns a type's group, preserving the order in which types are first encountered.
NativeInputUses& getInputUses(NativeInputGroups& groups, IRType* type)
{
    groups.addIfNotExists(type, NativeInputUses());
    return groups[type].getValue();
}

/// Supplies a group's entry values, then replaces its getters through ordinary helper parameters.
/// Record handlers bind each entry's resource first; payload/attribute handlers create parameters
/// here even when an entry has no getter, because those parameters are required by the native ABI.
void lowerInputUses(PortableRTInvocationParameter& parameter, const NativeInputUses& uses)
{
    for (auto stage : uses.stages)
        parameter.get(stage->adapter);
    for (auto getter : uses.getters)
        parameter.lower(getter);
}

/// Lowers mutable payload/data access through ordinary references. Empty and scalar payloads
/// remain intact for the existing native payload legalization to adapt later.
void lowerPayloadGroups(IRModule* module, NativeInputGroups& groups, bool callable)
{
    IRBuilder builder(module);
    for (auto& group : groups)
    {
        auto parameterType = builder.getBorrowInOutParamType(group.key);
        NativeInvocationParameter parameter = {
            module,
            parameterType,
            parameterType,
            callable ? LayoutResourceKind::CallablePayload : LayoutResourceKind::RayPayload,
            callable ? "data" : "payload",
            callable ? nullptr : "SV_RayPayload",
            true};
        lowerInputUses(parameter, group.value);
    }
}

/// Supplies application-defined hit attributes through borrowed native parameters.
void lowerAttributeGroups(IRModule* module, NativeInputGroups& groups)
{
    IRBuilder builder(module);
    for (auto& group : groups)
    {
        NativeInvocationParameter parameter = {
            module,
            group.key,
            builder.getBorrowInParamType(group.key, AddressSpace::Generic),
            LayoutResourceKind::HitAttributes,
            "attributes",
            "SV_IntersectionAttributes",
            false};
        lowerInputUses(parameter, group.value);
    }
}

/// Extracts triangle barycentrics from the native attribute struct and forwards them to helpers.
void lowerTriangleAttributes(IRModule* module, const NativeInputUses& uses)
{
    if (uses.stages.getCount() == 0 && uses.getters.getCount() == 0)
        return;

    IRBuilder builder(module);
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
    lowerInputUses(parameter, uses);
}

/// Gives each native invocation its selected SBT record while allowing helpers to share code.
/// A record resource belongs to an entry, so two entries using the same record type still get
/// separate resources. Empty records and OptiX SBT reads need no forwarded record parameter.
void lowerRecordGroups(
    IRModule* module,
    TargetRequest* target,
    NativeInputGroups& groups,
    const List<IRType*>& usedTypes)
{
    IRBuilder builder(module);
    for (auto recordType : usedTypes)
    {
        auto& uses = groups[recordType].getValue();
        // Check each type once, even when many getters read the same nested record.
        if (isEmptyType(recordType))
        {
            for (auto getter : uses.getters)
            {
                builder.setInsertBefore(getter);
                getter->replaceUsesWith(
                    as<IRVoidType>(recordType) ? builder.getVoidValue()
                                               : builder.emitDefaultConstruct(recordType));
                getter->removeAndDeallocate();
            }
            continue;
        }

        builder.setInsertInto(module->getModuleInst());
        auto bufferType = builder.getConstantBufferType(
            recordType,
            builder.getType(kIROp_DefaultBufferLayoutType));
        if (isCUDATarget(target))
        {
            for (auto getter : uses.getters)
            {
                builder.setInsertBefore(getter);
                auto recordPointer =
                    builder.emitIntrinsicInst(bufferType, kIROp_GetOptiXSbtDataPtr, 0, nullptr);
                getter->replaceUsesWith(builder.emitLoad(recordPointer));
                getter->removeAndDeallocate();
            }
            continue;
        }

        SLANG_RELEASE_ASSERT(isKhronosTarget(target) || isD3DTarget(target));
        PortableRTInvocationParameter parameter(module, recordType, "record");
        for (auto stage : uses.stages)
        {
            builder.setInsertBefore(stage->adapter);
            auto record = builder.createGlobalParam(bufferType);
            builder.addNameHintDecoration(record, UnownedTerminatedStringSlice("record"));
            builder.addEntryPointParamDecoration(record, stage->adapter);
            // Reuse the reflected storage, packing, and binding so the generated shader reads
            // the record bytes that the host prepared for this particular entry.
            SLANG_RELEASE_ASSERT(stage->recordLayout);
            builder.addLayoutDecoration(record, stage->recordLayout);
            builder.setInsertBefore(stage->adapter->getFirstBlock()->getFirstOrdinaryInst());
            parameter.bind(stage->adapter, builder.emitLoad(record));
        }
        lowerInputUses(parameter, uses);
    }
}

/// Lowers native stage inputs after collecting each entry and getter once.
/// Consider a closest-hit shader that reads input.payload, input.attributes, and input.record.
/// Each getter joins the group for its kind and type; only that group's entries and getters are
/// visited when lowering it. Keep separate maps because attributes and records can share a type
/// while referring to different invocation values. Each group uses the existing helper-parameter
/// forwarding algorithm.
void lowerNativeInputs(
    IRModule* module,
    TargetRequest* target,
    const List<NativeStage>& stages,
    DiagnosticSink* sink)
{
    NativeInputGroups payloads;
    NativeInputGroups callableData;
    NativeInputGroups attributes;
    NativeInputUses triangles;
    NativeInputGroups records;
    List<IRType*> usedRecordTypes;

    // Index entries once. Payload and attribute groups also retain unused ABI parameters.
    // Record entries are indexed here, but only a getter activates a record type below.
    for (const auto& stage : stages)
    {
        auto kind = Stage(stage.input->getStageKind()->getValue());
        if (kind != Stage::Intersection)
        {
            auto& groups = kind == Stage::Callable ? callableData : payloads;
            getInputUses(groups, stage.input->getPayloadType()).stages.add(&stage);
        }
        if (kind == Stage::ClosestHit || kind == Stage::AnyHit)
        {
            if (getPrimitiveKind(stage.input) == KnownBuiltinDeclName::RayTracingTrianglePrimitive)
                triangles.stages.add(&stage);
            else
                getInputUses(attributes, stage.input->getAttributesType()).stages.add(&stage);
        }
        getInputUses(records, stage.input->getRecordType()).stages.add(&stage);
    }

    List<IRInst*> operations;
    // Logical Input<C> structs have no fields to load. Their getters remain portable operations
    // until this pass supplies native parameters or record loads, including inside helpers.
    collectInputOperations(module->getModuleInst(), operations);
    IRBuilder builder(module);
    for (auto operation : operations)
    {
        switch (operation->getOp())
        {
        case kIROp_PortableRTGetPayload:
        case kIROp_PortableRTGetCallableData:
            {
                auto& groups =
                    operation->getOp() == kIROp_PortableRTGetPayload ? payloads : callableData;
                auto type = cast<IRPtrTypeBase>(operation->getDataType())->getValueType();
                getInputUses(groups, type).getters.add(operation);
                break;
            }
        case kIROp_PortableRTGetHitAttributes:
            {
                auto input = PortableRT::getStageInputInfo(operation->getOperand(0)->getDataType());
                SLANG_RELEASE_ASSERT(input);
                if (getPrimitiveKind(input) == KnownBuiltinDeclName::RayTracingTrianglePrimitive)
                {
                    // TriangleData is a token for builtin properties. Barycentric getters below
                    // obtain the native value, even through helpers or copied TriangleData values.
                    builder.setInsertBefore(operation);
                    operation->replaceUsesWith(
                        builder.emitDefaultConstruct(operation->getDataType()));
                    operation->removeAndDeallocate();
                }
                else
                    getInputUses(attributes, operation->getDataType()).getters.add(operation);
                break;
            }
        case kIROp_PortableRTGetTriangleBarycentricCoord:
            triangles.getters.add(operation);
            break;
        case kIROp_PortableRTGetRecord:
            {
                auto type = operation->getDataType();
                auto& uses = getInputUses(records, type);
                // Process record types in getter order and create no resource for stage-only types.
                if (uses.getters.getCount() == 0)
                    usedRecordTypes.add(type);
                uses.getters.add(operation);
                break;
            }
        case kIROp_PortableRTGetCurveParameter:
            sink->diagnose(Diagnostics::Unimplemented{
                .feature = "portable ray-tracing curve input on a native SBT target",
                .location = operation->sourceLoc});
            break;
        default:
            SLANG_UNEXPECTED("unexpected portable ray-tracing input operation");
        }
    }

    // Preserve native parameter order and the existing helper-argument order across input kinds.
    lowerPayloadGroups(module, payloads, false);
    lowerPayloadGroups(module, callableData, true);
    lowerAttributeGroups(module, attributes);
    lowerTriangleAttributes(module, triangles);
    lowerRecordGroups(module, target, records, usedRecordTypes);
}

} // namespace

void lowerPortableRayTracingEntryPoints(
    IRModule* module,
    TargetProgram* targetProgram,
    List<IRFunc*>& entryPoints,
    DiagnosticSink* sink,
    bool& requiresResultTypeLowering)
{
    auto target = targetProgram->getTargetReq();
    if (isMetalTarget(target))
    {
        lowerMetalPortableRayTracing(
            module,
            targetProgram,
            entryPoints,
            sink,
            requiresResultTypeLowering);
        return;
    }
    if (!isD3DTarget(target) && !isKhronosTarget(target) && !isCUDATarget(target))
        return;

    List<NativeStage> stages;
    // Create a wrapper entry point for each selected PortableRT stage.
    // Turn the original entry point into an ordinary function called by the wrapper.
    for (auto& entryPoint : entryPoints)
    {
        auto input = PortableRT::getEntryPointInputInfo(entryPoint);
        if (!input)
            continue;
        auto layoutDecoration = entryPoint->findDecoration<IRLayoutDecoration>();
        SLANG_RELEASE_ASSERT(layoutDecoration);
        auto sourceLayout = cast<IREntryPointLayout>(layoutDecoration->getLayout());
        SLANG_RELEASE_ASSERT(
            Stage(input->getStageKind()->getValue()) ==
            entryPoint->findDecoration<IREntryPointDecoration>()->getProfile().getStage());
        IRVarLayout* recordLayout = nullptr;
        for (auto field : getScopeStructLayout(sourceLayout)->getFieldLayoutAttrs())
        {
            if (field->getLayout()->findOffsetAttr(LayoutResourceKind::ShaderRecord))
            {
                SLANG_RELEASE_ASSERT(!recordLayout);
                recordLayout = field->getLayout();
            }
        }
        entryPoint = createAdapter(module, entryPoint);
        stages.add({entryPoint, input, recordLayout});
    }

    // Remove any remaining entry markers and denormal modes from logical implementations.
    // Selected stages now have wrappers; implementations reached only through calls are helpers.
    PortableRT::demoteStageImplementations(module);

    lowerNativeInputs(module, target, stages, sink);

    // Rebuild each wrapper's parameter layout now that payload and attribute parameters exist.
    // createAdapter started with an empty parameter layout. Reuse the layouts attached to the
    // generated parameters and preserve the existing result layout.
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
    }
}
} // namespace Slang
