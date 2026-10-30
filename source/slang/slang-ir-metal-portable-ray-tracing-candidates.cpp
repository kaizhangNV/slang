#include "slang-ir-metal-portable-ray-tracing-candidates.h"

#include "slang-ir-clone.h"
#include "slang-ir-lower-error-handling.h"
#include "slang-ir-metal-portable-ray-tracing-instance-path.h"
#include "slang-ir-portable-ray-tracing-util.h"
#include "slang-ir-util.h"

namespace Slang
{
namespace PortableRT
{
namespace Metal
{
namespace
{

// Zero is the ordinary AnyHit return. The other two values are nonlocal stage exits: IgnoreHit
// exits AnyHit, whereas AcceptHitAndEndSearch also exits its enclosing Intersection invocation.
enum class CandidateExit : UInt
{
    Accept,
    Ignore,
    EndSearch,
};

struct CandidateResult
{
    IRStructType* type = nullptr;
    bool hasDistance = false;
};

struct CandidateState
{
    IRStructType* type = nullptr;
    IRStructKey* accepted = nullptr;
    IRStructKey* currentMax = nullptr;
    IRStructKey* distance = nullptr;
    IRStructKey* hitKind = nullptr;
    IRStructKey* attributes = nullptr;
    IRStructKey* minDistance = nullptr;
    IRStructKey* origin = nullptr;
    IRStructKey* direction = nullptr;
    IRStructKey* opaque = nullptr;
};

struct CandidateParameter
{
    IRType* type;
    const char* semantic;
    IROp operation = kIROp_Invalid;
};

/// Collects the ordinary specialized call graph without interpreting source interface syntax.
void collectFunctions(IRFunc* function, HashSet<IRFunc*>& seen, List<IRFunc*>& functions)
{
    if (!seen.add(function))
        return;
    functions.add(function);
    for (auto block : function->getBlocks())
        for (auto inst : block->getOrdinaryInsts())
            if (auto call = as<IRCall>(inst))
                if (auto callee = as<IRFunc>(call->getCallee()))
                    collectFunctions(callee, seen, functions);
}

/// Identifies operations whose meaning depends on a particular candidate invocation. Pure
/// payload/record/builtin helpers remain shared and receive their values in the later batch.
bool needsCandidateSpecialization(IROp op)
{
    switch (op)
    {
    case kIROp_MetalRTIgnoreHit:
    case kIROp_MetalRTAcceptHitAndEndSearch:
    case kIROp_MetalRTReportHit:
    case kIROp_MetalRTGetCurrentDistance:
    case kIROp_MetalRTGetHitKind:
    case kIROp_MetalRTGetObjectSpaceRay:
    case kIROp_PortableRTGetHitAttributes:
        return true;
    default:
        return false;
    }
}

/// Specializes only the graph that depends on the enclosing hit group. Consider two groups
/// sharing one Intersection method but choosing different AnyHit methods: ReportHit must call
/// the selected group's filter. IRCloneEnv remaps those calls while unchanged helpers keep their
/// original identity. NoInline remains meaningful; this does not inline away stage control flow.
IRFunc* specializeCandidateGraph(IRModule* module, IRFunc* root, List<IRFunc*>& clones)
{
    HashSet<IRFunc*> seen;
    List<IRFunc*> functions;
    collectFunctions(root, seen, functions);
    HashSet<IRFunc*> selected;
    selected.add(root);
    for (auto function : functions)
        for (auto block : function->getBlocks())
            for (auto inst : block->getOrdinaryInsts())
                if (needsCandidateSpecialization(inst->getOp()))
                    selected.add(function);
    bool changed;
    do
    {
        changed = false;
        for (auto function : functions)
            for (auto block : function->getBlocks())
                for (auto inst : block->getOrdinaryInsts())
                    if (auto call = as<IRCall>(inst))
                        if (selected.contains(as<IRFunc>(call->getCallee())))
                            changed |= selected.add(function);
    } while (changed);

    IRBuilder builder(module);
    builder.setInsertInto(module->getModuleInst());
    IRCloneEnv environment;
    for (auto function : functions)
        if (selected.contains(function))
        {
            auto clone = cloneInstAndOperands(&environment, &builder, function);
            environment.mapOldValToNew.add(function, clone);
        }
    for (auto function : functions)
        if (selected.contains(function))
        {
            auto clone = cast<IRFunc>(lookUp(&environment, function));
            cloneInstDecorationsAndChildren(&environment, module, function, clone);
            removeLinkageDecorations(clone);
            clones.add(clone);
        }
    return cast<IRFunc>(lookUp(&environment, root));
}

IRStructKey* addField(
    IRBuilder& builder,
    IRStructType* type,
    IRType* fieldType,
    const char* name,
    const char* semantic = nullptr)
{
    auto key = builder.createStructKey();
    builder.addNameHintDecoration(key, UnownedTerminatedStringSlice(name));
    if (semantic)
        builder.addTargetSystemValueDecoration(key, UnownedTerminatedStringSlice(semantic));
    builder.createStructField(type, key, fieldType);
    return key;
}

/// Defines the native return contract. Custom attributes stay in ray_data, not in this result.
CandidateResult createResult(IRModule* module, bool procedural)
{
    IRBuilder builder(module);
    builder.setInsertInto(module->getModuleInst());
    CandidateResult result;
    result.type = builder.createStructType();
    result.hasDistance = procedural;
    builder.addNameHintDecoration(result.type, toSlice("PortableRTCandidateResult"));
    addField(builder, result.type, builder.getBoolType(), "accept", "accept_intersection");
    addField(builder, result.type, builder.getBoolType(), "continueSearch", "continue_search");
    if (procedural)
        addField(builder, result.type, builder.getFloatType(), "distance", "distance");
    return result;
}

IRInst* emitResult(
    IRBuilder& builder,
    const CandidateResult& result,
    IRInst* accept,
    IRInst* continueSearch,
    IRInst* distance = nullptr)
{
    List<IRInst*> fields = {accept, continueSearch};
    if (result.hasDistance)
    {
        SLANG_RELEASE_ASSERT(distance);
        fields.add(distance);
    }
    return builder.emitMakeStruct(result.type, fields.getCount(), fields.getBuffer());
}

IRInst* loadField(IRBuilder& builder, IRInst* state, IRStructKey* key)
{
    return builder.emitLoad(builder.emitFieldAddress(state, key));
}

void storeField(IRBuilder& builder, IRInst* state, IRStructKey* key, IRInst* value)
{
    builder.emitStore(builder.emitFieldAddress(state, key), value);
}

CandidateState createState(IRModule* module, const SchemaHitGroup& group, bool procedural)
{
    IRBuilder builder(module);
    builder.setInsertInto(module->getModuleInst());
    CandidateState state;
    state.type = builder.createStructType();
    builder.addNameHintDecoration(state.type, toSlice("PortableRTCandidateState"));
    state.accepted = addField(builder, state.type, builder.getBoolType(), "accepted");
    state.currentMax = addField(builder, state.type, builder.getFloatType(), "currentMax");
    state.distance = addField(builder, state.type, builder.getFloatType(), "candidateDistance");
    state.hitKind = addField(builder, state.type, builder.getUIntType(), "candidateHitKind");
    if (procedural)
        state.attributes =
            addField(builder, state.type, group.attributesType, "candidateAttributes");
    state.minDistance = addField(builder, state.type, builder.getFloatType(), "minDistance");
    auto float3 = builder.getVectorType(builder.getFloatType(), 3);
    state.origin = addField(builder, state.type, float3, "objectOrigin");
    state.direction = addField(builder, state.type, float3, "objectDirection");
    state.opaque = addField(builder, state.type, builder.getBoolType(), "opaque");
    return state;
}

/// Produces the native IFT inputs once for a primitive kind. Geometry/instance identifiers are
/// also needed by runtime SBT selection even when no source stage explicitly reads them.
List<CandidateParameter> getParameters(
    IRBuilder& builder,
    MetalRTGeometryKind geometry,
    const StageRequirements& requirements,
    UInt tags,
    IRType* instancePathType)
{
    List<CandidateParameter> parameters;
    bool instancing = (tags & UInt(MetalRTTag::Instancing)) != 0;
    bool procedural = geometry == MetalRTGeometryKind::BoundingBox;
    auto floatType = builder.getFloatType();
    auto uintType = builder.getUIntType();
    auto float3 = builder.getVectorType(floatType, 3);
    if (procedural)
    {
        parameters.add({floatType, "min_distance", kIROp_MetalRTGetMinDistance});
        parameters.add({floatType, "max_distance", kIROp_MetalRTGetCurrentDistance});
        parameters.add({builder.getBoolType(), "opaque"});
    }
    else if (
        requirements.has(MetalRTStageRequirement::Distance) ||
        requirements.has(MetalRTStageRequirement::ObjectSpaceRay))
        parameters.add({floatType, "distance", kIROp_MetalRTGetCurrentDistance});
    if (geometry == MetalRTGeometryKind::Triangle)
    {
        if (requirements.has(MetalRTStageRequirement::TriangleBarycentricCoord))
            parameters.add(
                {builder.getVectorType(floatType, 2),
                 "barycentric_coord",
                 kIROp_PortableRTGetTriangleBarycentricCoord});
        if (requirements.has(MetalRTStageRequirement::TriangleFrontFacing) ||
            requirements.has(MetalRTStageRequirement::HitKind))
            parameters.add(
                {builder.getBoolType(), "front_facing", kIROp_MetalRTGetTriangleFrontFacing});
    }
    if (geometry == MetalRTGeometryKind::Curve &&
        requirements.has(MetalRTStageRequirement::CurveParameter))
        parameters.add({floatType, "curve_parameter", kIROp_PortableRTGetCurveParameter});
    parameters.add({uintType, "geometry_id", kIROp_MetalRTGetGeometryIndex});
    if (requirements.has(MetalRTStageRequirement::PrimitiveIndex))
        parameters.add({uintType, "primitive_id", kIROp_MetalRTGetPrimitiveIndex});
    if (instancing)
        parameters.add({instancePathType, "instance_id", kIROp_MetalRTGetInstanceIndex});
    if (instancing && (requirements.has(MetalRTStageRequirement::InstanceID) ||
                       requirements.has(MetalRTStageRequirement::InstanceIDAtLevel)))
        parameters.add({instancePathType, "user_instance_id", kIROp_MetalRTGetInstanceID});
    if (requirements.has(MetalRTStageRequirement::WorldSpaceOrigin))
        parameters.add(
            {float3,
             instancing ? "world_space_origin" : "origin",
             kIROp_MetalRTGetWorldSpaceOrigin});
    if (requirements.has(MetalRTStageRequirement::WorldSpaceDirection))
        parameters.add(
            {float3,
             instancing ? "world_space_direction" : "direction",
             kIROp_MetalRTGetWorldSpaceDirection});
    if (requirements.has(MetalRTStageRequirement::ObjectSpaceRay))
    {
        if (instancing || !requirements.has(MetalRTStageRequirement::WorldSpaceOrigin))
            parameters.add({float3, "origin"});
        if (instancing || !requirements.has(MetalRTStageRequirement::WorldSpaceDirection))
            parameters.add({float3, "direction"});
    }
    auto matrixType = builder.getMatrixType(
        floatType,
        builder.getIntValue(builder.getIntType(), 4),
        builder.getIntValue(builder.getIntType(), 3),
        builder.getIntValue(builder.getIntType(), kMatrixLayoutMode_RowMajor));
    if (instancing && requirements.has(MetalRTStageRequirement::ObjectToWorld))
        parameters.add({matrixType, "object_to_world_transform", kIROp_MetalRTGetObjectToWorld});
    if (instancing && requirements.has(MetalRTStageRequirement::WorldToObject))
        parameters.add({matrixType, "world_to_object_transform", kIROp_MetalRTGetWorldToObject});
    return parameters;
}

/// Calls an unchanged static stage with its zero-storage input token. The token is not a payload
/// copy: its intrinsic properties are supplied through invocation parameters in a later step.
void emitStageCall(IRBuilder& builder, IRFunc* invoke)
{
    auto type = invoke->getParamType(0);
    IRInst* input;
    if (auto pointer = as<IRPtrTypeBase>(type))
    {
        input = builder.emitVar(pointer->getValueType());
        builder.emitStore(input, builder.emitDefaultConstruct(pointer->getValueType()));
    }
    else
        input = builder.emitDefaultConstruct(type);
    builder.emitCallInst(invoke->getResultType(), invoke, 1, &input);
}

/// Replaces candidate reads with loads at the original operation, not snapshots at the adapter.
/// For example, a noinline helper reading IntersectionInput.objectSpaceRay.TMax before and after
/// ReportHit sees the updated accepted distance on its second call.
void lowerMutableInputs(
    IRModule* module,
    const List<IRFunc*>& functions,
    PortableRTInvocationParameter& parameter,
    const CandidateState& state,
    bool intersection)
{
    List<IRInst*> operations;
    for (auto function : functions)
        for (auto block : function->getBlocks())
            for (auto inst : block->getOrdinaryInsts())
                if (inst->getOp() == kIROp_MetalRTGetCurrentDistance ||
                    inst->getOp() == kIROp_MetalRTGetHitKind ||
                    inst->getOp() == kIROp_MetalRTGetObjectSpaceRay ||
                    inst->getOp() == kIROp_PortableRTGetHitAttributes)
                    operations.add(inst);
    for (auto operation : operations)
    {
        IRBuilder builder(module);
        if (operation->getOp() == kIROp_PortableRTGetHitAttributes && !state.attributes)
        {
            // Built-in attribute aggregates are zero-storage views. Their leaf getters receive
            // native barycentrics/curve data later and need no procedural candidate-state pointer.
            builder.setInsertBefore(operation);
            operation->replaceUsesWith(builder.emitDefaultConstruct(operation->getDataType()));
            operation->removeAndDeallocate();
            continue;
        }
        auto context = parameter.get(getParentFunc(operation));
        builder.setInsertBefore(operation);
        IRInst* value = nullptr;
        switch (operation->getOp())
        {
        case kIROp_MetalRTGetCurrentDistance:
            value = loadField(builder, context, intersection ? state.currentMax : state.distance);
            break;
        case kIROp_MetalRTGetHitKind:
            value = loadField(builder, context, state.hitKind);
            break;
        case kIROp_PortableRTGetHitAttributes:
            value = loadField(builder, context, state.attributes);
            break;
        case kIROp_MetalRTGetObjectSpaceRay:
            {
                IRInst* fields[] = {
                    loadField(builder, context, state.origin),
                    loadField(builder, context, state.minDistance),
                    loadField(builder, context, state.direction),
                    loadField(builder, context, intersection ? state.currentMax : state.distance)};
                value = builder.emitMakeStruct(operation->getDataType(), 4, fields);
                break;
            }
        default:
            SLANG_UNEXPECTED("expected candidate input");
        }
        operation->replaceUsesWith(value);
        operation->removeAndDeallocate();
    }
}

struct ExitHandler
{
    enum Kind
    {
        AnyHitDecision,
        BuiltinResult,
        ProceduralResult
    } kind;
    CandidateResult result;
    IRInst* state = nullptr;
    CandidateState stateType;
};

/// Converts only control-dependent candidate functions to Slang's existing throwing IR. This
/// preserves arbitrary helper return types: IgnoreHit in a value-returning helper does not need
/// to invent a value that its caller could accidentally consume. The existing error lowering
/// then turns these edges into ordinary Result calls before invocation threading continues.
bool lowerCandidateExits(
    IRModule* module,
    const List<IRFunc*>& functions,
    const Dictionary<IRFunc*, ExitHandler>& handlers,
    DiagnosticSink* sink)
{
    HashSet<IRFunc*> throwing;
    for (auto function : functions)
        for (auto block : function->getBlocks())
            for (auto inst : block->getOrdinaryInsts())
                if (inst->getOp() == kIROp_MetalRTIgnoreHit ||
                    inst->getOp() == kIROp_MetalRTAcceptHitAndEndSearch ||
                    inst->getOp() == kIROp_Throw)
                    throwing.add(function);
    bool changed;
    do
    {
        changed = false;
        for (auto function : functions)
            if (!handlers.containsKey(function))
                for (auto block : function->getBlocks())
                    for (auto inst : block->getOrdinaryInsts())
                        if (auto call = as<IRCall>(inst))
                            if (throwing.contains(as<IRFunc>(call->getCallee())))
                                changed |= throwing.add(function);
    } while (changed);

    IRBuilder builder(module);
    IRInst* errorType = builder.getUIntType();
    auto attribute = builder.getAttr(kIROp_FuncThrowTypeAttr, 1, &errorType);
    for (auto function : throwing)
    {
        List<IRType*> types;
        for (auto parameter : function->getParams())
            types.add(parameter->getDataType());
        function->setFullType(builder.getFuncType(
            types.getCount(),
            types.getBuffer(),
            function->getResultType(),
            attribute));
    }
    List<IRInst*> exits;
    List<IRCall*> calls;
    for (auto function : functions)
        for (auto block : function->getBlocks())
            for (auto inst : block->getOrdinaryInsts())
            {
                if (inst->getOp() == kIROp_MetalRTIgnoreHit ||
                    inst->getOp() == kIROp_MetalRTAcceptHitAndEndSearch)
                    exits.add(inst);
                if (auto call = as<IRCall>(inst))
                    if (throwing.contains(as<IRFunc>(call->getCallee())))
                        calls.add(call);
            }
    // Convert calls first: an early exit may make a later call unreachable, but retaining its
    // continuation keeps every original SSA use valid until the ordinary CFG simplifier runs.
    for (auto call : calls)
    {
        auto function = getParentFunc(call);
        auto success = builder.createBlock();
        success->insertAfter(call->getParent());
        builder.setInsertInto(success);
        auto result = builder.emitParam(call->getDataType());
        call->replaceUsesWith(result);
        for (auto suffix = call->getNextInst(); suffix;)
        {
            auto next = suffix->getNextInst();
            suffix->insertAtEnd(success);
            suffix = next;
        }
        auto failure = builder.createBlock();
        failure->insertAfter(success);
        builder.setInsertInto(failure);
        auto error = builder.emitParam(builder.getUIntType());
        if (auto handler = handlers.tryGetValue(function))
        {
            if (handler->kind == ExitHandler::AnyHitDecision)
                builder.emitReturn(error);
            else if (handler->kind == ExitHandler::BuiltinResult)
            {
                auto ignored = builder.emitEql(
                    error,
                    builder.getIntValue(builder.getUIntType(), UInt(CandidateExit::Ignore)));
                builder.emitReturn(emitResult(
                    builder,
                    handler->result,
                    builder.emitNot(builder.getBoolType(), ignored),
                    ignored));
            }
            else
                builder.emitReturn(emitResult(
                    builder,
                    handler->result,
                    loadField(builder, handler->state, handler->stateType.accepted),
                    builder.getBoolValue(false),
                    loadField(builder, handler->state, handler->stateType.currentMax)));
        }
        else
            builder.emitThrow(error);
        List<IRInst*> arguments;
        for (UInt i = 0; i < call->getArgCount(); ++i)
            arguments.add(call->getArg(i));
        builder.setInsertBefore(call);
        auto replacement = builder.emitTryCallInst(
            call->getDataType(),
            success,
            failure,
            call->getCallee(),
            arguments.getCount(),
            arguments.getBuffer());
        call->transferDecorationsTo(replacement);
        call->removeAndDeallocate();
    }
    for (auto operation : exits)
    {
        auto exit = operation->getOp() == kIROp_MetalRTIgnoreHit ? CandidateExit::Ignore
                                                                 : CandidateExit::EndSearch;
        auto continuation = builder.createBlock();
        continuation->insertAfter(operation->getParent());
        for (auto suffix = operation->getNextInst(); suffix;)
        {
            auto next = suffix->getNextInst();
            suffix->insertAtEnd(continuation);
            suffix = next;
        }
        builder.setInsertBefore(operation);
        builder.emitThrow(builder.getIntValue(builder.getUIntType(), UInt(exit)));
        operation->removeAndDeallocate();
    }
    lowerErrorHandling(module, sink);
    for (auto function : functions)
        sortBlocksInFunc(function);
    return throwing.getCount() != 0;
}

/// Builds ReportHit's state transition once per hit group. Rejected candidates leave committed
/// attributes alone, but AnyHit's payload writes persist. Accepted candidates update the interval
/// before control resumes in Intersection; an end-search decision propagates out instead.
IRFunc* createReportHit(
    IRModule* module,
    const CandidateState& state,
    const RayData& carrier,
    IRType* attributesType,
    IRFunc* anyHitDecision)
{
    IRBuilder builder(module);
    builder.setInsertInto(module->getModuleInst());
    auto function = builder.createFunc();
    builder.addNameHintDecoration(function, toSlice("portableRTReportHit"));
    IRType* types[] = {
        builder.getPtrType(state.type),
        builder.getPtrType(carrier.type, AddressSpace::ThreadLocal),
        builder.getFloatType(),
        builder.getUIntType(),
        attributesType};
    function->setFullType(builder.getFuncType(5, types, builder.getBoolType()));
    builder.setInsertInto(function);
    auto entry = builder.emitBlock();
    IRInst* context = builder.emitParam(types[0]);
    auto rayData = builder.emitParam(types[1]);
    auto distance = builder.emitParam(types[2]);
    auto hitKind = builder.emitParam(types[3]);
    auto attributes = builder.emitParam(types[4]);
    auto candidate = builder.createBlock();
    auto reject = builder.createBlock();
    auto after = builder.createBlock();
    function->addBlock(candidate);
    function->addBlock(reject);
    function->addBlock(after);
    auto inRange = builder.emitAnd(
        builder.getBoolType(),
        builder.emitGeq(distance, loadField(builder, context, state.minDistance)),
        builder.emitGeq(loadField(builder, context, state.currentMax), distance));
    builder.emitIfElse(inRange, candidate, reject, after);
    builder.setInsertInto(reject);
    builder.emitReturn(builder.getBoolValue(false));
    builder.setInsertInto(after);
    builder.emitUnreachable();
    builder.setInsertInto(candidate);
    storeField(builder, context, state.distance, distance);
    storeField(builder, context, state.hitKind, hitKind);
    storeField(builder, context, state.attributes, attributes);
    auto commit = builder.createBlock();
    function->addBlock(commit);
    IRInst* decision = builder.getIntValue(builder.getUIntType(), UInt(CandidateExit::Accept));
    IRParam* decisionParameter = nullptr;
    if (anyHitDecision)
    {
        auto evaluate = builder.createBlock();
        auto opaque = builder.createBlock();
        function->addBlock(evaluate);
        function->addBlock(opaque);
        builder.emitIfElse(loadField(builder, context, state.opaque), opaque, evaluate, commit);
        builder.setInsertInto(opaque);
        builder.emitBranch(commit, 1, &decision);
        builder.setInsertInto(evaluate);
        IRInst* result = builder.emitCallInst(builder.getUIntType(), anyHitDecision, 1, &context);
        builder.emitBranch(commit, 1, &result);
        builder.setInsertInto(commit);
        decisionParameter = builder.emitParam(builder.getUIntType());
        decision = decisionParameter;
        auto accept = builder.createBlock();
        function->addBlock(accept);
        builder.emitIfElse(
            builder.emitEql(
                decision,
                builder.getIntValue(builder.getUIntType(), UInt(CandidateExit::Ignore))),
            reject,
            accept,
            after);
        builder.setInsertInto(accept);
    }
    else
    {
        builder.emitBranch(commit);
        builder.setInsertInto(commit);
    }
    storeField(builder, context, state.accepted, builder.getBoolValue(true));
    storeField(builder, context, state.currentMax, distance);
    if (auto key = carrier.customAttributeKeys.tryGetValue(attributesType))
        storeField(builder, rayData, *key, attributes);
    if (carrier.customHitKindKey)
        storeField(builder, rayData, carrier.customHitKindKey, hitKind);
    SLANG_RELEASE_ASSERT(carrier.rayFlagsKey);
    auto flags = loadField(builder, rayData, carrier.rayFlagsKey);
    auto firstHit = builder.emitNeq(
        builder.emitBitAnd(
            builder.getUIntType(),
            flags,
            builder.getIntValue(builder.getUIntType(), 4)),
        builder.getIntValue(builder.getUIntType(), 0));
    auto terminate = builder.emitOr(
        builder.getBoolType(),
        firstHit,
        builder.emitEql(
            decision,
            builder.getIntValue(builder.getUIntType(), UInt(CandidateExit::EndSearch))));
    auto terminateBlock = builder.createBlock();
    auto continueBlock = builder.createBlock();
    function->addBlock(terminateBlock);
    function->addBlock(continueBlock);
    builder.emitIfElse(terminate, terminateBlock, continueBlock, after);
    builder.setInsertInto(terminateBlock);
    builder.emitThrow(builder.getIntValue(builder.getUIntType(), UInt(CandidateExit::EndSearch)));
    builder.setInsertInto(continueBlock);
    builder.emitReturn(builder.getBoolValue(true));
    SLANG_UNUSED(entry);
    return function;
}

/// Supplies immutable invocation values at the callback boundary. Only the canonical accessor
/// result type is used as its key; mutable candidate reads were lowered separately to loads.
void collectBindings(
    IRModule* module,
    VisibleAdapter& adapter,
    IRInst* rayData,
    const RayData& carrier,
    IRInst* recordAddress,
    const Dictionary<IROp, IRInst*>& nativeValues)
{
    IRBuilder builder(module);
    builder.setInsertBefore(adapter.function->getFirstBlock()->getFirstOrdinaryInst());
    Dictionary<IROp, IRInst*> values = nativeValues;
    struct CarrierField
    {
        IRStructKey* key;
        IROp operation;
    };
    CarrierField fields[] = {
        {carrier.minDistanceKey, kIROp_MetalRTGetMinDistance},
        {carrier.rayTimeKey, kIROp_MetalRTGetRayTime},
        {carrier.rayFlagsKey, kIROp_MetalRTGetRayFlags},
        {carrier.dispatchRaysIndexKey, kIROp_MetalRTGetDispatchRaysIndex},
        {carrier.dispatchRaysDimensionsKey, kIROp_MetalRTGetDispatchRaysDimensions}};
    for (auto field : fields)
        if (field.key && !values.containsKey(field.operation))
            values.add(field.operation, loadField(builder, rayData, field.key));
    auto header = builder.emitFieldAddress(rayData, carrier.headerKey);
    adapter.records = loadField(builder, header, carrier.header->recordsKey);
    HashSet<IRFunc*> seen;
    List<IRFunc*> functions;
    collectFunctions(adapter.function, seen, functions);
    HashSet<KeyValuePair<IROp, IRType*>> queries;
    for (auto function : functions)
        for (auto block : function->getBlocks())
            for (auto operation : block->getOrdinaryInsts())
            {
                auto op = operation->getOp();
                auto type = operation->getDataType();
                IRInst* value = nullptr;
                if (op == kIROp_PortableRTGetPayload)
                    value = getRayPayloadReference(
                        builder,
                        carrier,
                        rayData,
                        cast<IRPtrTypeBase>(type)->getValueType());
                else if (auto found = values.tryGetValue(op))
                    value = *found;
                else if (op == kIROp_PortableRTGetRecord)
                {
                    if (isEmptyType(type))
                        value = as<IRVoidType>(type) ? builder.getVoidValue()
                                                     : builder.emitDefaultConstruct(type);
                    else
                        value = builder.emitLoad(builder.emitBitCast(
                            builder.getPtrType(type, AddressSpace::UserPointer),
                            recordAddress));
                }
                if (value && queries.add({op, type}))
                    adapter.bindings.add({op, type, value});
            }
}

struct GroupCallback
{
    VisibleAdapter adapter;
    Index functionIndex;
    const RayData* carrier;
    IRInst* rayData;
    IRInst* record;
    Dictionary<IROp, IRInst*> values;
};

/// Creates a typed group callback. The native dispatcher copies ray_data into thread storage
/// before calling it, so ordinary payload references and helper parameters have one address space.
GroupCallback createGroupCallback(
    IRModule* module,
    const SchemaHitGroup& group,
    Index functionIndex,
    const RayData& carrier,
    MetalRTGeometryKind geometry,
    const List<CandidateParameter>& parameters,
    IRType* instancePathType,
    const CandidateResult& resultType,
    List<IRFunc*>& controlFunctions,
    Dictionary<IRFunc*, ExitHandler>& handlers)
{
    bool procedural = geometry == MetalRTGeometryKind::BoundingBox;
    IRBuilder builder(module);
    builder.setInsertInto(module->getModuleInst());
    GroupCallback result;
    result.functionIndex = functionIndex;
    result.carrier = &carrier;
    auto callback = builder.createFunc();
    result.adapter.function = callback;
    builder.addNameHintDecoration(callback, toSlice("portableRTCandidateGroup"));
    List<IRType*> types;
    for (auto parameter : parameters)
        types.add(parameter.type);
    types.add(builder.getPtrType(builder.getUInt8Type(), AddressSpace::Global));
    types.add(builder.getPtrType(carrier.type, AddressSpace::ThreadLocal));
    callback->setFullType(builder.getFuncType(types, resultType.type));
    builder.setInsertInto(callback);
    builder.emitBlock();
    Dictionary<String, IRInst*> semantics;
    for (auto parameter : parameters)
    {
        auto value = builder.emitParam(parameter.type);
        builder.addNameHintDecoration(value, UnownedTerminatedStringSlice(parameter.semantic));
        semantics.add(String(parameter.semantic), value);
        if (parameter.operation != kIROp_Invalid)
            result.values.add(parameter.operation, value);
    }
    result.record = builder.emitParam(types[types.getCount() - 2]);
    result.rayData = builder.emitParam(types.getLast());
    if (auto path = result.values.tryGetValue(kIROp_MetalRTGetInstanceIndex))
    {
        IRInst* count;
        if (instancePathType->getOp() == kIROp_MetalRTInstancePath)
        {
            auto getCount = createInstancePathOperation(module, InstancePathOperation::Count);
            count = builder.emitCallInst(builder.getUIntType(), getCount, 1, path);
        }
        else
            count = builder.getIntValue(builder.getUIntType(), 1);
        result.values.add(kIROp_MetalRTGetInstanceCount, count);
    }
    else
    {
        // Primitive-AS traversal has no instance path, and its object/world transforms are
        // identity. Keep the empty native view available for guarded indexed reads in a helper.
        result.values.add(
            kIROp_MetalRTGetInstanceCount,
            builder.getIntValue(builder.getUIntType(), 0));
        SLANG_RELEASE_ASSERT(instancePathType->getOp() == kIROp_MetalRTInstancePath);
        auto empty = createInstancePathOperation(module, InstancePathOperation::Empty);
        auto emptyPath = builder.emitCallInst(instancePathType, empty, 0, nullptr);
        result.values.add(kIROp_MetalRTGetInstanceIndex, emptyPath);
        result.values.add(kIROp_MetalRTGetInstanceID, emptyPath);
        auto floatType = builder.getFloatType();
        auto matrixType = builder.getMatrixType(
            floatType,
            builder.getIntValue(builder.getIntType(), 4),
            builder.getIntValue(builder.getIntType(), 3),
            builder.getIntValue(builder.getIntType(), kMatrixLayoutMode_RowMajor));
        auto zero = builder.getFloatValue(floatType, 0);
        auto one = builder.getFloatValue(floatType, 1);
        IRInst* elements[] = {one, zero, zero, zero, one, zero, zero, zero, one, zero, zero, zero};
        auto identity = builder.emitMakeMatrix(matrixType, 12, elements);
        result.values.add(kIROp_MetalRTGetObjectToWorld, identity);
        result.values.add(kIROp_MetalRTGetWorldToObject, identity);
    }
    auto state = createState(module, group, procedural);
    auto context = builder.emitVar(state.type);
    builder.emitStore(context, builder.emitDefaultConstruct(state.type));
    if (auto distance = result.values.tryGetValue(kIROp_MetalRTGetCurrentDistance))
    {
        storeField(builder, context, state.currentMax, *distance);
        storeField(builder, context, state.distance, *distance);
    }
    if (auto minimum = result.values.tryGetValue(kIROp_MetalRTGetMinDistance))
        storeField(builder, context, state.minDistance, *minimum);
    else if (carrier.minDistanceKey)
        storeField(
            builder,
            context,
            state.minDistance,
            loadField(builder, result.rayData, carrier.minDistanceKey));
    if (auto origin = semantics.tryGetValue(String("origin")))
        storeField(builder, context, state.origin, *origin);
    if (auto direction = semantics.tryGetValue(String("direction")))
        storeField(builder, context, state.direction, *direction);
    if (auto opaque = semantics.tryGetValue(String("opaque")))
        storeField(builder, context, state.opaque, *opaque);
    if (auto frontFacing = result.values.tryGetValue(kIROp_MetalRTGetTriangleFrontFacing))
    {
        IRInst* fields[] = {
            *frontFacing,
            builder.getIntValue(builder.getUIntType(), 254),
            builder.getIntValue(builder.getUIntType(), 255)};
        storeField(
            builder,
            context,
            state.hitKind,
            builder.emitIntrinsicInst(builder.getUIntType(), kIROp_Select, 3, fields));
    }
    PortableRTInvocationParameter stateParameter(module, context->getDataType(), "candidateState");
    stateParameter.bind(callback, context);
    PortableRTInvocationParameter carrierParameter(
        module,
        result.rayData->getDataType(),
        "rayData");
    carrierParameter.bind(callback, result.rayData);
    controlFunctions.add(callback);
    List<IRFunc*> stageClones;
    if (!procedural)
    {
        if (group.anyHit.isPresent)
        {
            auto invoke = specializeCandidateGraph(module, group.anyHit.invoke, stageClones);
            emitStageCall(builder, invoke);
        }
        builder.emitReturn(emitResult(
            builder,
            resultType,
            builder.getBoolValue(true),
            builder.getBoolValue(true)));
        lowerMutableInputs(module, stageClones, stateParameter, state, false);
        handlers.add(callback, {ExitHandler::BuiltinResult, resultType, context, state});
    }
    else if (!group.intersection.isPresent)
        builder.emitReturn(emitResult(
            builder,
            resultType,
            builder.getBoolValue(false),
            builder.getBoolValue(true),
            builder.getFloatValue(builder.getFloatType(), 0)));
    else
    {
        IRFunc* anyHitDecision = nullptr;
        if (group.anyHit.isPresent)
        {
            List<IRFunc*> anyHitClones;
            auto invoke = specializeCandidateGraph(module, group.anyHit.invoke, anyHitClones);
            IRBuilder helperBuilder(module);
            helperBuilder.setInsertInto(module->getModuleInst());
            anyHitDecision = helperBuilder.createFunc();
            IRType* stateType = context->getDataType();
            anyHitDecision->setFullType(
                helperBuilder.getFuncType(1, &stateType, helperBuilder.getUIntType()));
            helperBuilder.addNameHintDecoration(
                anyHitDecision,
                toSlice("portableRTAnyHitDecision"));
            helperBuilder.setInsertInto(anyHitDecision);
            helperBuilder.emitBlock();
            auto stateValue = helperBuilder.emitParam(stateType);
            emitStageCall(helperBuilder, invoke);
            helperBuilder.emitReturn(helperBuilder.getIntValue(
                helperBuilder.getUIntType(),
                UInt(CandidateExit::Accept)));
            PortableRTInvocationParameter anyHitState(module, stateType, "candidateState");
            anyHitState.bind(anyHitDecision, stateValue);
            lowerMutableInputs(module, anyHitClones, anyHitState, state, false);
            controlFunctions.addRange(anyHitClones);
            controlFunctions.add(anyHitDecision);
            handlers.add(anyHitDecision, {ExitHandler::AnyHitDecision, {}, nullptr, {}});
        }
        auto report = createReportHit(module, state, carrier, group.attributesType, anyHitDecision);
        controlFunctions.add(report);
        auto invoke = specializeCandidateGraph(module, group.intersection.invoke, stageClones);
        emitStageCall(builder, invoke);
        builder.emitReturn(emitResult(
            builder,
            resultType,
            loadField(builder, context, state.accepted),
            builder.getBoolValue(true),
            loadField(builder, context, state.currentMax)));
        lowerMutableInputs(module, stageClones, stateParameter, state, true);
        List<IRInst*> reports;
        for (auto function : stageClones)
            for (auto block : function->getBlocks())
                for (auto operation : block->getOrdinaryInsts())
                    if (operation->getOp() == kIROp_MetalRTReportHit)
                        reports.add(operation);
        for (auto operation : reports)
        {
            auto function = getParentFunc(operation);
            auto stateValue = stateParameter.get(function);
            auto rayDataValue = carrierParameter.get(function);
            builder.setInsertBefore(operation);
            IRInst* arguments[] = {
                stateValue,
                rayDataValue,
                operation->getOperand(0),
                operation->getOperand(1),
                operation->getOperand(2)};
            operation->replaceUsesWith(
                builder.emitCallInst(builder.getBoolType(), report, 5, arguments));
            operation->removeAndDeallocate();
        }
        handlers.add(callback, {ExitHandler::ProceduralResult, resultType, context, state});
    }
    controlFunctions.addRange(stageClones);
    return result;
}

/// Looks up the runtime hit record using the schema's shared header. Function indices name code,
/// not physical slots, so the host can repeat a group with different record data any number of
/// times.
IRInst* getRecordAddress(
    IRBuilder& builder,
    IRInst* header,
    const RayDataHeader& headerType,
    IRInst* geometry,
    IRInst* instance,
    IRFunc* contribution,
    IRIntegerValue stride)
{
    auto uintType = builder.getUIntType();
    auto records = loadField(builder, header, headerType.recordsKey);
    auto offset = loadField(builder, header, headerType.sbtOffsetKey);
    auto geometryStride = loadField(builder, header, headerType.sbtStrideKey);
    auto index =
        builder.emitAdd(uintType, offset, builder.emitMul(uintType, geometry, geometryStride));
    if (instance)
    {
        SLANG_RELEASE_ASSERT(contribution);
        IRInst* arguments[] = {records, instance};
        index = builder.emitAdd(
            uintType,
            index,
            builder.emitCallInst(uintType, contribution, 2, arguments));
    }
    auto sectionOffset = builder.emitLoad(builder.emitGetOffsetPtr(
        records,
        builder.getIntValue(uintType, UInt(MetalRTRecordSection::Hit))));
    auto byteOffset = builder.emitAdd(
        uintType,
        sectionOffset,
        builder.emitMul(uintType, index, builder.getIntValue(uintType, stride)));
    auto bytes = builder.emitBitCast(
        builder.getPtrType(builder.getUInt8Type(), AddressSpace::Global),
        records);
    return builder.emitGetOffsetPtr(bytes, byteOffset);
}

/// Creates one native IFT entry per primitive kind, then dispatches through the runtime record's
/// function index. The payload carrier becomes known only inside a selected arm. Copying through
/// that concrete type respects Metal's distinct ray_data and thread address spaces.
IRFunc* createDispatcher(
    IRModule* module,
    const List<GroupCallback>& callbacks,
    const List<CandidateParameter>& parameters,
    const CandidateResult& resultType,
    MetalRTGeometryKind geometry,
    const RayDataHeader& header,
    UInt tags,
    IRIntegerValue maxLevels,
    IRIntegerValue hitStride,
    IRFunc* contribution,
    UnownedStringSlice name)
{
    IRBuilder builder(module);
    builder.setInsertInto(module->getModuleInst());
    auto function = builder.createFunc();
    builder.addNameHintDecoration(function, name);
    builder.addExportDecoration(function, name);
    builder.addExternCppDecoration(function, name);
    builder.addKeepAliveDecoration(function);
    IRInst* decoration[] = {
        builder.getIntValue(builder.getIntType(), UInt(geometry)),
        builder.getIntValue(builder.getIntType(), tags),
        builder.getIntValue(builder.getIntType(), maxLevels)};
    builder.addDecoration(function, kIROp_MetalIntersectionFunctionDecoration, decoration, 3);
    List<IRType*> types;
    for (auto parameter : parameters)
        types.add(parameter.type);
    types.add(builder.getPtrType(header.type, AddressSpace::MetalRayData));
    function->setFullType(builder.getFuncType(types, resultType.type));
    builder.setInsertInto(function);
    auto entry = builder.emitBlock();
    List<IRInst*> values;
    IRInst* geometryIndex = nullptr;
    IRInst* instanceIndex = nullptr;
    for (auto parameter : parameters)
    {
        auto value = builder.emitParam(parameter.type);
        builder.addNameHintDecoration(value, UnownedTerminatedStringSlice(parameter.semantic));
        builder.addTargetSystemValueDecoration(
            value,
            UnownedTerminatedStringSlice(parameter.semantic));
        values.add(value);
        if (parameter.operation == kIROp_MetalRTGetGeometryIndex)
            geometryIndex = value;
        else if (parameter.operation == kIROp_MetalRTGetInstanceIndex)
            instanceIndex = value;
    }
    auto rayData = builder.emitParam(types.getLast());
    builder.addNameHintDecoration(rayData, toSlice("rayData"));
    builder.addTargetSystemValueDecoration(rayData, toSlice("payload"));
    auto record = getRecordAddress(
        builder,
        rayData,
        header,
        geometryIndex,
        instanceIndex,
        contribution,
        hitStride);
    auto uintType = builder.getUIntType();
    auto functionIndex = builder.emitLoad(
        builder.emitBitCast(builder.getPtrType(uintType, AddressSpace::Global), record));
    auto recordData =
        builder.emitGetOffsetPtr(record, builder.getIntValue(uintType, kMetalRTRecordHeaderSize));
    auto defaultBlock = builder.createBlock();
    auto breakBlock = builder.createBlock();
    function->addBlock(defaultBlock);
    function->addBlock(breakBlock);
    List<IRInst*> cases;
    for (const auto& callback : callbacks)
    {
        auto block = builder.createBlock();
        function->addBlock(block);
        cases.add(builder.getIntValue(uintType, callback.functionIndex));
        cases.add(block);
        builder.setInsertInto(block);
        auto type = callback.carrier->type;
        auto source =
            builder.emitBitCast(builder.getPtrType(type, AddressSpace::MetalRayData), rayData);
        auto local = builder.emitVar(type);
        builder.emitStore(local, builder.emitLoad(source));
        List<IRInst*> arguments = values;
        arguments.add(recordData);
        arguments.add(local);
        auto result = builder.emitCallInst(resultType.type, callback.adapter.function, arguments);
        builder.emitStore(source, builder.emitLoad(local));
        builder.emitReturn(result);
    }
    if (geometry != MetalRTGeometryKind::BoundingBox)
    {
        auto empty = builder.createBlock();
        function->addBlock(empty);
        cases.add(builder.getIntValue(uintType, 0xffffffffu));
        cases.add(empty);
        builder.setInsertInto(empty);
        builder.emitReturn(emitResult(
            builder,
            resultType,
            builder.getBoolValue(true),
            builder.getBoolValue(true)));
    }
    builder.setInsertInto(defaultBlock);
    builder.emitReturn(emitResult(
        builder,
        resultType,
        builder.getBoolValue(false),
        builder.getBoolValue(true),
        builder.getFloatValue(builder.getFloatType(), 0)));
    builder.setInsertInto(breakBlock);
    builder.emitUnreachable();
    builder.setInsertInto(entry);
    builder
        .emitSwitch(functionIndex, breakBlock, defaultBlock, cases.getCount(), cases.getBuffer());
    return function;
}

} // namespace

CandidateAdapters createCandidateAdapters(
    IRModule* module,
    const Schema& schema,
    const Dictionary<IRType*, RayData>& carriers,
    const RayDataHeader& header,
    const StageRequirements& requirements,
    UInt tags,
    IRIntegerValue maxLevels,
    IRIntegerValue hitStride,
    IRFunc* instanceContribution,
    IRType* instancePathParamType,
    UnownedStringSlice symbolPrefix,
    DiagnosticSink* sink)
{
    CandidateAdapters result;
    IRBuilder builder(module);
    List<IRFunc*> controlFunctions;
    Dictionary<IRFunc*, ExitHandler> handlers;
    List<GroupCallback> allCallbacks;
    struct Geometry
    {
        MetalRTGeometryKind kind;
        MetalRTCandidateKind slot;
        KnownBuiltinDeclName role;
        const char* suffix;
    };
    Geometry geometries[] = {
        {MetalRTGeometryKind::Triangle,
         MetalRTCandidateKind::Triangle,
         KnownBuiltinDeclName::RayTracingTrianglePrimitive,
         "_triangle"},
        {MetalRTGeometryKind::BoundingBox,
         MetalRTCandidateKind::BoundingBox,
         KnownBuiltinDeclName::RayTracingBoundingBoxPrimitive,
         "_bounding_box"},
        {MetalRTGeometryKind::Curve,
         MetalRTCandidateKind::Curve,
         KnownBuiltinDeclName::RayTracingCurvePrimitive,
         "_curve"}};
    for (auto geometry : geometries)
    {
        bool needsFunction = false;
        for (const auto& group : schema.hitGroups)
            if (getBuiltinRole(group.primitiveType) == geometry.role &&
                (group.anyHit.isPresent || group.intersection.isPresent))
                needsFunction = true;
        if (!needsFunction)
            continue;
        auto nativeResult = createResult(module, geometry.kind == MetalRTGeometryKind::BoundingBox);
        auto parameters =
            getParameters(builder, geometry.kind, requirements, tags, instancePathParamType);
        List<GroupCallback> callbacks;
        for (Index i = 0; i < schema.hitGroups.getCount(); ++i)
        {
            auto& group = schema.hitGroups[i];
            if (getBuiltinRole(group.primitiveType) != geometry.role)
                continue;
            callbacks.add(createGroupCallback(
                module,
                group,
                i,
                carriers.getValue(getRayPayloadStorageType(builder, group.payloadType)),
                geometry.kind,
                parameters,
                instancePathParamType,
                nativeResult,
                controlFunctions,
                handlers));
        }
        StringBuilder name;
        name << symbolPrefix << geometry.suffix;
        result.intersectionFunctions.add(
            geometry.slot,
            createDispatcher(
                module,
                callbacks,
                parameters,
                nativeResult,
                geometry.kind,
                header,
                tags,
                maxLevels,
                hitStride,
                instanceContribution,
                name.getUnownedSlice()));
        allCallbacks.addRange(callbacks);
    }
    result.producesResultTypes = lowerCandidateExits(module, controlFunctions, handlers, sink);
    for (auto& callback : allCallbacks)
    {
        collectBindings(
            module,
            callback.adapter,
            callback.rayData,
            *callback.carrier,
            callback.record,
            callback.values);
        result.invocationRoots.add(callback.adapter);
    }
    return result;
}

} // namespace Metal
} // namespace PortableRT
} // namespace Slang
