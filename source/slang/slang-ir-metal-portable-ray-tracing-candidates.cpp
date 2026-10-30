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

/// This layout is a compiler ABI shared by independently compiled stages. Schema-specific
/// payload and attribute types are represented only by pointers into the glue's thread storage.
/// In particular, these pointers are never stored in Metal's ray_data address space.
struct CandidateState
{
    IRStructType* type = nullptr;
    IRStructKey* payload = nullptr;
    IRStructKey* record = nullptr;
    IRStructKey* accepted = nullptr;
    IRStructKey* currentMax = nullptr;
    IRStructKey* distance = nullptr;
    IRStructKey* hitKind = nullptr;
    IRStructKey* attributes = nullptr;
    IRStructKey* committedAttributes = nullptr;
    IRStructKey* committedHitKind = nullptr;
    IRStructKey* minDistance = nullptr;
    IRStructKey* origin = nullptr;
    IRStructKey* direction = nullptr;
    IRStructKey* opaque = nullptr;
    IRStructKey* instanceIndices = nullptr;
    IRStructKey* instanceIDs = nullptr;
    IRStructKey* instanceCount = nullptr;
    Dictionary<IROp, IRStructKey*> inputs;
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
    case kIROp_MetalRTGetInstanceIndex:
    case kIROp_MetalRTGetInstanceID:
        return true;
    default:
        return false;
    }
}

/// Clones only functions whose candidate control or mutable reads need stage-specific lowering.
/// Consider a helper that reads objectSpaceRay.TMax: Intersection observes the shrinking interval,
/// while AnyHit observes the reported candidate distance. The ordinary source helper retains its
/// identity for other callers; each selected stage owns one lowered graph, independent of schemas.
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

CandidateState createState(IRModule* module)
{
    IRBuilder builder(module);
    builder.setInsertInto(module->getModuleInst());
    CandidateState state;
    state.type = builder.createStructType();
    builder.addNameHintDecoration(state.type, toSlice("PortableRTCandidateState"));
    auto pointer = builder.getPtrType(builder.getVoidType(), AddressSpace::ThreadLocal);
    auto uintPointer = builder.getPtrType(builder.getUIntType(), AddressSpace::ThreadLocal);
    auto floatType = builder.getFloatType();
    auto uintType = builder.getUIntType();
    auto float3 = builder.getVectorType(floatType, 3);
    auto uint3 = builder.getVectorType(uintType, 3);
    state.payload = addField(builder, state.type, pointer, "payload");
    state.record = addField(
        builder,
        state.type,
        builder.getPtrType(builder.getUInt8Type(), AddressSpace::Global),
        "record");
    state.accepted = addField(builder, state.type, builder.getBoolType(), "accepted");
    state.currentMax = addField(builder, state.type, floatType, "currentMax");
    state.distance = addField(builder, state.type, floatType, "candidateDistance");
    state.hitKind = addField(builder, state.type, uintType, "candidateHitKind");
    state.attributes = addField(builder, state.type, pointer, "candidateAttributes");
    state.committedAttributes = addField(builder, state.type, pointer, "committedAttributes");
    state.committedHitKind = addField(builder, state.type, uintPointer, "committedHitKind");
    state.minDistance = addField(builder, state.type, floatType, "minDistance");
    state.origin = addField(builder, state.type, float3, "objectOrigin");
    state.direction = addField(builder, state.type, float3, "objectDirection");
    state.opaque = addField(builder, state.type, builder.getBoolType(), "opaque");
    state.instanceIndices = addField(builder, state.type, uintPointer, "instanceIndices");
    state.instanceIDs = addField(builder, state.type, uintPointer, "instanceIDs");
    state.instanceCount = addField(builder, state.type, uintType, "instanceCount");
    state.inputs.add(kIROp_MetalRTGetMinDistance, state.minDistance);
    state.inputs.add(kIROp_MetalRTGetInstanceCount, state.instanceCount);
    // Keep one fixed field order. A body's choice of getters cannot change this contract.
    struct Input
    {
        IROp op;
        IRType* type;
        const char* name;
    };
    auto matrix = builder.getMatrixType(
        floatType,
        builder.getIntValue(builder.getIntType(), 4),
        builder.getIntValue(builder.getIntType(), 3),
        builder.getIntValue(builder.getIntType(), kMatrixLayoutMode_RowMajor));
    Input inputs[] = {
        {kIROp_MetalRTGetRayTime, floatType, "rayTime"},
        {kIROp_MetalRTGetRayFlags, uintType, "rayFlags"},
        {kIROp_MetalRTGetDispatchRaysIndex, uint3, "dispatchRaysIndex"},
        {kIROp_MetalRTGetDispatchRaysDimensions, uint3, "dispatchRaysDimensions"},
        {kIROp_PortableRTGetTriangleBarycentricCoord,
         builder.getVectorType(floatType, 2),
         "barycentrics"},
        {kIROp_MetalRTGetTriangleFrontFacing, builder.getBoolType(), "frontFacing"},
        {kIROp_PortableRTGetCurveParameter, floatType, "curveParameter"},
        {kIROp_MetalRTGetWorldSpaceOrigin, float3, "worldOrigin"},
        {kIROp_MetalRTGetWorldSpaceDirection, float3, "worldDirection"},
        {kIROp_MetalRTGetPrimitiveIndex, uintType, "primitiveIndex"},
        {kIROp_MetalRTGetGeometryIndex, uintType, "geometryIndex"},
        {kIROp_MetalRTGetObjectToWorld, matrix, "objectToWorld"},
        {kIROp_MetalRTGetWorldToObject, matrix, "worldToObject"},
    };
    for (auto input : inputs)
        state.inputs.add(input.op, addField(builder, state.type, input.type, input.name));
    return state;
}

List<CandidateParameter> getParameters(
    IRBuilder& builder,
    MetalRTGeometryKind geometry,
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
    else
        parameters.add({floatType, "distance", kIROp_MetalRTGetCurrentDistance});
    if (geometry == MetalRTGeometryKind::Triangle)
    {
        parameters.add(
            {builder.getVectorType(floatType, 2),
             "barycentric_coord",
             kIROp_PortableRTGetTriangleBarycentricCoord});
        parameters.add(
            {builder.getBoolType(), "front_facing", kIROp_MetalRTGetTriangleFrontFacing});
    }
    if (geometry == MetalRTGeometryKind::Curve)
        parameters.add({floatType, "curve_parameter", kIROp_PortableRTGetCurveParameter});
    parameters.add({uintType, "geometry_id", kIROp_MetalRTGetGeometryIndex});
    parameters.add({uintType, "primitive_id", kIROp_MetalRTGetPrimitiveIndex});
    if (instancing)
        parameters.add({instancePathType, "instance_id", kIROp_MetalRTGetInstanceIndex});
    if (instancing)
        parameters.add({instancePathType, "user_instance_id", kIROp_MetalRTGetInstanceID});
    parameters.add(
        {float3, instancing ? "world_space_origin" : "origin", kIROp_MetalRTGetWorldSpaceOrigin});
    parameters.add(
        {float3,
         instancing ? "world_space_direction" : "direction",
         kIROp_MetalRTGetWorldSpaceDirection});
    if (instancing)
    {
        parameters.add({float3, "origin"});
        parameters.add({float3, "direction"});
    }
    auto matrixType = builder.getMatrixType(
        floatType,
        builder.getIntValue(builder.getIntType(), 4),
        builder.getIntValue(builder.getIntType(), 3),
        builder.getIntValue(builder.getIntType(), kMatrixLayoutMode_RowMajor));
    if (instancing)
        parameters.add({matrixType, "object_to_world_transform", kIROp_MetalRTGetObjectToWorld});
    if (instancing)
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
                    inst->getOp() == kIROp_PortableRTGetHitAttributes ||
                    inst->getOp() == kIROp_MetalRTGetInstanceIndex ||
                    inst->getOp() == kIROp_MetalRTGetInstanceID)
                    operations.add(inst);
    for (auto operation : operations)
    {
        IRBuilder builder(module);
        if (operation->getOp() == kIROp_PortableRTGetHitAttributes &&
            isBuiltinHitAttributeView(operation))
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
            if (isEmptyType(operation->getDataType()))
                value = builder.emitDefaultConstruct(operation->getDataType());
            else
                value = builder.emitLoad(builder.emitBitCast(
                    builder.getPtrType(operation->getDataType(), AddressSpace::ThreadLocal),
                    loadField(builder, context, state.attributes)));
            break;
        case kIROp_MetalRTGetInstanceIndex:
        case kIROp_MetalRTGetInstanceID:
            {
                auto key = operation->getOp() == kIROp_MetalRTGetInstanceIndex
                               ? state.instanceIndices
                               : state.instanceIDs;
                auto path = loadField(builder, context, key);
                value = builder.emitLoad(builder.emitGetOffsetPtr(path, operation->getOperand(0)));
                break;
            }
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

/// Converts only control-dependent candidate functions to Slang's existing throwing IR. This
/// preserves arbitrary helper return types: IgnoreHit in a value-returning helper does not need
/// to invent a value that its caller could accidentally consume. The existing error lowering
/// then turns these edges into ordinary Result calls before invocation threading continues.
bool lowerCandidateExits(
    IRModule* module,
    const List<IRFunc*>& functions,
    const HashSet<IRFunc*>& handlers,
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
            if (!handlers.contains(function))
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
        if (handlers.contains(function))
            builder.emitReturn(error);
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

/// Builds ReportHit for one selected intersection shader. Consider two groups sharing this
/// shader but using different any-hit filters: the caller supplies the table and index, so the
/// same compiled ReportHit calls either filter. Rejection preserves payload writes but leaves
/// committed attributes and the accepted distance unchanged.
IRFunc* createReportHit(IRModule* module, const CandidateState& state, IRType* attributesType)
{
    IRBuilder builder(module);
    builder.setInsertInto(module->getModuleInst());
    auto function = builder.createFunc();
    builder.addNameHintDecoration(function, toSlice("portableRTReportHit"));
    auto intersectionSignature = getIntersectionFunctionSignature(builder);
    IRType* types[] = {
        builder.getPtrType(state.type, AddressSpace::ThreadLocal),
        intersectionSignature->getParamType(1),
        intersectionSignature->getParamType(2),
        builder.getUIntType(),
        builder.getFloatType(),
        builder.getUIntType(),
        attributesType};
    function->setFullType(builder.getFuncType(SLANG_COUNT_OF(types), types, builder.getBoolType()));
    builder.setInsertInto(function);
    builder.emitBlock();
    auto context = builder.emitParam(types[0]);
    auto globals = builder.emitParam(types[1]);
    auto anyHitTable = builder.emitParam(types[2]);
    auto anyHitIndex = builder.emitParam(types[3]);
    auto distance = builder.emitParam(types[4]);
    auto hitKind = builder.emitParam(types[5]);
    auto attributes = builder.emitParam(types[6]);
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
    if (!isEmptyType(attributesType))
        builder.emitStore(
            builder.emitBitCast(
                builder.getPtrType(attributesType, AddressSpace::ThreadLocal),
                loadField(builder, context, state.attributes)),
            attributes);
    auto evaluate = builder.createBlock();
    auto skip = builder.createBlock();
    auto commit = builder.createBlock();
    auto accept = builder.createBlock();
    function->addBlock(evaluate);
    function->addBlock(skip);
    function->addBlock(commit);
    function->addBlock(accept);
    auto absent =
        builder.emitEql(anyHitIndex, builder.getIntValue(builder.getUIntType(), 0xffffffffu));
    auto bypass =
        builder.emitOr(builder.getBoolType(), absent, loadField(builder, context, state.opaque));
    builder.emitIfElse(bypass, skip, evaluate, commit);
    builder.setInsertInto(skip);
    IRInst* accepted = builder.getIntValue(builder.getUIntType(), UInt(CandidateExit::Accept));
    builder.emitBranch(commit, 1, &accepted);
    builder.setInsertInto(evaluate);
    IRInst* callOperands[] = {
        anyHitTable,
        anyHitIndex,
        builder.emitBitCast(getAnyHitFunctionSignature(builder)->getParamType(0), context),
        globals};
    IRInst* result = builder.emitIntrinsicInst(
        builder.getUIntType(),
        kIROp_MetalRTCandidateDispatch,
        SLANG_COUNT_OF(callOperands),
        callOperands);
    builder.emitBranch(commit, 1, &result);
    builder.setInsertInto(commit);
    auto decision = builder.emitParam(builder.getUIntType());
    builder.emitIfElse(
        builder.emitEql(
            decision,
            builder.getIntValue(builder.getUIntType(), UInt(CandidateExit::Ignore))),
        reject,
        accept,
        after);
    builder.setInsertInto(accept);
    storeField(builder, context, state.accepted, builder.getBoolValue(true));
    storeField(builder, context, state.currentMax, distance);
    if (!isEmptyType(attributesType))
        builder.emitStore(
            builder.emitBitCast(
                builder.getPtrType(attributesType, AddressSpace::ThreadLocal),
                loadField(builder, context, state.committedAttributes)),
            attributes);
    builder.emitStore(loadField(builder, context, state.committedHitKind), hitKind);
    auto flags = loadField(builder, context, state.inputs.getValue(kIROp_MetalRTGetRayFlags));
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
    return function;
}

/// Supplies immutable values at the independently compiled provider boundary. Mutable getters
/// are lowered to state loads at their original use so ReportHit can change the current interval.
void collectBindings(
    IRModule* module,
    VisibleAdapter& adapter,
    IRInst* context,
    const CandidateState& state)
{
    IRBuilder builder(module);
    builder.setInsertAfter(context);
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
                if (!queries.add({op, type}))
                    continue;
                IRInst* value = nullptr;
                if (op == kIROp_PortableRTGetPayload)
                {
                    value = getStageDataReference(
                        builder,
                        loadField(builder, context, state.payload),
                        cast<IRPtrTypeBase>(type)->getValueType());
                }
                else if (auto key = state.inputs.tryGetValue(op))
                    value = loadField(builder, context, *key);
                else if (op == kIROp_PortableRTGetRecord)
                {
                    if (isEmptyType(type))
                        value = as<IRVoidType>(type) ? builder.getVoidValue()
                                                     : builder.emitDefaultConstruct(type);
                    else
                        value = builder.emitLoad(builder.emitBitCast(
                            builder.getPtrType(type, AddressSpace::UserPointer),
                            loadField(builder, context, state.record)));
                }
                if (value)
                    adapter.bindings.add({op, type, value});
            }
}

struct GroupCallback
{
    IRFunc* function = nullptr;
    Index functionIndex;
    const RayData* carrier;
};

/// Copies a native instance path into the fixed provider ABI. The arrays remain alive throughout
/// the callback, including nested any-hit calls, and only initialized path elements are exposed.
void initializeInstancePath(
    IRBuilder& builder,
    IRModule* module,
    IRInst* context,
    const CandidateState& state,
    IRInst* indices,
    IRInst* ids,
    IRIntegerValue maxLevels)
{
    SLANG_RELEASE_ASSERT(indices && ids);
    auto uintType = builder.getUIntType();
    bool indexed = indices->getDataType()->getOp() == kIROp_MetalRTInstancePath;
    // maxLevels belongs to the explicit multilevel contract. Ordinary instancing uses scalar
    // builtins and has one path element even when no max_levels tag is present.
    SLANG_RELEASE_ASSERT(!indexed || maxLevels > 1);
    auto capacity = indexed ? maxLevels - 1 : 1;
    auto arrayType = builder.getArrayType(uintType, builder.getIntValue(uintType, capacity));
    auto indexStorage = builder.emitVar(arrayType);
    auto idStorage = builder.emitVar(arrayType);
    auto zero = builder.getIntValue(uintType, 0);
    storeField(
        builder,
        context,
        state.instanceIndices,
        builder.emitElementAddress(indexStorage, zero));
    storeField(builder, context, state.instanceIDs, builder.emitElementAddress(idStorage, zero));
    if (!indexed)
    {
        builder.emitStore(builder.emitElementAddress(indexStorage, zero), indices);
        builder.emitStore(builder.emitElementAddress(idStorage, zero), ids);
        storeField(builder, context, state.instanceCount, builder.getIntValue(uintType, 1));
        return;
    }
    auto countFunction = createInstancePathOperation(module, InstancePathOperation::Count);
    auto elementFunction = createInstancePathOperation(module, InstancePathOperation::Element);
    auto count = builder.emitCallInst(uintType, countFunction, 1, &indices);
    storeField(builder, context, state.instanceCount, count);
    auto function = getParentFunc(context);
    for (IRIntegerValue i = 0; i < capacity; ++i)
    {
        auto element = builder.getIntValue(uintType, i);
        auto copy = builder.createBlock();
        auto next = builder.createBlock();
        function->addBlock(copy);
        function->addBlock(next);
        builder.emitIfElse(builder.emitLess(element, count), copy, next, next);
        builder.setInsertInto(copy);
        IRInst* indexArgs[] = {indices, element};
        IRInst* idArgs[] = {ids, element};
        builder.emitStore(
            builder.emitElementAddress(indexStorage, element),
            builder.emitCallInst(uintType, elementFunction, 2, indexArgs));
        builder.emitStore(
            builder.emitElementAddress(idStorage, element),
            builder.emitCallInst(uintType, elementFunction, 2, idArgs));
        builder.emitBranch(next);
        builder.setInsertInto(next);
    }
}

/// Creates storage and calls the group's independently supplied handles. It uses only checked
/// schema types and stage presence; no source implementation is retained or inspected here.
GroupCallback createGroupCallback(
    IRModule* module,
    const SchemaHitGroup& group,
    Index functionIndex,
    const RayData& carrier,
    MetalRTGeometryKind geometry,
    const List<CandidateParameter>& parameters,
    const CandidateResult& resultType,
    IRIntegerValue maxLevels)
{
    bool procedural = geometry == MetalRTGeometryKind::BoundingBox;
    IRBuilder builder(module);
    builder.setInsertInto(module->getModuleInst());
    GroupCallback result;
    result.functionIndex = functionIndex;
    result.carrier = &carrier;
    auto callback = builder.createFunc();
    result.function = callback;
    builder.addNameHintDecoration(callback, toSlice("portableRTCandidateGroup"));
    auto anyHitSignature = getAnyHitFunctionSignature(builder);
    auto intersectionSignature = getIntersectionFunctionSignature(builder);
    IRInst* intersectionOperands[] = {
        intersectionSignature,
        builder.getIntValue(builder.getIntType(), Int(Stage::Intersection))};
    auto intersectionTableType =
        builder.getType(kIROp_MetalVisibleFunctionTable, 2, intersectionOperands);
    List<IRType*> types;
    for (auto parameter : parameters)
        types.add(parameter.type);
    types.add(builder.getPtrType(builder.getUInt8Type(), AddressSpace::Global));
    types.add(builder.getPtrType(carrier.type, AddressSpace::ThreadLocal));
    types.add(intersectionSignature->getParamType(2));
    types.add(intersectionTableType);
    types.add(anyHitSignature->getParamType(1));
    callback->setFullType(builder.getFuncType(types, resultType.type));
    builder.setInsertInto(callback);
    builder.emitBlock();
    Dictionary<String, IRInst*> semantics;
    Dictionary<IROp, IRInst*> values;
    for (auto parameter : parameters)
    {
        auto value = builder.emitParam(parameter.type);
        builder.addNameHintDecoration(value, UnownedTerminatedStringSlice(parameter.semantic));
        semantics.add(String(parameter.semantic), value);
        if (parameter.operation != kIROp_Invalid)
            values.add(parameter.operation, value);
    }
    auto record = builder.emitParam(types[types.getCount() - 5]);
    auto rayData = builder.emitParam(types[types.getCount() - 4]);
    auto anyHitTable = builder.emitParam(types[types.getCount() - 3]);
    auto intersectionTable = builder.emitParam(types[types.getCount() - 2]);
    auto globals = builder.emitParam(types.getLast());
    auto state = createState(module);
    auto context = builder.emitVar(state.type);
    builder.emitStore(context, builder.emitDefaultConstruct(state.type));
    auto erasedPointer = anyHitSignature->getParamType(0);
    // Empty payloads have no carrier storage. The selected provider creates its nominal empty
    // token; taking and erasing an address here would incorrectly make zero-byte storage live.
    IRInst* payload = builder.emitDefaultConstruct(erasedPointer);
    if (carrier.payloadKey)
        payload = builder.emitBitCast(
            erasedPointer,
            builder.emitFieldAddress(rayData, carrier.payloadKey));
    storeField(builder, context, state.payload, payload);
    storeField(builder, context, state.record, record);
    if (procedural)
    {
        if (!isEmptyType(group.attributesType))
        {
            auto attributes = builder.emitVar(group.attributesType);
            storeField(
                builder,
                context,
                state.attributes,
                builder.emitBitCast(erasedPointer, attributes));
            auto committed = builder.emitFieldAddress(
                rayData,
                carrier.customAttributeKeys.getValue(group.attributesType));
            storeField(
                builder,
                context,
                state.committedAttributes,
                builder.emitBitCast(erasedPointer, committed));
        }
        SLANG_RELEASE_ASSERT(carrier.customHitKindKey);
        storeField(
            builder,
            context,
            state.committedHitKind,
            builder.emitFieldAddress(rayData, carrier.customHitKindKey));
    }
    struct CarrierInput
    {
        IRStructKey* key;
        IROp op;
    };
    CarrierInput carrierInputs[] = {
        {carrier.minDistanceKey, kIROp_MetalRTGetMinDistance},
        {carrier.rayTimeKey, kIROp_MetalRTGetRayTime},
        {carrier.rayFlagsKey, kIROp_MetalRTGetRayFlags},
        {carrier.dispatchRaysIndexKey, kIROp_MetalRTGetDispatchRaysIndex},
        {carrier.dispatchRaysDimensionsKey, kIROp_MetalRTGetDispatchRaysDimensions}};
    for (auto input : carrierInputs)
        if (input.key && !values.containsKey(input.op))
            values.add(input.op, loadField(builder, rayData, input.key));
    for (auto input : state.inputs)
        if (auto value = values.tryGetValue(input.first))
            storeField(builder, context, input.second, *value);
    auto distance = values.getValue(kIROp_MetalRTGetCurrentDistance);
    storeField(builder, context, state.currentMax, distance);
    storeField(builder, context, state.distance, distance);
    storeField(builder, context, state.origin, semantics.getValue(String("origin")));
    storeField(builder, context, state.direction, semantics.getValue(String("direction")));
    if (auto opaque = semantics.tryGetValue(String("opaque")))
        storeField(builder, context, state.opaque, *opaque);
    if (auto front = values.tryGetValue(kIROp_MetalRTGetTriangleFrontFacing))
    {
        IRInst* fields[] = {
            *front,
            builder.getIntValue(builder.getUIntType(), 254),
            builder.getIntValue(builder.getUIntType(), 255)};
        storeField(
            builder,
            context,
            state.hitKind,
            builder.emitIntrinsicInst(builder.getUIntType(), kIROp_Select, 3, fields));
    }
    if (auto indices = values.tryGetValue(kIROp_MetalRTGetInstanceIndex))
        initializeInstancePath(
            builder,
            module,
            context,
            state,
            *indices,
            values.getValue(kIROp_MetalRTGetInstanceID),
            maxLevels);
    else
    {
        // A primitive acceleration structure has no instance transforms.
        auto floatType = builder.getFloatType();
        auto zero = builder.getFloatValue(floatType, 0);
        auto one = builder.getFloatValue(floatType, 1);
        IRInst* elements[] = {one, zero, zero, zero, one, zero, zero, zero, one, zero, zero, zero};
        auto matrixType =
            cast<IRPtrTypeBase>(
                builder
                    .emitFieldAddress(context, state.inputs.getValue(kIROp_MetalRTGetObjectToWorld))
                    ->getDataType())
                ->getValueType();
        auto identity = builder.emitMakeMatrix(matrixType, 12, elements);
        storeField(
            builder,
            context,
            state.inputs.getValue(kIROp_MetalRTGetObjectToWorld),
            identity);
        storeField(
            builder,
            context,
            state.inputs.getValue(kIROp_MetalRTGetWorldToObject),
            identity);
    }
    auto index = builder.getIntValue(builder.getUIntType(), functionIndex);
    auto erasedState = builder.emitBitCast(erasedPointer, context);
    IRInst* decision = builder.getIntValue(builder.getUIntType(), UInt(CandidateExit::Accept));
    if (procedural)
    {
        SLANG_RELEASE_ASSERT(group.intersection.isPresent);
        auto anyHitIndex = group.anyHit.isPresent
                               ? index
                               : builder.getIntValue(builder.getUIntType(), 0xffffffffu);
        IRInst* arguments[] =
            {intersectionTable, index, erasedState, globals, anyHitTable, anyHitIndex};
        decision = builder.emitIntrinsicInst(
            builder.getUIntType(),
            kIROp_MetalRTCandidateDispatch,
            SLANG_COUNT_OF(arguments),
            arguments);
        builder.emitReturn(emitResult(
            builder,
            resultType,
            loadField(builder, context, state.accepted),
            builder.emitNeq(
                decision,
                builder.getIntValue(builder.getUIntType(), UInt(CandidateExit::EndSearch))),
            loadField(builder, context, state.currentMax)));
    }
    else
    {
        if (group.anyHit.isPresent)
        {
            IRInst* arguments[] = {anyHitTable, index, erasedState, globals};
            decision = builder.emitIntrinsicInst(
                builder.getUIntType(),
                kIROp_MetalRTCandidateDispatch,
                SLANG_COUNT_OF(arguments),
                arguments);
        }
        builder.emitReturn(emitResult(
            builder,
            resultType,
            builder.emitNeq(
                decision,
                builder.getIntValue(builder.getUIntType(), UInt(CandidateExit::Ignore))),
            builder.emitNeq(
                decision,
                builder.getIntValue(builder.getUIntType(), UInt(CandidateExit::EndSearch)))));
    }
    return result;
}

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
    const MetalRTDescriptorResources& resources,
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
    types.add(resources.physicalParameterBlockType);
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
    auto rayData = builder.emitParam(types[types.getCount() - 2]);
    builder.addNameHintDecoration(rayData, toSlice("rayData"));
    builder.addTargetSystemValueDecoration(rayData, toSlice("payload"));
    auto descriptor = builder.emitParam(types.getLast());
    builder.addNameHintDecoration(descriptor, toSlice("candidateDescriptor"));
    builder.addDecoration(descriptor, kIROp_MetalRTCandidateDescriptorDecoration);
    auto loadTable = [&](MetalRTDescriptorResource role)
    {
        auto field = resources.getField(role);
        auto pointerType = builder.getPtrType(field->getFieldType(), AddressSpace::Uniform);
        return builder.emitLoad(builder.emitFieldAddress(pointerType, descriptor, field->getKey()));
    };
    auto anyHitTable = loadTable(MetalRTDescriptorResource::AnyHitVisibleFunctionTable);
    auto intersectionTable = loadTable(MetalRTDescriptorResource::IntersectionVisibleFunctionTable);
    auto globals = builder.emitIntrinsicInst(
        getAnyHitFunctionSignature(builder)->getParamType(1),
        kIROp_MetalRTGetGlobalResourceContext,
        0,
        nullptr);
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
        arguments.add(anyHitTable);
        arguments.add(intersectionTable);
        arguments.add(globals);
        auto result = builder.emitCallInst(resultType.type, callback.function, arguments);
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

IRFuncType* getAnyHitFunctionSignature(IRBuilder& builder)
{
    IRType* parameters[] = {
        builder.getPtrType(builder.getVoidType(), AddressSpace::ThreadLocal),
        builder.getPtrType(builder.getUInt8Type(), AddressSpace::ThreadLocal)};
    return builder.getFuncType(SLANG_COUNT_OF(parameters), parameters, builder.getUIntType());
}

IRFuncType* getIntersectionFunctionSignature(IRBuilder& builder)
{
    auto anyHit = getAnyHitFunctionSignature(builder);
    IRInst* tableOperands[] = {
        anyHit,
        builder.getIntValue(builder.getIntType(), IRIntegerValue(Stage::AnyHit))};
    IRType* parameters[] = {
        anyHit->getParamType(0),
        anyHit->getParamType(1),
        builder.getType(kIROp_MetalVisibleFunctionTable, 2, tableOperands),
        builder.getUIntType()};
    return builder.getFuncType(SLANG_COUNT_OF(parameters), parameters, builder.getUIntType());
}

VisibleAdapter createSelectedCandidateAdapter(
    IRModule* module,
    const SchemaStage& stage,
    UnownedStringSlice symbolName,
    bool& requiresResultTypeLowering,
    DiagnosticSink* sink)
{
    SLANG_RELEASE_ASSERT(stage.invoke && stage.input);
    SLANG_RELEASE_ASSERT(stage.stage == Stage::AnyHit || stage.stage == Stage::Intersection);
    bool intersection = stage.stage == Stage::Intersection;
    auto state = createState(module);
    IRBuilder builder(module);
    builder.setInsertInto(module->getModuleInst());
    VisibleAdapter result;
    result.function = builder.createFunc();
    auto signature = intersection ? getIntersectionFunctionSignature(builder)
                                  : getAnyHitFunctionSignature(builder);
    result.function->setFullType(signature);
    builder.addNameHintDecoration(result.function, symbolName);
    builder.addExportDecoration(result.function, symbolName);
    builder.addExternCppDecoration(result.function, symbolName);
    builder.addKeepAliveDecoration(result.function);
    IRInst* tableOperands[] = {
        signature,
        builder.getIntValue(builder.getIntType(), IRIntegerValue(stage.stage))};
    auto tableType = builder.getType(kIROp_MetalVisibleFunctionTable, 2, tableOperands);
    builder.addDecoration(
        result.function,
        kIROp_MetalVisibleFunctionDecoration,
        tableOperands[1],
        tableType);
    builder.setInsertInto(result.function);
    builder.emitBlock();
    auto erasedState = builder.emitParam(signature->getParamType(0));
    builder.addNameHintDecoration(erasedState, toSlice("candidateState"));
    auto globals = builder.emitParam(signature->getParamType(1));
    builder.addNameHintDecoration(globals, toSlice("globalResourceContext"));
    builder.addDecoration(globals, kIROp_MetalRTGlobalResourceContextDecoration);
    IRInst* anyHitTable = nullptr;
    IRInst* anyHitIndex = nullptr;
    if (intersection)
    {
        anyHitTable = builder.emitParam(signature->getParamType(2));
        builder.addNameHintDecoration(anyHitTable, toSlice("anyHitFunctions"));
        anyHitIndex = builder.emitParam(signature->getParamType(3));
        builder.addNameHintDecoration(anyHitIndex, toSlice("anyHitIndex"));
    }
    auto context =
        builder.emitBitCast(builder.getPtrType(state.type, AddressSpace::ThreadLocal), erasedState);
    List<IRFunc*> functions;
    auto invoke = specializeCandidateGraph(module, stage.invoke, functions);
    emitStageCall(builder, invoke);
    builder.emitReturn(builder.getIntValue(builder.getUIntType(), UInt(CandidateExit::Accept)));
    PortableRTInvocationParameter stateParameter(module, context->getDataType(), "candidateState");
    stateParameter.bind(result.function, context);
    lowerMutableInputs(module, functions, stateParameter, state, intersection);
    if (intersection)
    {
        auto report = createReportHit(module, state, stage.input->getAttributesType());
        PortableRTInvocationParameter globalsParameter(
            module,
            globals->getDataType(),
            "globalResources");
        PortableRTInvocationParameter tableParameter(
            module,
            anyHitTable->getDataType(),
            "anyHitFunctions");
        PortableRTInvocationParameter indexParameter(
            module,
            anyHitIndex->getDataType(),
            "anyHitIndex");
        globalsParameter.bind(result.function, globals);
        tableParameter.bind(result.function, anyHitTable);
        indexParameter.bind(result.function, anyHitIndex);
        List<IRInst*> reports;
        for (auto function : functions)
            for (auto block : function->getBlocks())
                for (auto operation : block->getOrdinaryInsts())
                    if (operation->getOp() == kIROp_MetalRTReportHit)
                        reports.add(operation);
        for (auto operation : reports)
        {
            auto function = getParentFunc(operation);
            IRInst* arguments[] = {
                stateParameter.get(function),
                globalsParameter.get(function),
                tableParameter.get(function),
                indexParameter.get(function),
                operation->getOperand(0),
                operation->getOperand(1),
                operation->getOperand(2)};
            builder.setInsertBefore(operation);
            operation->replaceUsesWith(builder.emitCallInst(
                builder.getBoolType(),
                report,
                SLANG_COUNT_OF(arguments),
                arguments));
            operation->removeAndDeallocate();
        }
        functions.add(report);
    }
    functions.add(result.function);
    HashSet<IRFunc*> handlers;
    handlers.add(result.function);
    requiresResultTypeLowering |= lowerCandidateExits(module, functions, handlers, sink);
    collectBindings(module, result, context, state);
    return result;
}

CandidateAdapters createCandidateAdapters(
    IRModule* module,
    const Schema& schema,
    const MetalRTDescriptorResources& resources,
    const Dictionary<IRType*, RayData>& carriers,
    const RayDataHeader& header,
    UInt tags,
    IRIntegerValue maxLevels,
    IRIntegerValue hitStride,
    IRFunc* instanceContribution,
    IRType* instancePathParamType,
    UnownedStringSlice symbolPrefix)
{
    CandidateAdapters result;
    IRBuilder builder(module);
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
        auto parameters = getParameters(builder, geometry.kind, tags, instancePathParamType);
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
                nativeResult,
                maxLevels));
        }
        StringBuilder name;
        name << symbolPrefix << geometry.suffix;
        result.intersectionFunctions.add(
            geometry.slot,
            createDispatcher(
                module,
                callbacks,
                resources,
                parameters,
                nativeResult,
                geometry.kind,
                header,
                tags,
                maxLevels,
                hitStride,
                instanceContribution,
                name.getUnownedSlice()));
    }
    return result;
}

} // namespace Metal
} // namespace PortableRT
} // namespace Slang
