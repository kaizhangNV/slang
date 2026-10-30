#include "slang-ir-metal-portable-ray-tracing-acceleration.h"

#include "slang-ir-util.h"
#include "slang-rich-diagnostics.h"
#include "slang-target-program.h"

namespace Slang
{
namespace PortableRT
{
namespace Metal
{

bool getTraceContextRequirements(
    const Schema& schema,
    TargetRequest* target,
    IRInst* useSite,
    DiagnosticSink* sink,
    TraceContextRequirements& result)
{
    result = {};
    auto structure = cast<IRRaytracingAccelerationStructureType>(schema.accelerationStructure);
    // Source forms have no operands (portable instancing) or one checked topology argument.
    // Physical AS types are produced later and must never replace the schema's source type.
    SLANG_RELEASE_ASSERT(structure->getOperandCount() <= 1);
    if (structure->getOperandCount())
    {
        auto levels = as<IRIntLit>(structure->getOperand(0));
        if (!levels || levels->getValue() < 1 || levels->getValue() > 32)
        {
            sink->diagnose(Diagnostics::InvalidMetalRayTracingMaxLevels{
                .levelCount = levels ? Int(levels->getValue()) : Int(-1),
                .location = useSite->sourceLoc});
            return false;
        }
        if (levels->getValue() == 1)
            result.tags = 0;
        else
        {
            if (!target->getTargetCaps().implies(CapabilityAtom::metallib_3_1))
            {
                sink->diagnose(Diagnostics::MetalRayTracingMultilevelRequiresMetallib31{
                    .location = useSite->sourceLoc});
                return false;
            }
            result.maxLevels = levels->getValue();
        }
    }
    switch (getBuiltinRole(schema.motion))
    {
    case KnownBuiltinDeclName::RayTracingNoMotion:
        break;
    case KnownBuiltinDeclName::RayTracingPrimitiveMotion:
        result.tags |= UInt(MetalRTTag::PrimitiveMotion);
        break;
    case KnownBuiltinDeclName::RayTracingInstanceMotion:
        result.tags |= UInt(MetalRTTag::InstanceMotion);
        break;
    case KnownBuiltinDeclName::RayTracingPrimitiveAndInstanceMotion:
        result.tags |= UInt(MetalRTTag::PrimitiveMotion) | UInt(MetalRTTag::InstanceMotion);
        break;
    default:
        SLANG_UNEXPECTED("expected a checked closed ray-motion type");
    }
    if ((result.tags & UInt(MetalRTTag::InstanceMotion)) &&
        !(result.tags & UInt(MetalRTTag::Instancing)))
    {
        sink->diagnose(Diagnostics::MetalRayTracingInstanceMotionRequiresInstancing{
            .location = useSite->sourceLoc});
        return false;
    }
    if (target->getTargetCaps().implies(CapabilityAtom::metal_raytracing_extended_limits))
        result.tags |= UInt(MetalRTTag::ExtendedLimits);
    return true;
}

namespace
{

// Builds the canonical native resource type, preserving source topology while retaining only
// tags accepted by Metal acceleration_structure. Tags such as triangle_data, max_levels, and
// extended_limits belong to the traversal/table ABI and must not leak onto the resource.
static IRType* getPhysicalType(
    IRBuilder& builder,
    IRRaytracingAccelerationStructureType* sourceType,
    UInt tagMask)
{
    SLANG_RELEASE_ASSERT(sourceType->getOperandCount() <= 2);
    IRInst* topology = sourceType->getOperandCount() == 0
                           ? builder.getIntValue(builder.getIntType(), 0)
                           : sourceType->getOperand(0);
    IRInst* operands[] = {
        topology,
        builder.getIntValue(
            builder.getIntType(),
            IRIntegerValue(
                tagMask & (UInt(MetalRTTag::Instancing) | UInt(MetalRTTag::PrimitiveMotion) |
                           UInt(MetalRTTag::InstanceMotion)))),
    };
    return builder.getType(
        kIROp_RaytracingAccelerationStructureType,
        SLANG_COUNT_OF(operands),
        operands);
}

// Rebuilds one homogeneous array container through the canonical builder. A changed Metal
// resource element has the same binding footprint, so its count and explicit stride survive.
static IRArrayTypeBase* getPhysicalArrayType(
    IRBuilder& builder,
    IRType* sourceType,
    IRType* elementType)
{
    auto array = cast<IRArrayTypeBase>(sourceType);
    if (auto count = array->getElementCount())
        return builder.getArrayType(elementType, count, array->getArrayStride());
    return builder.getUnsizedArrayType(elementType, array->getArrayStride());
}

// Finds the concrete field selected by a field access. Generic specialization can reuse one
// `IRStructKey` in several nominal struct types, so the key alone is not a storage identity; the
// aggregate type selects the exact `IRStructField` that owns the storage.
static IRStructField* findStorageField(IRBuilder& builder, IRInst* aggregate, IRInst* fieldKey)
{
    auto aggregateType = cast<IRType>(aggregate->getDataType());
    auto structType = as<IRStructType>(tryGetPointedToType(&builder, aggregateType));
    if (!structType)
        structType = as<IRStructType>(aggregateType);
    return structType ? findStructField(structType, cast<IRStructKey>(fieldKey)) : nullptr;
}

// Physicalizes every representation of one Metal acceleration-structure value.
//
// The source type carries topology, while the trace context carries motion requirements. Metal
// combines both facts in the native acceleration-structure type, so the physical type cannot be
// assigned only to the final trace operand. Consider this example:
//
//     struct SceneParameters { rt::AccelerationStructure scene; }
//     ParameterBlock<SceneParameters> parameters;
//     tracer.trace(desc, parameters.scene, program, payload);
//
// The trace sees an `IRFieldExtract`, but Metal declares the resource from the corresponding
// `IRStructField`. This context follows the ordinary SSA/storage equivalence edges between field
// accesses, loads, local variables, function parameters, calls, and phi parameters. Every member
// of that connected component receives one physical type. Recording assignments by the concrete
// storage instruction also makes two traces with incompatible motion requirements diagnose the
// shared field or variable instead of leaving an internally inconsistent Metal declaration.
struct MetalAccelerationStructurePhysicalizationContext
{
    MetalAccelerationStructurePhysicalizationContext(IRModule* module, DiagnosticSink* sink)
        : sink(sink), builder(module)
    {
    }

    DiagnosticSink* sink;
    IRBuilder builder;
    Dictionary<IRInst*, IRType*> assignedTypes;
    Dictionary<IRFunc*, IRType*> assignedResultTypes;
    Dictionary<IRInst*, IRInst*> conflictOwners;
    HashSet<IRInst*> diagnosedConflicts;

    // Derives the native Metal type required by one trace and applies it to the trace operand's
    // complete producer/storage component.
    bool physicalize(IRInst* value, UInt tagMask, IRInst* useSite)
    {
        auto sourceType = cast<IRRaytracingAccelerationStructureType>(value->getDataType());
        return assignValue(value, getPhysicalType(builder, sourceType, tagMask), useSite);
    }

private:
    // Reports one error per shared producer or storage location, while retaining the trace use as
    // the diagnostic location that explains which requirement introduced the conflict.
    bool diagnoseConflict(IRInst* owner, IRInst* useSite)
    {
        // Several field extracts can name the same storage location. Associate those transient
        // SSA values with the field so that incompatible traces report one conflict for the
        // shared resource instead of one error for every extraction.
        if (auto storageOwner = conflictOwners.tryGetValue(owner))
            owner = *storageOwner;
        if (diagnosedConflicts.add(owner))
        {
            sink->diagnose(Diagnostics::MetalRayTracingAccelerationStructureMotionConflict{
                .location = useSite && useSite->sourceLoc.isValid() ? useSite->sourceLoc
                                                                    : owner->sourceLoc});
        }
        return false;
    }

    // Records an exact type once and tells the caller whether it must traverse this node's edges.
    // Returning `isNew == false` terminates cycles such as a loop phi feeding itself.
    bool assignInstructionType(IRInst* inst, IRType* physicalType, IRInst* useSite, bool& outIsNew)
    {
        outIsNew = false;
        if (auto assignedType = assignedTypes.tryGetValue(inst))
        {
            return *assignedType == physicalType || diagnoseConflict(inst, useSite);
        }
        assignedTypes.add(inst, physicalType);
        inst->setFullType(physicalType);
        outIsNew = true;
        return true;
    }

    // Rebuilds a pointer with a physical pointee while preserving its existing address space.
    IRPtrTypeBase* getPhysicalPointerType(IRInst* pointer, IRType* valueType)
    {
        auto pointerType = as<IRPtrTypeBase>(pointer->getDataType());
        SLANG_RELEASE_ASSERT(pointerType);
        return builder.getPtrTypeWithAddressSpace(valueType, pointerType);
    }

    // Dispatches propagation by whether an SSA edge transports a value or a pointer to storage.
    bool assignTypedValue(IRInst* value, IRType* physicalType, IRInst* useSite)
    {
        if (as<IRPtrTypeBase>(physicalType))
            return assignPointer(value, cast<IRPtrTypeBase>(physicalType), useSite);
        SLANG_RELEASE_ASSERT(
            as<IRRaytracingAccelerationStructureType>(physicalType) ||
            as<IRArrayTypeBase>(physicalType));
        return assignValue(value, physicalType, useSite);
    }

    // Joins a parameter with the values supplied by every predecessor or direct caller.
    bool assignParameterInputs(IRParam* parameter, IRType* physicalType, IRInst* useSite)
    {
        auto block = cast<IRBlock>(parameter->getParent());
        if (auto func = as<IRFunc>(block->getParent()); func && block == func->getFirstBlock())
        {
            auto parameterIndex = block->getParamIndex(parameter);
            SLANG_RELEASE_ASSERT(parameterIndex >= 0);
            fixUpFuncType(func);

            // An entry-point parameter has no callers. A helper parameter is shared by every
            // direct call, so all arguments must join the same physical storage component.
            for (auto use = func->firstUse; use; use = use->nextUse)
            {
                auto call = as<IRCall>(use->getUser());
                if (!call || call->getCallee() != func ||
                    UInt(parameterIndex) >= call->getArgCount())
                    continue;
                if (!assignTypedValue(call->getArg(UInt(parameterIndex)), physicalType, useSite))
                    return false;
            }
            return true;
        }

        // Non-entry block parameters are SSA phi values. `getPhiArgs` follows the compiler's
        // canonical predecessor-edge representation instead of rediscovering operands by index.
        for (auto argument : getPhiArgs(parameter))
        {
            if (!assignTypedValue(argument, physicalType, useSite))
                return false;
        }
        return true;
    }

    // Keeps a helper's result type, return operands, and direct call results physically identical.
    bool assignFunctionResult(IRFunc* func, IRType* physicalType, IRInst* useSite)
    {
        if (auto assignedType = assignedResultTypes.tryGetValue(func))
            return *assignedType == physicalType || diagnoseConflict(func, useSite);
        assignedResultTypes.add(func, physicalType);
        fixUpFuncType(func, physicalType);

        // The function signature, every returned value, and every direct call result describe one
        // ABI value. Traverse both directions so a helper returning an acceleration structure does
        // not leave its declaration at the source-only type.
        for (auto block : func->getBlocks())
        {
            if (auto returnInst = as<IRReturn>(block->getTerminator()))
            {
                if (!assignTypedValue(returnInst->getVal(), physicalType, useSite))
                    return false;
            }
        }
        for (auto use = func->firstUse; use; use = use->nextUse)
        {
            auto call = as<IRCall>(use->getUser());
            if (call && call->getCallee() == func && !assignTypedValue(call, physicalType, useSite))
            {
                return false;
            }
        }
        return true;
    }

    // Joins each use of an argument with the matching parameter of a direct helper call.
    bool assignCallArgument(IRCall* call, IRInst* argument, IRType* physicalType, IRInst* useSite)
    {
        auto callee = as<IRFunc>(getResolvedInstForDecorations(call->getCallee()));
        auto firstBlock = callee ? callee->getFirstBlock() : nullptr;
        if (!firstBlock)
            return true;

        for (UInt i = 0; i < call->getArgCount(); ++i)
        {
            if (call->getArg(i) != argument)
                continue;
            SLANG_RELEASE_ASSERT(i < callee->getParamCount());
            if (!assignTypedValue(getParamAt(firstBlock, i), physicalType, useSite))
                return false;
        }
        return true;
    }

    // Joins each branch argument with the corresponding phi parameter in the target block.
    bool assignBranchArgument(
        IRUnconditionalBranch* branch,
        IRInst* argument,
        IRType* physicalType,
        IRInst* useSite)
    {
        for (UInt i = 0; i < branch->getArgCount(); ++i)
        {
            if (branch->getArg(i) != argument)
                continue;
            if (!assignTypedValue(getParamAt(branch->getTargetBlock(), i), physicalType, useSite))
            {
                return false;
            }
        }
        return true;
    }

    // Propagates a physical value type to storage and interprocedural control-flow consumers.
    bool assignValueUses(IRInst* value, IRType* physicalType, IRInst* useSite)
    {
        for (auto use = value->firstUse; use; use = use->nextUse)
        {
            auto user = use->getUser();
            if (auto store = as<IRStore>(user))
            {
                if (store->getVal() == value &&
                    !assignPointer(
                        store->getPtr(),
                        getPhysicalPointerType(store->getPtr(), physicalType),
                        useSite))
                {
                    return false;
                }
            }
            else if (auto call = as<IRCall>(user))
            {
                if (!assignCallArgument(call, value, physicalType, useSite))
                    return false;
            }
            else if (auto branch = as<IRUnconditionalBranch>(user))
            {
                if (!assignBranchArgument(branch, value, physicalType, useSite))
                    return false;
            }
            else if (auto select = as<IRSelect>(user))
            {
                if (!assignTypedValue(select, physicalType, useSite))
                    return false;
            }
            else if (auto element = as<IRGetElement>(user))
            {
                auto array = cast<IRArrayTypeBase>(physicalType);
                if (!assignValue(element, array->getElementType(), useSite))
                    return false;
            }
            else if (as<IRMakeArray>(user) || as<IRMakeArrayFromElement>(user))
            {
                // One array has one native element type even when different runtime indices
                // select its members. Join the container instead of retagging one extraction.
                auto array = getPhysicalArrayType(builder, user->getDataType(), physicalType);
                if (!assignValue(user, array, useSite))
                    return false;
            }
            else if (auto aggregate = as<IRMakeStruct>(user))
            {
                // A resource used to initialize a nominal field must agree with every access to
                // that field, including accesses made through other aggregate values.
                auto structure = cast<IRStructType>(aggregate->getDataType());
                UInt index = 0;
                for (auto field : structure->getFields())
                {
                    if (aggregate->getOperand(index++) == value &&
                        !assignField(field, physicalType, useSite))
                        return false;
                }
            }
            else if (auto returnInst = as<IRReturn>(user))
            {
                if (returnInst->getVal() != value)
                    continue;
                auto block = cast<IRBlock>(returnInst->getParent());
                auto func = cast<IRFunc>(block->getParent());
                if (!assignFunctionResult(func, physicalType, useSite))
                    return false;
            }
        }
        return true;
    }

    // Assigns the native type to a concrete field and every SSA access that selects that field.
    bool assignField(IRStructField* field, IRType* physicalType, IRInst* useSite)
    {
        if (auto assignedType = assignedTypes.tryGetValue(field))
            return *assignedType == physicalType || diagnoseConflict(field, useSite);
        assignedTypes.add(field, physicalType);
        conflictOwners.set(field, field);
        field->setFieldType(physicalType);

        // A struct key can be shared by specialized nominal structs. Resolve every access through
        // its aggregate and update only accesses that select this exact field. Acceleration-
        // structure topology and motion change Metal's native C++ type but not its resource kind
        // or binding footprint, so the existing struct-field layout remains authoritative.
        auto fieldKey = field->getKey();
        for (auto use = fieldKey->firstUse; use; use = use->nextUse)
        {
            auto user = use->getUser();
            if (auto extract = as<IRFieldExtract>(user))
            {
                if (findStorageField(builder, extract->getBase(), extract->getField()) != field)
                    continue;
                conflictOwners.set(extract, field);
                if (!assignValue(extract, physicalType, useSite))
                {
                    return false;
                }
            }
            else if (auto address = as<IRFieldAddress>(user))
            {
                if (findStorageField(builder, address->getBase(), address->getField()) != field)
                    continue;
                conflictOwners.set(address, field);
                if (!assignPointer(address, getPhysicalPointerType(address, physicalType), useSite))
                {
                    return false;
                }
            }
        }
        // Field initializers are not uses of the field key: MakeStruct transports them by the
        // nominal struct's declared field order. Join those producers explicitly.
        auto structure = cast<IRStructType>(field->getParent());
        UInt fieldIndex = 0;
        for (auto candidate : structure->getFields())
        {
            if (candidate == field)
                break;
            ++fieldIndex;
        }
        for (auto use = structure->firstUse; use; use = use->nextUse)
        {
            auto aggregate = as<IRMakeStruct>(use->getUser());
            if (aggregate && aggregate->getDataType() == structure &&
                !assignValue(aggregate->getOperand(fieldIndex), physicalType, useSite))
                return false;
        }
        return true;
    }

    // Assigns a pointer's native pointee type and joins all values loaded from or stored into it.
    bool assignPointer(IRInst* pointer, IRPtrTypeBase* physicalType, IRInst* useSite)
    {
        bool isNew = false;
        if (!assignInstructionType(pointer, physicalType, useSite, isNew))
            return false;
        if (!isNew)
            return true;

        if (auto fieldAddress = as<IRFieldAddress>(pointer))
        {
            auto field =
                findStorageField(builder, fieldAddress->getBase(), fieldAddress->getField());
            SLANG_RELEASE_ASSERT(field);
            if (!assignField(field, physicalType->getValueType(), useSite))
                return false;
        }
        else if (auto parameter = as<IRParam>(pointer))
        {
            if (!assignParameterInputs(parameter, physicalType, useSite))
                return false;
        }
        else if (auto select = as<IRSelect>(pointer))
        {
            if (!assignPointer(select->getTrueResult(), physicalType, useSite) ||
                !assignPointer(select->getFalseResult(), physicalType, useSite))
                return false;
        }
        else if (auto element = as<IRGetElementPtr>(pointer))
        {
            auto base = element->getBase();
            auto baseType = cast<IRPtrTypeBase>(base->getDataType());
            auto array = getPhysicalArrayType(
                builder,
                baseType->getValueType(),
                physicalType->getValueType());
            if (!assignPointer(base, getPhysicalPointerType(base, array), useSite))
                return false;
        }

        // Changing storage changes both its producers and every load. Walking all loads/stores is
        // what makes a local variable one conflict domain instead of independently retagged SSA
        // values.
        for (auto use = pointer->firstUse; use; use = use->nextUse)
        {
            auto user = use->getUser();
            if (auto load = as<IRLoad>(user))
            {
                if (load->getPtr() == pointer &&
                    !assignValue(load, physicalType->getValueType(), useSite))
                {
                    return false;
                }
            }
            else if (auto store = as<IRStore>(user))
            {
                if (store->getPtr() == pointer &&
                    !assignValue(store->getVal(), physicalType->getValueType(), useSite))
                {
                    return false;
                }
            }
            else if (auto call = as<IRCall>(user))
            {
                if (!assignCallArgument(call, pointer, physicalType, useSite))
                    return false;
            }
            else if (auto branch = as<IRUnconditionalBranch>(user))
            {
                if (!assignBranchArgument(branch, pointer, physicalType, useSite))
                    return false;
            }
            else if (auto select = as<IRSelect>(user))
            {
                if (!assignPointer(select, physicalType, useSite))
                    return false;
            }
            else if (auto element = as<IRGetElementPtr>(user))
            {
                auto array = cast<IRArrayTypeBase>(physicalType->getValueType());
                if (!assignPointer(
                        element,
                        getPhysicalPointerType(element, array->getElementType()),
                        useSite))
                    return false;
            }
        }
        return true;
    }

    // Assigns a native value type, follows its producer to storage, and updates all consumers.
    bool assignValue(IRInst* value, IRType* physicalType, IRInst* useSite)
    {
        bool isNew = false;
        if (!assignInstructionType(value, physicalType, useSite, isNew))
            return false;
        if (!isNew)
            return true;

        if (auto load = as<IRLoad>(value))
        {
            if (!assignPointer(
                    load->getPtr(),
                    getPhysicalPointerType(load->getPtr(), physicalType),
                    useSite))
            {
                return false;
            }
        }
        else if (auto fieldExtract = as<IRFieldExtract>(value))
        {
            auto field =
                findStorageField(builder, fieldExtract->getBase(), fieldExtract->getField());
            SLANG_RELEASE_ASSERT(field);
            if (!assignField(field, physicalType, useSite))
                return false;
        }
        else if (auto parameter = as<IRParam>(value))
        {
            if (!assignParameterInputs(parameter, physicalType, useSite))
                return false;
        }
        else if (auto select = as<IRSelect>(value))
        {
            if (!assignValue(select->getTrueResult(), physicalType, useSite) ||
                !assignValue(select->getFalseResult(), physicalType, useSite))
                return false;
        }
        else if (auto element = as<IRGetElement>(value))
        {
            auto base = element->getBase();
            auto array = getPhysicalArrayType(builder, base->getDataType(), physicalType);
            if (!assignValue(base, array, useSite))
                return false;
        }
        else if (as<IRMakeArray>(value) || as<IRMakeArrayFromElement>(value))
        {
            auto array = cast<IRArrayTypeBase>(physicalType);
            for (UInt index = 0; index < value->getOperandCount(); ++index)
                if (!assignValue(value->getOperand(index), array->getElementType(), useSite))
                    return false;
        }
        else if (auto call = as<IRCall>(value))
        {
            auto callee = as<IRFunc>(getResolvedInstForDecorations(call->getCallee()));
            if (callee && !assignFunctionResult(callee, physicalType, useSite))
                return false;
        }
        else if (auto globalConstant = as<IRGlobalConstant>(value))
        {
            if (auto initializer = globalConstant->getValue())
            {
                if (!assignValue(initializer, physicalType, useSite))
                    return false;
            }
        }

        return assignValueUses(value, physicalType, useSite);
    }
};

} // namespace

bool physicalizeAccelerationStructures(
    IRModule* module,
    const List<AccelerationStructureUse>& uses,
    DiagnosticSink* sink)
{
    MetalAccelerationStructurePhysicalizationContext context(module, sink);
    bool success = true;
    for (const auto& use : uses)
        success &= context.physicalize(use.value, use.tags, use.useSite);
    return success;
}

} // namespace Metal
} // namespace PortableRT
} // namespace Slang
