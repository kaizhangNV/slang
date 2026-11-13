#include "slang/slang-ir-metal-portable-ray-tracing-acceleration.h"
#include "slang/slang-ir-util.h"
#include "slang/slang-rich-diagnostics.h"
#include "static-unit-test-env.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

// This fixture deliberately retains the value and storage edges that ordinary optimization can
// remove before code generation: a helper select/return, aggregate construction/extraction,
// and a stored local. The common source resource type must not join unrelated globals or fields.
SLANG_UNIT_TEST(metalPortableAccelerationStructureComponents)
{
    StaticUnitTestEnv env(unitTestContext);
    IRFixtureBuilder fixture(env.getSessionImpl());
    auto module = fixture.getModule();
    IRBuilder builder(module);
    builder.setInsertInto(module);
    IRInst* topology = builder.getIntValue(builder.getIntType(), 0);
    auto sourceType = builder.getType(kIROp_RaytracingAccelerationStructureType, 1, &topology);
    auto nativeType = builder.getType(kIROp_RaytracingAccelerationStructureType, 0, nullptr);
    auto nativeScene = builder.createGlobalParam(nativeType);
    auto first = builder.createGlobalParam(sourceType);
    auto second = builder.createGlobalParam(sourceType);
    auto independent = builder.createGlobalParam(sourceType);
    auto condition = builder.createGlobalParam(builder.getBoolType());
    auto fieldKey = builder.createStructKey();
    auto holder = builder.createStructType();
    auto field = builder.createStructField(holder, fieldKey, sourceType);
    // Specializations of one source aggregate can share a key, but not a concrete field.
    auto unrelatedHolder = builder.createStructType();
    auto unrelatedField = builder.createStructField(unrelatedHolder, fieldKey, sourceType);

    auto helper = builder.createFunc();
    IRType* parameterTypes[] = {sourceType, sourceType, builder.getBoolType()};
    helper->setFullType(builder.getFuncType(3, parameterTypes, sourceType));
    builder.setInsertInto(helper);
    builder.emitBlock();
    auto firstParameter = builder.emitParam(sourceType);
    auto secondParameter = builder.emitParam(sourceType);
    auto conditionParameter = builder.emitParam(builder.getBoolType());
    IRInst* selectOperands[] = {conditionParameter, firstParameter, secondParameter};
    auto selection = builder.emitIntrinsicInst(sourceType, kIROp_Select, 3, selectOperands);
    builder.emitReturn(selection);

    builder.setInsertInto(module);
    auto caller = builder.createFunc();
    caller->setFullType(builder.getFuncType(0, nullptr, builder.getVoidType()));
    builder.setInsertInto(caller);
    builder.emitBlock();
    IRInst* arguments[] = {first, second, condition};
    auto call = builder.emitCallInst(sourceType, helper, 3, arguments);
    IRInst* initializer[] = {call};
    auto aggregate = builder.emitMakeStruct(holder, 1, initializer);
    auto extracted = builder.emitFieldExtract(sourceType, aggregate, fieldKey);
    auto storage = builder.emitVar(sourceType);
    builder.emitStore(storage, extracted);
    auto loaded = builder.emitLoad(storage);
    builder.emitReturn();

    List<PortableRT::Metal::AccelerationStructureUse> uses;
    auto moving = UInt(MetalRTTag::Instancing) | UInt(MetalRTTag::PrimitiveMotion);
    uses.add({loaded, moving, loaded});
    uses.add({independent, UInt(MetalRTTag::Instancing), independent});
    DiagnosticSink sink;
    List<PortableRT::Metal::MetalRTAccelerationStructureResources> resources;
    SLANG_CHECK_ABORT(
        PortableRT::Metal::physicalizeAccelerationStructures(module, uses, &sink, resources));
    SLANG_CHECK(sink.getErrorCount() == 0);
    auto physicalType = loaded->getDataType();
    SLANG_CHECK(physicalType != sourceType);
    SLANG_CHECK(physicalType->getOp() == kIROp_ParameterBlockType);
    bool foundMovingResources = false;
    for (const auto& resource : resources)
    {
        if (resource.physicalParameterBlockType != physicalType)
            continue;
        foundMovingResources = true;
        auto nativeAS = resource.nativeASField->getFieldType();
        SLANG_CHECK(nativeAS->getOp() == kIROp_RaytracingAccelerationStructureType);
        SLANG_CHECK(nativeAS->getOperandCount() == 2);
        SLANG_CHECK(getIntVal(nativeAS->getOperand(1)) == moving);
        auto metadata = as<IRPtrTypeBase>(resource.metadataField->getFieldType());
        SLANG_CHECK_ABORT(metadata);
        SLANG_CHECK(metadata->getValueType() == builder.getUIntType());
    }
    SLANG_CHECK(foundMovingResources);
    SLANG_CHECK(first->getDataType() == physicalType);
    SLANG_CHECK(second->getDataType() == physicalType);
    SLANG_CHECK(firstParameter->getDataType() == physicalType);
    SLANG_CHECK(secondParameter->getDataType() == physicalType);
    SLANG_CHECK(selection->getDataType() == physicalType);
    SLANG_CHECK(call->getDataType() == physicalType);
    SLANG_CHECK(extracted->getDataType() == physicalType);
    SLANG_CHECK(field->getFieldType() == physicalType);
    SLANG_CHECK(cast<IRPtrTypeBase>(storage->getDataType())->getValueType() == physicalType);
    SLANG_CHECK(cast<IRFuncType>(helper->getDataType())->getResultType() == physicalType);
    SLANG_CHECK(independent->getDataType() != physicalType);
    SLANG_CHECK(unrelatedField->getFieldType()->getOp() == kIROp_ParameterBlockType);
    SLANG_CHECK(unrelatedField->getFieldType() != physicalType);
    SLANG_CHECK(nativeScene->getDataType() == nativeType);
}

// An array's element and container types must change together. Keep explicit array stride,
// MakeArray/MakeArrayFromElement, and pointer/value element accesses in this unoptimized fixture.
SLANG_UNIT_TEST(metalPortableAccelerationStructureArrays)
{
    StaticUnitTestEnv env(unitTestContext);
    IRFixtureBuilder fixture(env.getSessionImpl());
    auto module = fixture.getModule();
    IRBuilder builder(module);
    builder.setInsertInto(module);
    IRInst* topology = builder.getIntValue(builder.getIntType(), 0);
    auto sourceType = builder.getType(kIROp_RaytracingAccelerationStructureType, 1, &topology);
    auto count = builder.getIntValue(builder.getIntType(), 2);
    auto stride = builder.getIntValue(builder.getIntType(), 16);
    auto arrayType = builder.getArrayType(sourceType, count, stride);
    auto first = builder.createGlobalParam(sourceType);
    auto second = builder.createGlobalParam(sourceType);
    auto independent = builder.createGlobalParam(arrayType);
    auto caller = builder.createFunc();
    caller->setFullType(builder.getFuncType(0, nullptr, builder.getVoidType()));
    builder.setInsertInto(caller);
    builder.emitBlock();
    IRInst* initializers[] = {first, second};
    auto aggregate = builder.emitMakeArray(arrayType, 2, initializers);
    auto repeated = builder.emitMakeArrayFromElement(arrayType, first);
    auto repeatedElement = builder.emitElementExtract(
        sourceType,
        repeated,
        builder.getIntValue(builder.getIntType(), 0));
    auto storage = builder.emitVar(arrayType);
    builder.emitStore(storage, aggregate);
    auto address = builder.emitElementAddress(storage, IRIntegerValue(1));
    auto loaded = builder.emitLoad(address);
    builder.emitReturn();

    auto moving = UInt(MetalRTTag::Instancing) | UInt(MetalRTTag::PrimitiveMotion);
    List<PortableRT::Metal::AccelerationStructureUse> uses;
    uses.add({loaded, moving, loaded});
    DiagnosticSink sink;
    List<PortableRT::Metal::MetalRTAccelerationStructureResources> resources;
    SLANG_CHECK_ABORT(
        PortableRT::Metal::physicalizeAccelerationStructures(module, uses, &sink, resources));
    SLANG_CHECK(sink.getErrorCount() == 0);
    auto physicalType = loaded->getDataType();
    auto physicalArray = cast<IRArrayType>(aggregate->getDataType());
    SLANG_CHECK(physicalArray != arrayType);
    SLANG_CHECK(physicalArray->getElementType() == physicalType);
    SLANG_CHECK(physicalArray->getElementCount() == count);
    SLANG_CHECK(physicalArray->getArrayStride() == stride);
    SLANG_CHECK(first->getDataType() == physicalType);
    SLANG_CHECK(second->getDataType() == physicalType);
    SLANG_CHECK(repeated->getDataType() == physicalArray);
    SLANG_CHECK(repeatedElement->getDataType() == physicalType);
    SLANG_CHECK(cast<IRPtrTypeBase>(storage->getDataType())->getValueType() == physicalArray);
    SLANG_CHECK(cast<IRPtrTypeBase>(address->getDataType())->getValueType() == physicalType);
    auto independentArray = cast<IRArrayType>(independent->getDataType());
    SLANG_CHECK(independentArray->getElementType()->getOp() == kIROp_ParameterBlockType);
    SLANG_CHECK(independentArray->getElementType() != physicalType);
    SLANG_CHECK(independentArray->getElementCount() == count);
    SLANG_CHECK(independentArray->getArrayStride() == stride);
}

// One bound structure cannot serve traces that require two incompatible native motion types.
SLANG_UNIT_TEST(metalPortableAccelerationStructureMotionConflict)
{
    StaticUnitTestEnv env(unitTestContext);
    IRFixtureBuilder fixture(env.getSessionImpl());
    auto module = fixture.getModule();
    IRBuilder builder(module);
    builder.setInsertInto(module);
    IRInst* topology = builder.getIntValue(builder.getIntType(), 0);
    auto sourceType = builder.getType(kIROp_RaytracingAccelerationStructureType, 1, &topology);
    auto first = builder.createGlobalParam(sourceType);
    List<PortableRT::Metal::AccelerationStructureUse> uses;
    uses.add({first, UInt(MetalRTTag::Instancing), first});
    uses.add({first, UInt(MetalRTTag::Instancing) | UInt(MetalRTTag::PrimitiveMotion), first});
    DiagnosticSink sink;
    List<PortableRT::Metal::MetalRTAccelerationStructureResources> resources;
    SLANG_CHECK(
        !PortableRT::Metal::physicalizeAccelerationStructures(module, uses, &sink, resources));
    SLANG_CHECK(sink.getErrorCount() == 1);
}

// Loading a descriptor handle must recover the whole portable binding. Its stored resource type
// needs the same motion signature as the trace, including when the handle came from a resource.
SLANG_UNIT_TEST(metalPortableAccelerationStructureHandles)
{
    StaticUnitTestEnv env(unitTestContext);
    IRFixtureBuilder fixture(env.getSessionImpl());
    auto module = fixture.getModule();
    IRBuilder builder(module);
    builder.setInsertInto(module);
    IRInst* topology = builder.getIntValue(builder.getIntType(), 0);
    auto sourceType = builder.getType(kIROp_RaytracingAccelerationStructureType, 1, &topology);
    IRInst* resourceType = sourceType;
    auto handleType = builder.getType(kIROp_DescriptorHandleType, 1, &resourceType);
    auto first = builder.createGlobalParam(sourceType);
    auto caller = builder.createFunc();
    caller->setFullType(builder.getFuncType(0, nullptr, builder.getVoidType()));
    builder.setInsertInto(caller);
    builder.emitBlock();
    IRInst* resource = first;
    auto handle =
        builder.emitIntrinsicInst(handleType, kIROp_CastResourceToDescriptorHandle, 1, &resource);
    auto storage = builder.emitVar(handleType);
    builder.emitStore(storage, handle);
    IRInst* loadedHandle = builder.emitLoad(storage);
    auto loadedResource = builder.emitIntrinsicInst(
        sourceType,
        kIROp_CastDescriptorHandleToResource,
        1,
        &loadedHandle);
    builder.emitReturn();

    auto moving = UInt(MetalRTTag::Instancing) | UInt(MetalRTTag::PrimitiveMotion);
    List<PortableRT::Metal::AccelerationStructureUse> uses;
    uses.add({loadedResource, moving, loadedResource});
    DiagnosticSink sink;
    List<PortableRT::Metal::MetalRTAccelerationStructureResources> resources;
    SLANG_CHECK_ABORT(
        PortableRT::Metal::physicalizeAccelerationStructures(module, uses, &sink, resources));
    SLANG_CHECK(sink.getErrorCount() == 0);
    auto physicalType = loadedResource->getDataType();
    SLANG_CHECK(physicalType->getOp() == kIROp_ParameterBlockType);
    SLANG_CHECK(first->getDataType() == physicalType);
    auto physicalHandle = cast<IRDescriptorHandleType>(loadedHandle->getDataType());
    SLANG_CHECK(physicalHandle->getResourceType() == physicalType);
    SLANG_CHECK(handle->getDataType() == physicalHandle);
    SLANG_CHECK(cast<IRPtrTypeBase>(storage->getDataType())->getValueType() == physicalHandle);
}
