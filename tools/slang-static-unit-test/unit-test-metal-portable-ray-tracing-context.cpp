#include "slang/slang-ir-explicit-global-context.h"
#include "slang/slang-ir-metal-portable-ray-tracing-context.h"
#include "slang/slang-ir-metal-portable-ray-tracing-target.h"
#include "slang/slang-ir-specialize-address-space.h"
#include "slang/slang-ir-util.h"
#include "static-unit-test-env.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

// Builds the bindable resource shape consumed by explicit-global-context lowering, after
// ordinary parameter binding has assigned the source declaration its Metal buffer index.
static IRGlobalParam* createContextTestBuffer(IRBuilder& builder, UInt index)
{
    auto parameter =
        builder.createGlobalParam(builder.getPtrType(builder.getUIntType(), AddressSpace::Global));
    IRTypeLayout::Builder typeLayout(&builder);
    typeLayout.addResourceUsage(LayoutResourceKind::MetalBuffer, LayoutSize(1));
    IRVarLayout::Builder layout(&builder, typeLayout.build());
    layout.findOrAddResourceInfo(LayoutResourceKind::MetalBuffer)->offset = LayoutOffset(index);
    builder.addLayoutDecoration(parameter, layout.build());
    return parameter;
}

// Starts a defined function while preserving its selected physical signature exactly.
static IRFunc* startContextTestFunction(IRBuilder& builder, IRModule* module, IRFuncType* type)
{
    builder.setInsertInto(module);
    auto function = builder.createFunc();
    function->setFullType(type);
    builder.setInsertInto(function);
    builder.emitBlock();
    for (UInt i = 0; i < type->getParamCount(); ++i)
        builder.emitParam(type->getParamType(i));
    return function;
}

static IRParam* getContextTestParameter(IRFunc* function)
{
    for (auto parameter : function->getParams())
        if (parameter->findDecoration<IRExplicitGlobalContextDecoration>())
            return parameter;
    return nullptr;
}

// Models ray generation -> ordinary helper -> callable table, where only one callable reads
// a global. The helper needs the caller's context despite having no global read of its own.
// A second table member must acquire the same erased signature even though it needs no context.
SLANG_UNIT_TEST(metalPortableIndirectGlobalContext)
{
    StaticUnitTestEnv env(unitTestContext);
    IRFixtureBuilder fixture(env.getSessionImpl());
    auto module = fixture.getModule();
    IRBuilder builder(module);
    builder.setInsertInto(module);
    auto buffer = createContextTestBuffer(builder, 3);
    auto erasedPointer = builder.getPtrType(builder.getUInt8Type(), AddressSpace::ThreadLocal);
    IRType* parameters[] = {erasedPointer};
    auto signature = builder.getFuncType(1, parameters, builder.getVoidType());
    auto stage = builder.getIntValue(builder.getIntType(), IRIntegerValue(Stage::Callable));
    IRInst* tableOperands[] = {signature, stage};
    auto table = cast<IRMetalVisibleFunctionTable>(
        builder.getType(kIROp_MetalVisibleFunctionTable, 2, tableOperands));
    auto resources = builder.createStructType();
    auto callableField = builder.createStructField(resources, builder.createStructKey(), table);

    auto reader = startContextTestFunction(builder, module, signature);
    IRInst* visibleOperands[] = {stage, table};
    builder.addDecoration(reader, kIROp_MetalVisibleFunctionDecoration, visibleOperands, 2);
    builder.emitLoad(buffer);
    builder.emitReturn();
    auto empty = startContextTestFunction(builder, module, signature);
    builder.addDecoration(empty, kIROp_MetalVisibleFunctionDecoration, visibleOperands, 2);
    builder.emitReturn();

    auto voidSignature = builder.getFuncType(0, nullptr, builder.getVoidType());
    auto helper = startContextTestFunction(builder, module, voidSignature);
    auto zero = builder.getIntValue(builder.getUIntType(), 0);
    auto falseValue = builder.getBoolValue(false);
    auto index = builder.emitDefaultConstruct(builder.getVectorType(builder.getUIntType(), 3));
    auto data = builder.emitVar(builder.getUIntType());
    auto resourceValue = builder.emitVar(resources);
    auto records = builder.emitDefaultConstruct(
        builder.getPtrType(builder.getUIntType(), AddressSpace::Global));
    IRInst* resourceTypeOperands[] = {resources};
    auto parameterBlock = builder.getType(kIROp_ParameterBlockType, 1, resourceTypeOperands);
    IRInst* callOperands[] = {
        zero,
        data,
        index,
        index,
        falseValue,
        falseValue,
        resourceValue,
        records,
        builder.getIntValue(builder.getUIntType(), kMetalRTRecordHeaderSize),
        falseValue,
        parameterBlock,
        callableField->getKey(),
        falseValue,
        builder.emitDefaultConstruct(erasedPointer)};
    auto dispatch = cast<IRMetalRTCallDispatch>(builder.emitIntrinsicInst(
        builder.getVoidType(),
        kIROp_MetalRTCallDispatch,
        SLANG_COUNT_OF(callOperands),
        callOperands));
    builder.emitReturn();

    auto entry = startContextTestFunction(builder, module, voidSignature);
    builder
        .addEntryPointDecoration(entry, Profile(Stage::Compute), toSlice("main"), toSlice("test"));
    builder.emitCallInst(builder.getVoidType(), helper, 0, nullptr);
    builder.emitReturn();

    introduceExplicitGlobalContext(module, CodeGenTarget::Metal);
    auto helperContext = getContextTestParameter(helper);
    SLANG_CHECK_ABORT(helperContext != nullptr);
    SLANG_CHECK(dispatch->getGlobalContext() == helperContext);
    SLANG_CHECK(getContextTestParameter(entry) == nullptr);
    SLANG_CHECK(getContextTestParameter(reader) != nullptr);
    SLANG_CHECK(getContextTestParameter(empty) == nullptr);

    DiagnosticSink sink;
    finalizeMetalPortableRayTracingGlobalContext(module, &sink);
    SLANG_CHECK(sink.getErrorCount() == 0);
    SLANG_CHECK(cast<IRBoolLit>(dispatch->getHasGlobalContext())->getValue());
    SLANG_CHECK(getContextTestParameter(reader)->getDataType() == erasedPointer);
    SLANG_CHECK(getContextTestParameter(empty)->getDataType() == erasedPointer);
    auto physicalTable = cast<IRMetalVisibleFunctionTable>(callableField->getFieldType());
    SLANG_CHECK(reader->getDataType() == physicalTable->getFunctionType());
    SLANG_CHECK(empty->getDataType() == physicalTable->getFunctionType());
}

// The native IFT callback must receive the reached buffer at its existing binding, not every
// resource in the ray-generation program. It builds its own resource context rather than taking
// a caller context parameter, because traversal invokes it independently through the IFT.
SLANG_UNIT_TEST(metalPortableIntersectionGlobalBuffers)
{
    StaticUnitTestEnv env(unitTestContext);
    IRFixtureBuilder fixture(env.getSessionImpl());
    auto module = fixture.getModule();
    IRBuilder builder(module);
    builder.setInsertInto(module);
    auto usedBuffer = createContextTestBuffer(builder, 3);
    createContextTestBuffer(builder, 7);
    auto signature = builder.getFuncType(0, nullptr, builder.getVoidType());
    auto helper = startContextTestFunction(builder, module, signature);
    builder.emitLoad(usedBuffer);
    builder.emitReturn();
    auto callback = startContextTestFunction(builder, module, signature);
    IRInst* callbackOperands[] = {
        builder.getIntValue(builder.getIntType(), IRIntegerValue(MetalRTGeometryKind::Triangle)),
        builder.getIntValue(builder.getIntType(), UInt(MetalRTTag::Instancing)),
        builder.getIntValue(builder.getIntType(), 0)};
    builder.addDecoration(callback, kIROp_MetalIntersectionFunctionDecoration, callbackOperands, 3);
    builder.emitCallInst(builder.getVoidType(), helper, 0, nullptr);
    builder.emitReturn();

    introduceExplicitGlobalContext(module, CodeGenTarget::Metal);
    SLANG_CHECK(getContextTestParameter(callback) == nullptr);
    SLANG_CHECK(getContextTestParameter(helper) != nullptr);
    SLANG_CHECK(cast<IRFuncType>(callback->getDataType())->getParamCount() == 1);
    auto layout = findVarLayout(callback->getFirstParam());
    SLANG_CHECK_ABORT(layout != nullptr);
    SLANG_CHECK(layout->findOffsetAttr(LayoutResourceKind::MetalBuffer)->getOffset() == 3);
    DiagnosticSink sink;
    finalizeMetalPortableRayTracingGlobalContext(module, &sink);
    SLANG_CHECK(sink.getErrorCount() == 0);
}

// Mutable thread globals are not bindable IFT resources. Allocating a fresh callback-local value
// would silently change their semantics, so retain the context dependency and diagnose it.
SLANG_UNIT_TEST(metalPortableIntersectionMutableGlobalDiagnostic)
{
    StaticUnitTestEnv env(unitTestContext);
    IRFixtureBuilder fixture(env.getSessionImpl());
    auto module = fixture.getModule();
    IRBuilder builder(module);
    builder.setInsertInto(module);
    auto variable = builder.createGlobalVar(builder.getUIntType());
    auto callback = startContextTestFunction(
        builder,
        module,
        builder.getFuncType(0, nullptr, builder.getVoidType()));
    auto zero = builder.getIntValue(builder.getIntType(), 0);
    IRInst* operands[] = {zero, zero, zero};
    builder.addDecoration(callback, kIROp_MetalIntersectionFunctionDecoration, operands, 3);
    builder.emitLoad(variable);
    builder.emitReturn();
    introduceExplicitGlobalContext(module, CodeGenTarget::Metal);
    SLANG_CHECK(getContextTestParameter(callback) != nullptr);
    DiagnosticSink sink;
    finalizeMetalPortableRayTracingGlobalContext(module, &sink);
    SLANG_CHECK(sink.getErrorCount() == 1);
}

// Table functions have no source entry-point decoration. Nevertheless their device buffer
// parameters must specialize a reachable ordinary helper's generic pointer to the device space.
SLANG_UNIT_TEST(metalPortableTableAddressSpaceRoots)
{
    StaticUnitTestEnv env(unitTestContext);
    IRFixtureBuilder fixture(env.getSessionImpl());
    auto module = fixture.getModule();
    IRBuilder builder(module);
    builder.setInsertInto(module);
    auto genericPointer = builder.getPtrType(builder.getUIntType(), AddressSpace::Generic);
    auto devicePointer = builder.getPtrType(builder.getUIntType(), AddressSpace::Global);
    IRType* genericParameters[] = {genericPointer};
    IRType* deviceParameters[] = {devicePointer};
    auto genericSignature = builder.getFuncType(1, genericParameters, builder.getVoidType());
    auto deviceSignature = builder.getFuncType(1, deviceParameters, builder.getVoidType());
    auto helper = startContextTestFunction(builder, module, genericSignature);
    builder.emitLoad(helper->getFirstParam());
    builder.emitReturn();

    auto visible = startContextTestFunction(builder, module, deviceSignature);
    auto stage = builder.getIntValue(builder.getIntType(), IRIntegerValue(Stage::Miss));
    IRInst* tableOperands[] = {deviceSignature, stage};
    auto table = builder.getType(kIROp_MetalVisibleFunctionTable, 2, tableOperands);
    IRInst* visibleOperands[] = {stage, table};
    builder.addDecoration(visible, kIROp_MetalVisibleFunctionDecoration, visibleOperands, 2);
    IRInst* visibleArguments[] = {visible->getFirstParam()};
    builder.emitCallInst(builder.getVoidType(), helper, 1, visibleArguments);
    builder.emitReturn();

    auto intersection = startContextTestFunction(builder, module, deviceSignature);
    auto zero = builder.getIntValue(builder.getIntType(), 0);
    IRInst* intersectionOperands[] = {zero, zero, zero};
    builder.addDecoration(
        intersection,
        kIROp_MetalIntersectionFunctionDecoration,
        intersectionOperands,
        3);
    IRInst* intersectionArguments[] = {intersection->getFirstParam()};
    builder.emitCallInst(builder.getVoidType(), helper, 1, intersectionArguments);
    builder.emitReturn();

    NoOpInitialAddressSpaceAssigner assigner;
    specializeAddressSpace(module, &assigner);
    IRFunc* roots[] = {visible, intersection};
    for (auto root : roots)
    {
        auto call = cast<IRCall>(root->getFirstBlock()->getFirstOrdinaryInst());
        auto specializedHelper = cast<IRFunc>(call->getCallee());
        SLANG_CHECK(specializedHelper->getFirstParam()->getDataType() == devicePointer);
        SLANG_CHECK(!root->findDecoration<IREntryPointDecoration>());
    }
}
