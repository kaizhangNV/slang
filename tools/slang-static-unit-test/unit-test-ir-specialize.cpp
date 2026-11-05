#include "slang/slang-ir-specialize.h"
#include "slang/slang-ir-translate.h"
#include "static-unit-test-env.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

// Decorations may reference later instructions in the same generic block. Specialization
// must substitute those references even when the decorated hoistable type is cloned first.
SLANG_UNIT_TEST(irSpecializationRemapsForwardDecorationOperands)
{
    StaticUnitTestEnv env(unitTestContext);
    for (bool preexistingResult : {false, true})
    {
        IRFixtureBuilder fixture(env.getSessionImpl());
        initializeTranslationDictionary(fixture.getModule());
        IRBuilder builder(fixture.getModule());
        builder.setInsertInto(fixture.getModule());
        auto intType = builder.getIntType();
        auto two = builder.getIntValue(intType, 2);
        IRPtrType* existingPointer = preexistingResult ? builder.getPtrType(intType) : nullptr;

        auto generic = builder.emitGeneric();
        generic->setFullType(builder.getGenericKind());
        builder.setInsertInto(generic);
        builder.emitBlock();
        auto typeParameter = builder.emitParam(builder.getTypeKind());
        IRInst* pointerOperands[] = {typeParameter};
        auto genericPointer = builder.getType(kIROp_PtrType, 1, pointerOperands);
        IRInst* arrayOperands[] = {typeParameter, two};
        auto genericArray = builder.getType(kIROp_ArrayType, 2, arrayOperands);
        // Both types depend only on T, so either sibling order is valid. Pin the order
        // that exposes the forward decoration reference rather than ordinary SSA use order.
        genericArray->insertAfter(genericPointer);
        builder.addDependsOnDecoration(genericPointer, genericArray);
        builder.emitReturn(genericPointer);
        SLANG_CHECK_ABORT(genericPointer->getNextInst() == genericArray);

        builder.setInsertInto(fixture.getModule());
        IRInst* arguments[] = {intType};
        auto specialization = cast<IRSpecialize>(
            builder.emitSpecializeInst(builder.getTypeKind(), generic, 1, arguments));
        auto result = as<IRPtrType>(specializeGeneric(specialization));
        SLANG_CHECK_ABORT(result);
        SLANG_CHECK(result->getValueType() == intType);
        if (existingPointer)
            SLANG_CHECK(result == existingPointer);

        auto dependency = result->findDecoration<IRDependsOnDecoration>();
        SLANG_CHECK_ABORT(dependency);
        auto specializedArray = as<IRArrayType>(dependency->getOperand(0));
        SLANG_CHECK_ABORT(specializedArray);
        SLANG_CHECK(specializedArray != genericArray);
        SLANG_CHECK(specializedArray->getElementType() == intType);
        SLANG_CHECK(specializedArray == builder.getArrayType(intType, two));
        SLANG_CHECK(
            genericPointer->findDecoration<IRDependsOnDecoration>()->getOperand(0) == genericArray);

        // Reusing the specialization must keep the same canonical result and dependency,
        // including when the first specialization reused an already-existing pointer type.
        SLANG_CHECK(specializeGeneric(specialization) == result);
        UInt dependencyCount = 0;
        for (auto decoration : result->getDecorations())
            if (as<IRDependsOnDecoration>(decoration))
            {
                ++dependencyCount;
                SLANG_CHECK(decoration->getOperand(0) == specializedArray);
            }
        SLANG_CHECK(dependencyCount == 1);
    }
}
