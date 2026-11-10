#pragma once

#include "slang-ast-support-types.h"
#include "slang-compiler-fwd.h"
#include "slang-profile.h"

namespace Slang
{

class InterfaceDecl;
class FunctionDeclBase;
class AggTypeDecl;
class Decl;
class Type;
class ASTBuilder;
class MagicTypeModifier;
class TraceProgramDescriptorType;
struct SemanticsVisitor;
class EntryPoint;
struct IRBuilder;
struct IRInst;
struct IRType;

namespace PortableRT
{

// These helpers inspect already-checked declarations, types, and expressions. The checks taking
// a visitor use its existing diagnoseOnce state to avoid reporting the same error twice.

/// Checks the additional stage-implementation restrictions after ordinary conformance checking.
void checkStageRestrictions(
    SemanticsVisitor* visitor,
    DeclRef<InterfaceDecl> superInterfaceDeclRef,
    WitnessTable* witnessTable,
    SourceLoc conformanceLoc);

/// Diagnoses repeated program types in a checked schema section.
void diagnoseDuplicateSchemaEntries(
    ASTBuilder* astBuilder,
    DiagnosticSink* sink,
    Type* entryListType,
    AssocTypeDecl* associatedTypeRequirement,
    Type* schemaType,
    SourceLoc satisfyingDeclLoc);

/// Checks whether the variable's storage, parameter direction, and initializer allow an RT input.
void diagnoseInvalidVariableType(DiagnosticSink* sink, VarDeclBase* varDecl);

/// Diagnoses restricted RT types used as function results.
void diagnoseInvalidReturnType(DiagnosticSink* sink, CallableDecl* decl);

/// Diagnoses restricted RT types used as property results.
void diagnoseInvalidPropertyType(DiagnosticSink* sink, PropertyDecl* decl);

/// Allows the ray-tracing stage-interface and descriptor AST types in imported library
/// declarations.
bool isBuiltinTypeModifier(MagicTypeModifier* modifier, Decl* decl);

/// Returns the executable stage represented by the interface's AST type, or Stage::Unknown.
Stage getStageKind(InterfaceDecl* interfaceDecl);

/// Returns whether the interface represents an executable stage with an `invoke` requirement.
bool isExecutableStageInterface(InterfaceDecl* interfaceDecl);

/// Returns the stage named by the input struct's KnownBuiltin annotation, or Stage::Unknown.
Stage getStageInputKind(AggTypeDecl* typeDecl);

/// Returns the checked Context argument of a recognized stage-input type, or null otherwise.
Type* getStageInputContextType(Type* inputType);

/// Looks up a compiler-consumed associated-type role through canonical checked inheritance.
/// The result may remain dependent on a generic or extern context until ordinary linking.
Type* getAssociatedType(
    Linkage* linkage,
    ASTBuilder* astBuilder,
    Type* owner,
    KnownBuiltinDeclName role);

/// Returns the concrete type pack of a checked shader list, or null for an unresolved list.
/// A non-null pack with no elements represents an actual empty list.
ConcreteTypePack* getConcreteEntryTypes(Type* entryListType);

/// Adds PortableRTStageInputDecoration to the IR struct type for a recognized stage input.
/// Its six operands are the stage kind, acceleration-structure type, payload type (callable
/// data for CallableInput), record type, primitive type, and attributes type. Non-hit inputs use
/// void for the last two operands. Other aggregate declarations are unchanged.
/// The callback lowers checked semantic types synchronously in the caller's existing generic
/// environment. The builder uses that environment's insertion location.
void addStageInputDecoration(
    IRBuilder* builder,
    Linkage* linkage,
    ASTBuilder* astBuilder,
    AggTypeDecl* decl,
    IRType* irType,
    IRType* (*lowerType)(Type*, void*),
    void* userData);

/// Adds PortableRTShaderListDecoration to a recognized shader-list type, preserving its ordered
/// entry-type pack followed by record, payload, primitive, attributes, closest-hit, any-hit, and
/// intersection type packs. Each computed pack uses ordinary each/expand and associated-type
/// lowering in the list's generic environment.
/// Miss and callable lists retain record types; the other computed packs are empty for them.
/// Other aggregate declarations are unchanged. No lookup requests an invoke implementation.
void addShaderListDecoration(
    IRBuilder* builder,
    Linkage* linkage,
    ASTBuilder* astBuilder,
    AggTypeDecl* decl,
    IRType* irType,
    IRType* (*lowerType)(Type*, void*),
    void* userData);

/// Appends the descriptor schema's acceleration-structure, motion, and three list types to
/// operands, which already contains the lowered schema and its conformance witness. The callback
/// lowers each checked associated type in the descriptor's generic environment. Construct the IR
/// descriptor after this call so its ordinary operands retain all seven values from the start.
void appendTraceProgramDescriptorOperands(
    Linkage* linkage,
    ASTBuilder* astBuilder,
    TraceProgramDescriptorType* descriptorType,
    List<IRInst*>& operands,
    IRType* (*lowerType)(Type*, void*),
    void* userData);

/// Returns this executable interface's `invoke` requirement, or null for other interfaces.
FunctionDeclBase* getStageInvokeRequirement(InterfaceDecl* interfaceDecl);

/// Returns the type name used to form a structural ray-tracing entry-point symbol.
///
/// Nongeneric types use the source declaration path, including namespaces and enclosing types
/// but excluding module names. Generic specializations use ordinary type mangling to preserve
/// the selected arguments.
String getSourceTypeName(ASTBuilder* astBuilder, Type* stageType);

/// Returns the deterministic target symbol for a structural ray-tracing type and selected stage.
///
/// Dots in the source type name become underscores. The canonical stage name is appended so a
/// struct implementing multiple stages gets distinct target symbols while retaining its
/// source-level lookup name.
String getEntryPointName(UnownedStringSlice sourceTypeName, Stage stage);


/// Selects the checked stage implementation named by a stage type or its invoke member.
DeclRef<FuncDecl> findEntryPointByName(
    Linkage* linkage,
    Module* module,
    Name* name,
    Profile& ioProfile,
    DiagnosticSink* sink,
    bool* outFoundStruct,
    Type** outStageType);

/// Diagnoses stage inputs exposed by a selected native entry-point signature.
void diagnoseInputInNativeEntryPoint(EntryPoint* entryPoint, DiagnosticSink* sink);

/// Sets the public and target names for an entry point selected by its source stage type.
void initializeEntryPoint(EntryPoint* entryPoint, Name* entryPointName, Type* stageType);

} // namespace PortableRT
} // namespace Slang
