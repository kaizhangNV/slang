#pragma once

#include "slang-ir.h"

namespace Slang
{

namespace PortableRT
{
/// Returns the decoration on a logical stage-input type, such as MissInput<C>, or null for
/// other types. Also accepts a borrowed input. Before specialization, this can return the
/// generic declaration's decoration, whose operands still depend on C; use it only to identify
/// a logical input until specialization supplies the concrete types.
IRPortableRTStageInputDecoration* getStageInputInfo(IRType* type);

/// Returns the input contract for a portable stage entry, or null for an ordinary entry/function.
/// A portable stage has one logical Input<Context> parameter. That parameter is not a native
/// shader parameter; its target signature is generated after specialization.
IRPortableRTStageInputDecoration* getEntryPointInputInfo(IRFunc* function);

/// Removes a logical stage function's entry marker and denormal execution modes. Use this when
/// the implementation becomes an ordinary helper, including when Metal gives it a visible adapter.
/// The caller handles any keep-alive marker and layout needed by its target.
void demoteStageImplementation(IRFunc* function);

/// Removes remaining entry-point markers and denormal modes from logical stages after selected
/// entries have been replaced with target adapters. Consider compiling CallingMiss and
/// SharedMiss as separate outputs, where CallingMiss calls SharedMiss::invoke. SharedMiss is
/// an ordinary helper in CallingMiss's output, even though the source module also selected it.
void demoteStageImplementations(IRModule* module);
} // namespace PortableRT

/// Threads one typed invocation value through specialized ordinary function calls.
///
/// Consider a static stage method calling a noinline helper which changes a copied input's
/// payload. Both functions must use the same payload reference, not copies of payload storage.
/// This utility adds ordinary parameters to affected helpers and supplies their callers' values.
/// Target adapters bind the invocation's actual value, or override parameter creation to expose
/// it through a native entry-point parameter. No target ABI or input-view representation is
/// imposed on the helpers, so they do not need to inline to access invocation state.
class PortableRTInvocationParameter
{
public:
    PortableRTInvocationParameter(IRModule* module, IRType* valueType, const char* name);
    virtual ~PortableRTInvocationParameter() = default;

    /// Supplies a root's already-created invocation value, such as a loaded shader record or a
    /// reference into Metal ray_data. Bind roots before lowering their reachable operations.
    void bind(IRFunc* function, IRInst* value);

    /// Returns the invocation value in this function, adding a helper parameter and updating
    /// its direct callers when necessary. Call after generic and higher-order specialization.
    IRInst* get(IRFunc* function);

    /// Replaces an invocation-state operation with the value supplied to its containing function.
    void lower(IRInst* operation);

protected:
    /// Adds an ordinary helper parameter. A target implementation may override this for its
    /// adapters, returning the same logical value type even when the native parameter differs.
    virtual IRInst* createParameter(IRFunc* function);

    IRModule* m_module;
    IRType* m_valueType;
    const char* m_name;

private:
    Dictionary<IRFunc*, IRInst*> m_values;
};

} // namespace Slang
