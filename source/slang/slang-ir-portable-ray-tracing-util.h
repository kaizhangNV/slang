#pragma once

#include "slang-ir.h"

namespace Slang
{

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
