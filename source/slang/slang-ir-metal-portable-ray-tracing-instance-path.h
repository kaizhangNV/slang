#pragma once

#include "slang-ir.h"

namespace Slang
{
namespace PortableRT
{
namespace Metal
{

enum class InstancePathOperation
{
    Count,
    Element,
    Empty,
};

/// Creates an intrinsic helper for the canonical Metal array-view type. Consumers may retain
/// the helper locally; this is the single mapping from path operations to their native spelling.
IRFunc* createInstancePathOperation(IRModule* module, InstancePathOperation operation);

/// Looks up the selected AS's contribution for an instance index or a complete multilevel path.
/// Callers cache the helper by its canonical parameter type so candidate and committed-hit
/// dispatch interpret the same metadata buffer. The pointer addresses the flat array or trie
/// directly; no program-record directory is involved.
IRFunc* createInstanceContribution(IRModule* module, IRType* instancePathType);

} // namespace Metal
} // namespace PortableRT
} // namespace Slang
