// slang-ir-call-graph.h
#pragma once

#include "slang-ir-clone.h"
#include "slang-ir-insts.h"

namespace Slang
{

void buildEntryPointReferenceGraph(
    Dictionary<IRInst*, HashSet<IRFunc*>>& referencingEntryPoints,
    IRModule* module);

/// Builds the same reference graph from explicitly selected execution roots. Generated target
/// callbacks can be invoked by the driver without being source shader entry points.
void buildEntryPointReferenceGraph(
    Dictionary<IRInst*, HashSet<IRFunc*>>& referencingEntryPoints,
    ArrayView<IRFunc*> roots);

HashSet<IRFunc*>* getReferencingEntryPoints(
    Dictionary<IRInst*, HashSet<IRFunc*>>& m_referencingEntryPoints,
    IRInst* inst);

} // namespace Slang
