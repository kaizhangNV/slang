# Callable Shader Contract

Callable shaders use the same structural program layout as hit and miss groups, but form an
independent dynamically indexed table. They are not part of traversal hit selection.

## Source Model

A callable context fixes the trace context, callable-data ABI, and record type:

```slang
struct MaterialCallableContext : rt::ICallableGroupContext
{
    typealias TraceContext = PrimaryTraceContext;
    typealias CallableData = MaterialCallData;
    typealias Record = MaterialRecord;
}
```

The source stage and group are structural declarations:

```slang
struct ShadeMaterial : rt::ICallableShader<MaterialCallableContext>
{
    void invoke(rt::CallableInput<MaterialCallableContext> input)
    {
        input.data.result *= input.record.factor;
    }
}

struct MaterialCallableGroup : rt::ICallableGroup
{
    typealias Slot = rt::CallableSlot<0>;
    typealias Context = MaterialCallableContext;
    typealias Callable = ShadeMaterial;
}
```

Add the group to `ITraceProgramLayout.CallableGroups`, then invoke a dynamically selected slot:

```slang
rt::RayTracer<ProgramLayout> tracer;
tracer.callShader<MaterialCallableContext>(index, descriptor, data);
```

## Type Rule

The callable index may be dynamic, so every callable group in one selected layout must use the same
`CallableData` ABI. Slang diagnoses incompatible callable-data types while canonicalizing the
layout. Record types may differ because each slot resolves its own record.

This rule provides type safety for the table operation. It cannot prove that a runtime index is in
range or points at the material intended by application data; those remain host/data validation
responsibilities.

## Target Mapping

| Target | Lowering                                                                       |
| ------ | ------------------------------------------------------------------------------ |
| D3D12  | Native callable entry points, callable SBT records, and `CallShader`           |
| Vulkan | Native callable entry points, callable SBT records, and `OpExecuteCallableKHR` |
| Metal  | A typed visible-function table plus generated record-buffer lookup             |

Metal threads the descriptor resources and record buffer through each visible callable function so
nested callable dispatch uses the same program tables. All functions in one table receive a uniform
physical signature, including functions that do not read every argument.

Callable dispatch is legal only in logical stages whose target capability permits it. The compiler
checks the source stage before generating target adapters, so a generated Metal helper cannot make
an otherwise illegal source call valid.

## Practical Guidance

Use an ordinary device function when the callee is statically known. Use a callable shader only
when shader code must select the callee from a runtime SBT slot. Callable stages have pipeline,
stack, and scheduling costs even when a target compiler can inline or fuse part of the adapter.

The current implementation covers non-empty callable data, per-slot records, nested dispatch,
native D3D/Vulkan lowering, and Metal VFT lowering. Empty callable-data structs remain subject to
the independent legacy ABI bug tracked by
[#12718](https://github.com/shader-slang/slang/issues/12718).
