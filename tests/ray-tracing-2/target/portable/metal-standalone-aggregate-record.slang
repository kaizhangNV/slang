//TEST:SIMPLE(filecheck=TUPLE): -experimental-feature -entry TupleMiss -stage miss -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=MATRIX): -experimental-feature -entry MatrixMiss -stage miss -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}

struct Context<RecordType> : rt::IPayloadContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = RecordType;
}

// Independently selected adapters receive record bytes through the fixed device pointer.
// Restoring the concrete tuple or matrix layout must produce the read-only local reference
// expected by the ordinary helper after aggregate legalization.
[noinline]
void readTuple(rt::MissInput<Context<Tuple<uint, uint>>> input)
{
    input.payload += input.record._0;
}

[noinline]
void readMatrix(rt::MissInput<Context<uint2x2>> input)
{
    input.payload += input.record[0][1];
}

struct TupleMiss : rt::IMissShader
{
    typealias Context = ::Context<Tuple<uint, uint>>;
    static void invoke(rt::MissInput<Context> input) { readTuple(input); }
}

struct MatrixMiss : rt::IMissShader
{
    typealias Context = ::Context<uint2x2>;
    static void invoke(rt::MissInput<Context> input) { readMatrix(input); }
}

// TUPLE: readTuple
// TUPLE: visible]] void TupleMiss_miss(MissInvocationContext_{{[0-9]+}} thread* [[CONTEXT:invocation_[0-9]+]])
// TUPLE: *(Tuple_{{[0-9]+}} device*)((ulong)([[CONTEXT]]->recordData_{{[0-9]+}}))
// TUPLE-NOT: kernel]]
// MATRIX: readMatrix
// MATRIX: visible]] void MatrixMiss_miss(MissInvocationContext_{{[0-9]+}} thread* [[CONTEXT:invocation_[0-9]+]])
// MATRIX: [[CONTEXT]]->recordData_{{[0-9]+}}
// MATRIX-NOT: kernel]]
