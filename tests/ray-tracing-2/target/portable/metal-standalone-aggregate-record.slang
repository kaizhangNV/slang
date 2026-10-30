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

// These exported helpers have no caller to infer a parameter address space from. The wrapper
// must choose a concrete read-only borrow even before tuple/matrix legalization turns the
// source record into the struct/array representation that normal constref lowering sees.
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
// TUPLE: void TupleMiss_miss(
// TUPLE-NOT: kernel void
// MATRIX: readMatrix
// MATRIX: void MatrixMiss_miss(
// MATRIX-NOT: kernel void
