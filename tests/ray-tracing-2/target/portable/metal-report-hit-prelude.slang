//TEST:SIMPLE(filecheck=EMPTY): -experimental-feature -entry EmptyIntersection -stage intersection -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=TUPLE): -experimental-feature -entry TupleIntersection -stage intersection -target metal -profile metallib_2_4 -validate-ir
//TEST:SIMPLE(filecheck=MATRIX): -experimental-feature -entry MatrixIntersection -stage intersection -target metal -profile metallib_2_4 -validate-ir

import slang.raytracing.portable;

struct EmptyAttributes {}
struct TraceContext : rt::ITraceContext
{
    typealias AccelerationStructure = rt::AccelerationStructure;
    typealias Motion = rt::NoMotion;
}
struct HitContext<A> : rt::IHitContext
{
    typealias TraceContext = ::TraceContext;
    typealias Payload = uint;
    typealias Record = void;
    typealias Primitive = rt::BoundingBoxPrimitive<A>;
}

// EndSearch must cross both helpers before reaching the Intersection boundary. The payload
// increments after reportHit must be skipped, even though the prelude uses an ordinary return.
[noinline]
bool reportFromHelper<A>(rt::IntersectionInput<HitContext<A>> input, A attributes)
{
    bool accepted = input.reportHit(2.0, 7, attributes);
    input.payload += 1;
    return accepted;
}
[noinline]
void reportAndContinue<A>(rt::IntersectionInput<HitContext<A>> input, A attributes)
{
    if (reportFromHelper(input, attributes))
        input.payload += 10;
}

struct EmptyIntersection : rt::IIntersectionShader
{
    typealias Context = HitContext<EmptyAttributes>;
    static void invoke(rt::IntersectionInput<Context> input)
    {
        reportAndContinue(input, EmptyAttributes());
        input.payload += 100;
    }
}
struct TupleIntersection : rt::IIntersectionShader
{
    typealias Context = HitContext<Tuple<uint, float>>;
    static void invoke(rt::IntersectionInput<Context> input)
    {
        reportAndContinue(input, makeTuple(123u, 4.5f));
        input.payload += 100;
    }
}
struct MatrixIntersection : rt::IIntersectionShader
{
    typealias Context = HitContext<uint2x2>;
    static void invoke(rt::IntersectionInput<Context> input)
    {
        reportAndContinue(input, uint2x2(1, 2, 3, 4));
        input.payload += 100;
    }
}

// The prelude is a template, so it can precede the generated state and attribute declarations.
// Empty attributes pass no address to dereference; tuple/matrix attributes use their emitted
// Metal storage type. The report wrapper still produces a result carrying an exit decision.
// EMPTY: inline uint _slang_metalReportHit(
// EMPTY: struct PortableRTCandidateState_
// EMPTY: portableRTReportHit_
// EMPTY: _slang_metalReportHit<void>({{.*}}nullptr
// EMPTY: if({{.*}} == 2U)
// EMPTY: return
// EMPTY: ResultType_{{[0-9]+}} reportFromHelper_
// EMPTY: ResultType_{{[0-9]+}} [[REPORT:_S[0-9]+]] = rt_IntersectionInput_reportHit_
// EMPTY: if([[REPORT]].tag_{{[0-9]+}})
// EMPTY: return
// EMPTY: + 1U;
// EMPTY: ResultType_{{[0-9]+}} reportAndContinue_
// EMPTY: ResultType_{{[0-9]+}} [[HELPER:_S[0-9]+]] = reportFromHelper_
// EMPTY: if([[HELPER]].tag_{{[0-9]+}})
// EMPTY: return
// EMPTY: + 10U;
// EMPTY: EmptyIntersection_intersection(
// TUPLE: inline uint _slang_metalReportHit(
// TUPLE: portableRTReportHit_
// TUPLE: Tuple_{{[0-9]+}} [[ATTRIBUTES:_S[0-9]+]]
// TUPLE: _slang_metalReportHit<Tuple_{{[0-9]+}}>({{.*}}&[[ATTRIBUTES]])
// TUPLE: TupleIntersection_intersection(
// MATRIX: inline uint _slang_metalReportHit(
// MATRIX: portableRTReportHit_
// MATRIX: _slang_metalReportHit<
// MATRIX: MatrixIntersection_intersection(
