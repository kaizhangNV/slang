//TEST_IGNORE_FILE:

module descriptor_library;

import slang.raytracing.portable;

namespace descriptors
{
// Serializing an application helper must preserve its references to the standard RT type and
// the generic schema requirement, without redeclaring the compiler's descriptor type.
public rt::TraceProgramDescriptor<Program, N, Layout> identity<
    Program : rt::ITraceProgramSchema, let N : int, let Layout : rt::HitRecordLayout>(
    rt::TraceProgramDescriptor<Program, N, Layout> value)
{
    return value;
}
}
