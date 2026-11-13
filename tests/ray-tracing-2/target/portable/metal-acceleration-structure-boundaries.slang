//DIAGNOSTIC_TEST:SIMPLE(diag=CHECK): -experimental-feature -target hlsl -no-codegen
//DIAGNOSTIC_TEST:SIMPLE(diag=CHECK): -experimental-feature -target metal -no-codegen

import slang.raytracing.portable;

void usePortable(rt::AccelerationStructure scene) {}
void useNative(RaytracingAccelerationStructure scene) {}

void test(rt::AccelerationStructure portableScene, RaytracingAccelerationStructure nativeScene)
{
    // The portable binding contains instance contribution metadata on Metal. Passing a native
    // AS would leave that metadata missing; passing its wrapper to a native API is also invalid.
    usePortable(nativeScene);
/*CHECK:
                ^^^^^^^^^^^ type mismatch in expression
                ^^^^^^^^^^^ expected an expression of type 'rt.AccelerationStructure', got 'RaytracingAccelerationStructure'
*/
    useNative(portableScene);
/*CHECK:
              ^^^^^^^^^^^^^ type mismatch in expression
              ^^^^^^^^^^^^^ expected an expression of type 'RaytracingAccelerationStructure', got 'rt.AccelerationStructure'
*/
}
