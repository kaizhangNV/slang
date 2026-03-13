// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Simple SINGLE LAYER backward pass benchmark with tiled (OptimalLayout) weights.
// NOTE: The backward path for outerProductAccumulate tiled layout is not fully
// wired up yet, so this benchmark exercises the forward tiled layout path
// and whatever backward path the autodiff currently generates.

import slang.neural;

// Network configuration - single layer: INPUT_SIZE -> OUTPUT_SIZE
#ifndef INPUT_SIZE
#define INPUT_SIZE 64
#endif
#ifndef OUTPUT_SIZE
#define OUTPUT_SIZE 16
#endif

// CoopMat configuration
static const int SubgroupSize = 32;
#ifndef SUBGROUP_COUNT
#define SUBGROUP_COUNT 1
#endif
static const int SubgroupCount = SUBGROUP_COUNT;
static const int BatchSize = SubgroupSize * SubgroupCount;

// Use bindless address
typealias Address = BindlessAddress<half>;

// Tiled layout: weight matrix is OUTPUT_SIZE x INPUT_SIZE, stored as
// ceil(OUTPUT_SIZE/16) * ceil(INPUT_SIZE/16) tiles of 16x16 (256 elements each).
static const int NTileRows = (OUTPUT_SIZE + 15) / 16;
static const int NTileCols = (INPUT_SIZE + 15) / 16;
static const int TILED_WEIGHT_COUNT = NTileRows * NTileCols * 256;

// Bias is stored after the tiled weight data.
static const int WEIGHT_OFFSET = 0;
static const int BIAS_OFFSET = TILED_WEIGHT_COUNT;
static const int PARAM_COUNT = TILED_WEIGHT_COUNT + OUTPUT_SIZE;

// Query functions
public int get_input_size() { return INPUT_SIZE; }
public int get_output_size() { return OUTPUT_SIZE; }
public int get_total_params() { return PARAM_COUNT; }
public int get_tiled_weight_count() { return TILED_WEIGHT_COUNT; }

// Shared memory size for single layer.
// Use 48KB to fit all tiles at once in sumReduceTiles (avoids row-by-row syncs).
public struct LargeShMem : ISharedMemorySize { public static const uint Bytes = 40 * 1024; }
typealias ShMemSizeLayer = LargeShMem;

// Define vector types
typealias VInput = WaveTangledVector<half, ShMemSizeLayer, INPUT_SIZE, SubgroupSize, SubgroupCount>;
typealias VOutput = WaveTangledVector<half, ShMemSizeLayer, OUTPUT_SIZE, SubgroupSize, SubgroupCount>;

[Differentiable]
VOutput MatVecMul(Address paramAddr, in VInput input)
{
    let biasAddr = paramAddr.getOffset(TILED_WEIGHT_COUNT);
    return input.linearTransform<Address, OptimalLayout, VOutput>(paramAddr, biasAddr);
}

// Compute shader entry point
[shader("compute")]
#if SUBGROUP_COUNT == 2
[numthreads(64, 1, 1)]
#else
[numthreads(32, 1, 1)]
#endif
void compute_single_layer_backward_tiled(
    uint3 gid : SV_GroupID,
    uint gtid : SV_GroupIndex,
    uniform RWStructuredBuffer<half>.Handle params,
    uniform StructuredBuffer<half> inputs,
    uniform StructuredBuffer<half> grad_outputs,
    uniform RWStructuredBuffer<half>.Handle grad_params,
    uniform RWStructuredBuffer<half> grad_inputs,
    uniform int batch_size)
{
#if SUBGROUP_COUNT == 2
    int sampleIdx = int(gid.x) * 2 + int(gtid / 32);
#else
    int sampleIdx = int(gid.x);
#endif
    if (sampleIdx >= batch_size)
        return;

    let baseAddr = Address(params);
    let gradBaseAddr = Address(grad_params);

    // Load input
    half inputArr[INPUT_SIZE];
    int inBaseIdx = sampleIdx * INPUT_SIZE;
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        inputArr[i] = inputs[inBaseIdx + i];
    }
    let input = VInput(inputArr);

    // Load output gradient
    half gradOutArr[OUTPUT_SIZE];
    int gradOutBaseIdx = sampleIdx * OUTPUT_SIZE;
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        gradOutArr[i] = grad_outputs[gradOutBaseIdx + i];
    }
    let dOutput = VOutput(gradOutArr);

    // Backward pass — single address pair, getOffset called inside differentiable fn
    var paramAddrPair = DifferentialPtrPair<Address>(baseAddr, gradBaseAddr);
    var inputPair = diffPair(input);

    bwd_diff(MatVecMul)(paramAddrPair, inputPair, dOutput);

    // Write input gradients
    if (gtid == 0)
    {
        int gradInBaseIdx = sampleIdx * INPUT_SIZE;
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            grad_inputs[gradInBaseIdx + i] = inputPair.d[i];
        }
    }
}
