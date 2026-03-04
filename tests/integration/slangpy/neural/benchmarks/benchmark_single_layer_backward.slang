// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Simple SINGLE LAYER backward pass benchmark.
// Input -> Output directly (no hidden layer)
// This should require much less shared memory than 2-layer MLP.

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
static const int BatchSize = 32;

// Use bindless address
typealias Address = BindlessAddress<half>;

// Single layer parameter layout:
// weights [INPUT_SIZE * OUTPUT_SIZE], biases [OUTPUT_SIZE]
static const int LAYER_WEIGHTS = INPUT_SIZE * OUTPUT_SIZE;
static const int LAYER_BIASES = OUTPUT_SIZE;
static const int PARAM_COUNT = LAYER_WEIGHTS + LAYER_BIASES;

static const int WEIGHT_OFFSET = 0;
static const int BIAS_OFFSET = LAYER_WEIGHTS;

// Query functions
public int get_input_size() { return INPUT_SIZE; }
public int get_output_size() { return OUTPUT_SIZE; }
public int get_total_params() { return PARAM_COUNT; }

// Shared memory size for single layer (much smaller than 2-layer!)
typealias ShMemSize = SharedMemorySize<half, TargetEnum.CUDA, ExecutionMode.Training, SubgroupSize, BatchSize / SubgroupSize>;
typealias ShMemSizeLayer = ShMemSize.OfLayer1<INPUT_SIZE, OUTPUT_SIZE>;
// Define vector types
typealias VInput = WaveTangledVector<half, ShMemSizeLayer, INPUT_SIZE, SubgroupSize>;
typealias VOutput = WaveTangledVector<half, ShMemSizeLayer, OUTPUT_SIZE, SubgroupSize>;
typealias Act = ReLU<half>;
typealias Layer = FFLayer<half, VInput, VOutput, LinearLayout, Act, true>;

// Simple differentiable forward function - just one layer!
[Differentiable]
VOutput singleLayerForward(Address weightAddr, Address biasAddr, VInput input, Layer layer)
{
    return layer.eval<Address>(input, weightAddr, biasAddr);
}

[Differentiable]
VOutput MatVecMul(Address weightAddr, Address biasAddr, VInput input)
{
    return input.linearTransform<Address, LinearLayout, VOutput>(weightAddr, biasAddr);
}

// Compute shader entry point - one thread group per sample
[shader("compute")]
[numthreads(32, 1, 1)]
void compute_single_layer_backward(
    uint3 gid : SV_GroupID,
    uint gtid : SV_GroupIndex,
    uniform RWStructuredBuffer<half>.Handle params,
    uniform StructuredBuffer<half> inputs,
    uniform StructuredBuffer<half> grad_outputs,
    uniform RWStructuredBuffer<half>.Handle grad_params,
    uniform RWStructuredBuffer<half> grad_inputs,
    uniform int batch_size)
{
    int sampleIdx = int(gid.x);
    if (sampleIdx >= batch_size)
        return;

    let baseAddr = Address(params);
    let gradBaseAddr = Address(grad_params);
    let layer = Layer();

    // Compute weight/bias addresses outside differentiable function
    let weightAddr = baseAddr.getOffset(WEIGHT_OFFSET);
    let biasAddr = baseAddr.getOffset(BIAS_OFFSET);
    let dWeightAddr = gradBaseAddr.getOffset(WEIGHT_OFFSET);
    let dBiasAddr = gradBaseAddr.getOffset(BIAS_OFFSET);

    // Load input
    half inputArr[INPUT_SIZE];
    int inBaseIdx = sampleIdx * INPUT_SIZE;
    [ForceUnroll]
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        inputArr[i] = inputs[inBaseIdx + i];
    }
    let input = VInput(inputArr);

    // Load output gradient
    half gradOutArr[OUTPUT_SIZE];
    int gradOutBaseIdx = sampleIdx * OUTPUT_SIZE;
    [ForceUnroll]
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        gradOutArr[i] = grad_outputs[gradOutBaseIdx + i];
    }
    let dOutput = VOutput(gradOutArr);

    // Backward pass
    var weightAddrPair = DifferentialPtrPair<Address>(weightAddr, dWeightAddr);
    var biasAddrPair = DifferentialPtrPair<Address>(biasAddr, dBiasAddr);
    var inputPair = diffPair(input);

    // bwd_diff(singleLayerForward)(weightAddrPair, biasAddrPair, inputPair, layer, dOutput);
    bwd_diff(MatVecMul)(weightAddrPair, biasAddrPair, inputPair, dOutput);

    // Write input gradients
    if (gtid == 0)
    {
        int gradInBaseIdx = sampleIdx * INPUT_SIZE;
        [ForceUnroll]
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            grad_inputs[gradInBaseIdx + i] = inputPair.d[i];
        }
    }
}
