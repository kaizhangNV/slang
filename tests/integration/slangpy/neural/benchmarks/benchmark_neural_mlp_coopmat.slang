// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// CoopMat neural MLP benchmark using WaveTangledVector backend.
// Network architecture configurable via preprocessor defines.
// This matches the original neuralnetwork library benchmark for fair comparison.

import slang.neural;

// Network configuration - use defines or defaults
#ifndef INPUT_SIZE
#define INPUT_SIZE 64
#endif
#ifndef HIDDEN_SIZE
#define HIDDEN_SIZE 64
#endif
#ifndef OUTPUT_SIZE
#define OUTPUT_SIZE 16
#endif

// CoopMat configuration
static const int SubgroupSize = 32;
static const int BatchSize = 32;

typealias Address = BindlessAddress<float>;

// Parameter layout:
// Layer 1: weights [INPUT_SIZE * HIDDEN_SIZE], biases [HIDDEN_SIZE]
// Layer 2: weights [HIDDEN_SIZE * OUTPUT_SIZE], biases [OUTPUT_SIZE]
static const int LAYER1_WEIGHTS = INPUT_SIZE * HIDDEN_SIZE;
static const int LAYER1_BIASES = HIDDEN_SIZE;
static const int LAYER2_WEIGHTS = HIDDEN_SIZE * OUTPUT_SIZE;
static const int LAYER2_BIASES = OUTPUT_SIZE;

static const int PARAM_COUNT = LAYER1_WEIGHTS + LAYER1_BIASES + LAYER2_WEIGHTS + LAYER2_BIASES;

// Parameter offsets
static const int LAYER1_OFFSET = 0;
static const int LAYER2_OFFSET = LAYER1_WEIGHTS + LAYER1_BIASES;

// Query functions
public int get_input_size() { return INPUT_SIZE; }
public int get_hidden_size() { return HIDDEN_SIZE; }
public int get_output_size() { return OUTPUT_SIZE; }
public int get_total_params() { return PARAM_COUNT; }

// Shared memory size calculation for 2-layer network
typealias ShMemSize = SharedMemorySize<float, TargetEnum.CUDA, ExecutionMode.Inference, SubgroupSize, BatchSize / SubgroupSize>;
typealias ShMemSizeLayer1 = ShMemSize.OfLayer1<INPUT_SIZE, HIDDEN_SIZE>;
typealias ShMemSizeLayer2 = ShMemSize.OfLayer1<HIDDEN_SIZE, OUTPUT_SIZE>;
typealias ShMemSizeTotal = ShMemSizeLayer1;

// Define CoopMat types
typealias ShMemPool = SharedMemoryPool<ShMemSizeTotal>;

// Layer 1: INPUT -> HIDDEN
typealias VInput = WaveTangledVector<float, ShMemPool, INPUT_SIZE, SubgroupSize, 1>;
typealias VHidden = WaveTangledVector<float, ShMemPool, HIDDEN_SIZE, SubgroupSize, 1>;
typealias Act1 = ReLU<float>;
typealias Layer1 = FFLayer<float, VInput, VHidden, LinearLayout, Act1, true>;

// Layer 2: HIDDEN -> OUTPUT
typealias VOutput = WaveTangledVector<float, ShMemPool, OUTPUT_SIZE, SubgroupSize, 1>;
typealias Act2 = ReLU<float>;
typealias Layer2 = FFLayer<float, VHidden, VOutput, LinearLayout, Act2, true>;

// Compute shader entry point - one thread group per sample
[shader("compute")]
[numthreads(32, 1, 1)]
void compute_batch_forward(
    uint3 gid : SV_GroupID,
    uint gtid : SV_GroupIndex,
    uniform RWStructuredBuffer<float>.Handle params,
    StructuredBuffer<float> inputs,
    RWStructuredBuffer<float> outputs,
    uniform int batch_size)
{
    int sampleIdx = int(gid.x);
    if (sampleIdx >= batch_size)
        return;

    let baseAddr = Address(params);
    let layer1 = Layer1();
    let layer2 = Layer2();

    let layer1Addr = baseAddr.getOffset(LAYER1_OFFSET);
    let layer2Addr = baseAddr.getOffset(LAYER2_OFFSET);

    // Load input for this sample
    float inputArr[INPUT_SIZE];
    int inBaseIdx = sampleIdx * INPUT_SIZE;
    [ForceUnroll]
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        inputArr[i] = inputs[inBaseIdx + i];
    }

    // Forward pass through 2 layers
    let input = VInput(inputArr);
    let hidden = layer1.eval<Address>(input, layer1Addr);
    let output = layer2.eval<Address>(hidden, layer2Addr);

    // Only first thread writes result
    if (gtid == 0)
    {
        int outBaseIdx = sampleIdx * OUTPUT_SIZE;
        [ForceUnroll]
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            outputs[outBaseIdx + i] = output[i];
        }
    }
}
