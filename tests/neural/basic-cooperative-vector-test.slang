//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-vk -render-feature cooperative-vector -output-using-type -xslang -DTEST_HALF=1

import neural;

#if TEST_HALF
typealias ElementType = half;
#else
typealias ElementType = float;
#endif

// set up a 2x4 matrix for input parameters, the last 2 elements are for bias
//TEST_INPUT: ubuffer(data=[1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0], stride=4):name=parametersFloat
// 1 2 3 4
// 5 6 7 8
// bias = {9.0, 10.0}
RWStructuredBuffer<float> parametersFloat;

//TEST_INPUT: ubuffer(data=[0 0 0 0 0 0 0 0 0 0], stride=4):name=parameters
RWStructuredBuffer<ElementType> parameters;

// Create a buffer to store the test result
//TEST_INPUT: ubuffer(data=[0 0], stride=4):out,name=testResult
RWStructuredBuffer<uint> testResult;

//TEST_INPUT: ubuffer(data=[0 0 0 0 0 0 0 0 0 0], stride=4):out,name=debugResult
RWStructuredBuffer<ElementType> debugResult;

//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):name=dInput
RWStructuredBuffer<ElementType> dInput;

// set up a 2x4 matrix for derivative of parameters, the last 2 elements are for derivative of bias
//TEST_INPUT: ubuffer(data=[0 0 0 0 0 0 0 0 0 0], stride=4):out,name=dParameters
RWStructuredBuffer<ElementType> dParameters;

typealias BufferStorage = StructuredBufferStorage<ElementType>;

void setupParameters()
{
    for (int i = 0; i < 10; i++)
    {
        parameters[i] = ElementType(parametersFloat[i]);
    }
}

[Differentiable]
OutputVector TestCooperativeVectorMatMulAdd<InputVector, OutputVector, bool Bias>(
    InputVector input,
    BufferStorage storage,
    no_diff BufferStorage.Address address)
    where InputVector : IVector<ElementType, 4>
    where OutputVector : IVector<ElementType, 2>
{
    var outputVec = input.matmul<2, Bias, BufferStorage, OutputVector>(storage, address);
    return outputVec;
}

// Basic test on MatMul without bias, this test covers both forward and backward pass
bool BasicTest()
{
    ElementType[4] inputData = {ElementType(1.0), ElementType(2.0), ElementType(3.0), ElementType(4.0)};
    let input = CooperativeVector<ElementType, 4>(inputData);
    BufferStorage storage = BufferStorage(parameters);

    BufferStorage.Address address = 0;

    // Run the forward pass
    let outputVec = TestCooperativeVectorMatMulAdd<
                        CooperativeVector<ElementType, 4>,
                        CooperativeVector<ElementType, 2>,
                        false>
                        (input, storage, address);

    // (1*1 + 2*2 + 3*3 + 4*4) = 30.0
    // (5*1 + 6*2 + 7*3 + 8*4) = 70.0
    bool isPassed = (outputVec[0] == 30.0 && outputVec[1] == 70.0);

    BufferStorage dstorage = BufferStorage(dParameters);
    var parameterDiffPair = DifferentialPtrPair<BufferStorage>(storage, dstorage);
    ElementType[2] dOutputData = {ElementType(2.0), ElementType(1.0)};
    let dOutput = CooperativeVector<ElementType, 2>(dOutputData);
    var dPair = diffPair(input);

    // Run the backward pass
    // dInput = W^T * dOutput
    // dInput = {3, 7, 11, 15}
    bwd_diff(TestCooperativeVectorMatMulAdd<CooperativeVector<ElementType, 4>, CooperativeVector<ElementType, 2>, false>)
                (dPair, parameterDiffPair, address, dOutput);

    isPassed = isPassed &&
        dPair.d[0] == 3.0 && dPair.d[1] == 7.0 && dPair.d[2] == 11.0 && dPair.d[3] == 15.0;

    isPassed = isPassed &&
        dParameters[0] == 1.0 && dParameters[1] == 2.0 && dParameters[2] == 3.0 && dParameters[3] == 4.0 &&
        dParameters[4] == 1.0 && dParameters[5] == 2.0 && dParameters[6] == 3.0 && dParameters[7] == 4.0;

    debugResult[0] = dPair.p[0];
    debugResult[1] = dPair.p[1];
    debugResult[2] = dPair.p[2];
    debugResult[3] = dPair.p[3];
    debugResult[4] = dOutput[0];
    debugResult[5] = dOutput[1];
    // debugResult[6] = dParameters[6];
    // debugResult[7] = dParameters[7];

    return isPassed;
}
// Basic test on MatMul with bias, this test covers both forward and backward pass
bool BasicTestWithBias()
{
    ElementType[4] inputData = {ElementType(1.0), ElementType(2.0), ElementType(3.0), ElementType(4.0)};
    let input = CooperativeVector<ElementType, 4>(inputData);
    BufferStorage storage = BufferStorage(parameters);
    BufferStorage.Address address = 0;

    // Run the forward pass
    let outputVec = TestCooperativeVectorMatMulAdd<
                        CooperativeVector<ElementType, 4>,
                        CooperativeVector<ElementType, 2>,
                        true>
                        (input, storage, address);

    // (1*1 + 2*2 + 3*3 + 4*4) + 9.0 = 39.0
    // (5*1 + 6*2 + 7*3 + 8*4) + 10.0 = 80.0
    bool isPassed = (outputVec[0] == 39.0 && outputVec[1] == 80.0);
    return isPassed;
}
[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{
    setupParameters();

    testResult[0] = BasicTest();
    // testResult[1] = BasicTestWithBias();

    // BUFFER: 1
}
