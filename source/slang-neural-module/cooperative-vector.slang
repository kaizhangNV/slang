implementing neural;


public extension<T : __BuiltinFloatingPointType, int N> CoopVec<T, N> : IDifferentiable
    where T.Differential : __BuiltinFloatingPointType
{
    public typealias Differential = CoopVec<T.Differential, N>;
}

public struct CooperativeVector<T, int N> : IVector<T, N>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    /// The differential type for automatic differentiation.
    public typealias Differential = CooperativeVector<T.Differential, N>;
    internal typealias DType = CoopVec<T, N>;
    /// The compile-time size of the vector.
    public static const int Size = N;
    public static const DTypeEnum dTypeEnum = DTypeEnum.CooperativeVector;

    [DerivativeMember(Differential.m_data)]
    internal CoopVec<T, N> m_data;

    /// Default constructor - initializes all elements to zero.
    public __init() { m_data = CoopVec<T, N>(); }

    public __init(T value)
    {
        m_data = CoopVec<T, N>(value);
    }

    public __init(T[N] data)
    {
        // TODO: Need to have a array constructor for CoopVec intrinsic
        for (int i = 0; i < N; i++)
        {
            m_data[i] = data[i];
        }
    }

    public __init(This other)
    {
        this.m_data = other.m_data;
    }

    public __subscript(int index) -> T
    {
        get { return m_data[index]; }
        set { m_data[index] = newValue; }
    }

    internal DType getInternalData() { return m_data; }

    // CoopVec only supports the combination that all the input/matrix/bias/output are fp16.
    // TODO: We will need to make storage the different type from the input type here.
    internal static CooperativeVector<U, OutputSize> matmulInternal<U, V, int InputSize, int OutputSize, bool Bias>(
        CoopVec<U, InputSize> input,
        RWStructuredBuffer<V> parametersBuffer,
        int startIndex,
        bool transpose)
        where U : __BuiltinFloatingPointType
        where U.Differential == U
    {
        CooperativeVector<U, OutputSize> output = {};
        if (Bias)
        {
            int matrixStride = sizeof(V) * InputSize;
            int biasOffset = startIndex + InputSize * OutputSize;
            output.m_data =
                coopVecMatMulAdd<U, OutputSize, InputSize, U, V>(
                    input,                              // CoopVec<U, N> input,
                    CoopVecComponentType.Float16,       // CoopVecComponentType inputInterpretation,
                    parametersBuffer,                   // RWStructuredBuffer<V> matrix
                    startIndex,                         // int32_t matrixOffset,
                    CoopVecComponentType.Float16,       // CoopVecComponentType matrixInterpretation,
                    parametersBuffer,                   // RWStructuredBuffer<V> bias - it shares the same buffer with the matrix
                    biasOffset,                         // int32_t biasOffset,
                    CoopVecComponentType.Float16,       // CoopVecComponentType biasInterpretation,
                    CoopVecMatrixLayout.RowMajor,       // CoopVecMatrixLayout memoryLayout,
                    transpose,                          // bool transpose,
                    matrixStride                        // uint matrixStride (number of bytes between rows(row major)/columns(column major) of the matrix)
                );
        }
        else
        {
            int matrixStride = sizeof(U) * InputSize;
            output.m_data =
                coopVecMatMul<U, OutputSize, InputSize, U, V>(
                    input,                              // CoopVec<U, N> input,
                    CoopVecComponentType.Float16,       // CoopVecComponentType inputInterpretation,
                    parametersBuffer,                   // RWStructuredBuffer<V> matrix
                    startIndex,                         // int32_t matrixOffset,
                    CoopVecComponentType.Float16,       // CoopVecComponentType matrixInterpretation,
                    CoopVecMatrixLayout.RowMajor,       // CoopVecMatrixLayout memoryLayout,
                    transpose,                          // bool transpose,
                    matrixStride                        // uint matrixStride (number of bytes between rows(row major)/columns(column major) of the matrix)
                );
        }
        return output;
    }

    [BackwardDerivative(matmulBwd)]
    public OutputVector matmul<int OutputSize, bool Bias, Storage, OutputVector>(
        Storage storage,
        no_diff Storage.Address address)
        where Storage : IStorage<T>
        where Storage.Differential : IStorage<T.Differential>
        where Storage.Address == Storage.Differential.Address
        where OutputVector : IVector<T, OutputSize>
    {
        static_assert(This.dTypeEnum == OutputVector.dTypeEnum, "Data type mismatch between InputVector and OutputVector");

        OutputVector output = OutputVector();
        let buffer = storage.getBuffer();
        if (buffer is RWStructuredBuffer<T>)
        {
            // By using CoopVec, we will have to implement the backward pass manually, so it's fine to mark it as no_diff because
            // we don't want slang to auto-generate the backward pass for us.
            let structuredBuffer = buffer as RWStructuredBuffer<T>;
            int startIndex = bit_cast<int>(address);
            var output = no_diff matmulInternal<T, T, N, OutputSize, Bias>(m_data, structuredBuffer.value, startIndex, false);
            return bit_cast<OutputVector>(output);
        }
        else if (buffer is RWByteAddressBuffer)
        {
            static_assert(false, "Not implemented for RWByteAddressBuffer");
        }
        else
        {
            static_assert(false, "Currently only RWStructuredBuffer and RWByteAddressBuffer are supported for CooperativeVector");
        }
        return output;
    }

    static void matmulBwd<int OutputSize, bool Bias, Storage, OutputVector>(
        inout DifferentialPair<This> dthis,
        DifferentialPtrPair<Storage> dstorage,
        no_diff Storage.Address address,
        OutputVector.Differential dOutput)
            where Storage : IStorage<T>
            where Storage.Differential : IStorage<T.Differential>
            where Storage.Address == Storage.Differential.Address
            where OutputVector : IVector<T, OutputSize>
    {
        let diffBuffer = dstorage.d.getBuffer();
        if (diffBuffer is RWStructuredBuffer<T>)
        {
            var structuredDiffBuffer = diffBuffer as RWStructuredBuffer<T.Differential>;
            let structuredPrimalBuffer = dstorage.p.getBuffer() as RWStructuredBuffer<T>;
            int startIndex = bit_cast<int>(address);

            // We have to cast the data type to the concrete CoopVec type because that is the intrinsic requires
            if (let dOutputConcrete  =  dOutput as CooperativeVector<T.Differential, OutputSize>)
            {
                // Compute the derivative of the input: dInput = W^T * dOutput
                let dOutputData = dOutputConcrete.m_data;
                var output = matmulInternal<T.Differential, T, OutputSize, N, Bias>(dOutputData, structuredPrimalBuffer.value, startIndex, true);
                dthis = DifferentialPair<This>(dthis.p, output);

                // Compute the derivative of the weights: dW = dOutput * Input^T
                int matrixStride = sizeof(T.Differential) * N;

                // Since the outterproductaccumulate intrinsic requires 'a' and 'b' to be the same type, we have to cast the primal Input
                // from CoopVec<T, N> to CoopVec<T.Differential, N>. But since we have constraint that T.Differential == T, we can just
                // reinterpret the data safely.
                let inputCast = reinterpret<CoopVec<T.Differential, N>>(dthis.p.m_data);

                coopVecOuterProductAccumulate<T.Differential, OutputSize, N, T.Differential>
                    (dOutputData, inputCast, structuredDiffBuffer.value, startIndex, matrixStride, CoopVecMatrixLayout.TrainingOptimal, CoopVecComponentType.Float16);
            }
            else
            {
                static_assert(false, "Data type mismatch between OutputVector and Differential");
            }
        }
        else if (diffBuffer is RWByteAddressBuffer)
        {
            static_assert(false, "Not implemented for RWByteAddressBuffer");
        }
        else
        {
            static_assert(false, "Currently only RWStructuredBuffer and RWByteAddressBuffer are supported for CooperativeVector");
        }
    }

    [BackwardDifferentiable]
    public OutputVector matmul<int OutputSize, bool Bias, BindlessStorage, OutputVector>(
        BindlessStorage.Address parameters)
            where BindlessStorage : IStorage<T>
            where BindlessStorage.Address : IPointerLikeAddress<T>
            where BindlessStorage.Address.Differential : IPointerLikeAddress<T.Differential>
            where OutputVector : IVector<T, OutputSize>
    {
        OutputVector output = OutputVector();
        return output;
    }
}
