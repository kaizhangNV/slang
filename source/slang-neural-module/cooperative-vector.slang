implementing neural;

#include "ivector-common.slang"

internal struct CoopVecDataType : IComponentDataType {}

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

    internal typealias ComponentDataType = CoopVecDataType;

    /// The compile-time size of the vector.
    public static const int Size = N;


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

    // CoopVec only supports the combination that all the input/matrix/bias/output are fp16.
    // TODO: We will need to make storage the different type from the input type here.
    internal static CooperativeVector<U, OutputSize> matmulInternal<U, V, int InputSize, int OutputSize>(
        CoopVec<U, InputSize> input,
        RWStructuredBuffer<V> weightBuffer,
        int weightOffset,
        bool transpose)
        where U : __BuiltinFloatingPointType
        where U.Differential == U
    {
        CooperativeVector<U, OutputSize> output = {};

        int matrixStride = sizeof(U) * InputSize;
        output.m_data =
            coopVecMatMul<U, OutputSize, InputSize, U, V>(
                input,                              // CoopVec<U, N> input,
                CoopVecComponentType.Float16,       // CoopVecComponentType inputInterpretation,
                weightBuffer,                   // RWStructuredBuffer<V> matrix
                weightOffset,                         // int32_t matrixOffset,
                CoopVecComponentType.Float16,       // CoopVecComponentType matrixInterpretation,
                CoopVecMatrixLayout.RowMajor,       // CoopVecMatrixLayout memoryLayout,
                transpose,                          // bool transpose,
                matrixStride                        // uint matrixStride (number of bytes between rows(row major)/columns(column major) of the matrix)
            );

        return output;
    }

    internal static CooperativeVector<U, OutputSize> matmulBiasInternal<U, V, int InputSize, int OutputSize>(
        CoopVec<U, InputSize> input,
        RWStructuredBuffer<V> weightBuffer,
        int weightOffset,
        RWStructuredBuffer<V> biasBuffer,
        int biasOffset,
        bool transpose)
        where U : __BuiltinFloatingPointType
        where U.Differential == U
    {
        CooperativeVector<U, OutputSize> output = {};

        int matrixStride = sizeof(V) * InputSize;
        output.m_data =
            coopVecMatMulAdd<U, OutputSize, InputSize, U, V>(
                input,                              // CoopVec<U, N> input,
                CoopVecComponentType.Float16,       // CoopVecComponentType inputInterpretation,
                weightBuffer,                       // RWStructuredBuffer<V> matrix
                weightOffset,                        // int32_t matrixOffset,
                CoopVecComponentType.Float16,       // CoopVecComponentType matrixInterpretation,
                biasBuffer,                       // RWStructuredBuffer<V> bias - it shares the same buffer with the matrix
                biasOffset,                         // int32_t biasOffset,
                CoopVecComponentType.Float16,       // CoopVecComponentType biasInterpretation,
                CoopVecMatrixLayout.RowMajor,       // CoopVecMatrixLayout memoryLayout,
                transpose,                          // bool transpose,
                matrixStride                        // uint matrixStride (number of bytes between rows(row major)/columns(column major) of the matrix)
            );
        return output;
    }

    [BackwardDerivative(matmulBwd)]
    public OutputVector matmul<int OutputSize, Storage, OutputVector>(
        Storage weightStorage,
        no_diff Storage.Address weightAddress) TYPE_CONSTRAINTS
    {
        OutputVector output = OutputVector();
        let buffer = weightStorage.getBuffer();
        if (let structuredBuffer = buffer as RWStructuredBuffer<T>)
        {
            // By using CoopVec, we will have to implement the backward pass manually, so it's fine to mark it as no_diff because
            // we don't want slang to auto-generate the backward pass for us.
            int startIndex = bit_cast<int>(weightAddress);
            var output = no_diff matmulInternal<T, T, N, OutputSize>(m_data, structuredBuffer, startIndex, false);
            return bit_cast<OutputVector>(output);
        }
        else if (let byteAddressBuffer = buffer as RWByteAddressBuffer)
        {
            static_assert(false, "Not implemented for RWByteAddressBuffer");
        }
        else
        {
            static_assert(false, "Currently only RWStructuredBuffer and RWByteAddressBuffer are supported for CooperativeVector");
        }
        return output;
    }

    static void matmulBwd<int OutputSize, Storage, OutputVector>(
        inout DifferentialPair<This> dthis,
        DifferentialPtrPair<Storage> dWeightStorage,
        no_diff Storage.Address dWeightAddress,
        OutputVector.Differential dOutput) TYPE_CONSTRAINTS
    {
        // let diffBuffer = dWeightStorage.d.getBuffer();
        // if (diffBuffer is RWStructuredBuffer<T>)
        // {
        //     var structuredDiffBuffer = diffBuffer as RWStructuredBuffer<T.Differential>;
        //     let structuredPrimalBuffer = dWeightStorage.p.getBuffer() as RWStructuredBuffer<T>;
        //     int startIndex = bit_cast<int>(dWeightAddress);

        //     // We have to cast the data type to the concrete CoopVec type because that is the intrinsic requires
        //     if (let dOutputConcrete  =  dOutput as CooperativeVector<T.Differential, OutputSize>)
        //     {
        //         // Compute the derivative of the input: dInput = W^T * dOutput
        //         let dOutputData = dOutputConcrete.m_data;
        //         var output = matmulInternal<T.Differential, T, OutputSize, N, Bias>(dOutputData, structuredPrimalBuffer.value, startIndex, true);
        //         dthis = DifferentialPair<This>(dthis.p, output);

        //         // Compute the derivative of the weights: dW = dOutput * Input^T
        //         int matrixStride = sizeof(T.Differential) * N;

        //         // Since the outterproductaccumulate intrinsic requires 'a' and 'b' to be the same type, we have to cast the primal Input
        //         // from CoopVec<T, N> to CoopVec<T.Differential, N>. But since we have constraint that T.Differential == T, we can just
        //         // reinterpret the data safely.
        //         let inputCast = reinterpret<CoopVec<T.Differential, N>>(dthis.p.m_data);

        //         coopVecOuterProductAccumulate<T.Differential, OutputSize, N, T.Differential>
        //             (dOutputData, inputCast, structuredDiffBuffer.value, startIndex, matrixStride, CoopVecMatrixLayout.TrainingOptimal, CoopVecComponentType.Float16);
        //     }
        //     else
        //     {
        //         static_assert(false, "Data type mismatch between OutputVector and Differential");
        //     }
        // }
        // else if (diffBuffer is RWByteAddressBuffer)
        // {
        //     static_assert(false, "Not implemented for RWByteAddressBuffer");
        // }
        // else
        // {
        //     static_assert(false, "Currently only RWStructuredBuffer and RWByteAddressBuffer are supported for CooperativeVector");
        // }
    }

    [BackwardDifferentiable]
    public OutputVector matmul<int OutputSize, Storage, OutputVector>(
        Storage weightStorage,
        Storage biasStorage,
        no_diff Storage.Address weightAddress,
        no_diff Storage.Address biasAddress) TYPE_CONSTRAINTS
    {
        OutputVector output = OutputVector();
        return output;
    }

    [BackwardDifferentiable]
    public OutputVector matmul<int OutputSize, BindlessStorage, OutputVector>(
        BindlessStorage.Address parameters) TYPE_CONSTRAINTS_FOR_BINDLSS_STORAGE
    {
        OutputVector output = OutputVector();
        return output;
    }

    [BackwardDifferentiable]
    public OutputVector matmul<int OutputSize, BindlessStorage, OutputVector>(
        BindlessStorage.Address weightAddress,
        BindlessStorage.Address biasAddress) TYPE_CONSTRAINTS_FOR_BINDLSS_STORAGE
    {
        OutputVector output = OutputVector();
        return output;
    }
}
