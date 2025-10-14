// Those macro are used to provide type constraints for the IVector matmul methods.

#define TYPE_CONSTRAINTS \
        where Storage : IStorage<T> \
        where Storage.Differential : IStorage<T.Differential> \
        where Storage.Address == Storage.Differential.Address \
        where OutputVector : IVector<T, OutputSize> \
        where OutputVector.ComponentDataType == ComponentDataType

#define TYPE_CONSTRAINTS_FOR_BINDLSS_STORAGE \
        where BindlessStorage : IStorage<T> \
        where BindlessStorage.Address : IPointerLikeAddress<T> \
        where BindlessStorage.Address.Differential : IPointerLikeAddress<T.Differential> \
        where OutputVector : IVector<T, OutputSize> \
        where OutputVector.ComponentDataType == ComponentDataType
