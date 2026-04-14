# Tiled MMA Refactor Handoff

## Architecture

```
TiledMMAHelper (dispatcher, mma-tiled-layout-helper.slang)
  ├── forward()  → __target_switch { cuda: TiledMMACuda, spirv: TiledMMAVulkan }
  ├── backward() → __target_switch { cuda: TiledMMACuda, spirv: TiledMMAVulkan }
  │
  ├── TiledMMACuda (mma-tiled-cuda.slang)
  │     ├── forward():      InputMatrix(MatA) + WeightMatrix(MatB) via loadWeightRaw + matMul
  │     ├── backward():     fromArrayPacked + mmaTranspose + biasReduce + outerProduct
  │     ├── mma():          forward MMA (input MatA × weight MatB → result MatC)
  │     └── mmaTranspose(): backward MMA (dOut MatA × W^T MatB → dInput MatC)
  │
  └── TiledMMAVulkan (mma-tiled-vulkan.slang)
        ├── forward():      loadWeightTiled + fromVectorViaShMem + matMul + toVectorViaShMem
        ├── backward():     mmaTranspose + biasReduce + outerProductAccumulate
        ├── mmaTranspose(): transpose MMA (W^T × input), consistent naming with CUDA
        ├── biasReduce():   MMA-based (ones × dOutB → column sums), reuses pre-loaded dOutB
        └── outerProductAccumulate(): fromVectorViaShMem + matMul + cross-warp reduce + atomic store
```

## Current status: all 62 tests pass

```bash
./build/Release/bin/slang-test \
  tests/neural/basic-coopmat-vector-tiled-layout-test.slang \
  tests/neural/fflayer-wavetangled-vector-tiled-test.slang \
  tests/neural/mma-tiled-layout-test-*.slang \
  tests/neural/outerproduct-accumulate-tiled-test*.slang \
  tests/neural/mma-tiled-backward-test.slang \
  tests/neural/shared-memory-size.slang \
  -use-test-server -server-count 8
# 100% of tests passed (62/62)
```

Includes CUDA and Vulkan, half and float, single-warp and multi-warp, aligned and arbitrary-size, forward and backward, shared memory size verification.

## Performance (backward pass, G=2 warps)

| Size   | Network    | Slang (ms) | Tin2 (ms) | Ratio |
|--------|------------|-----------|-----------|-------|
| tiny   | 32 → 16   | 0.0562    | 0.0554    | 1.01  |
| small  | 64 → 16   | 0.0562    | 0.0544    | 1.03  |
| medium | 128 → 32  | 0.0627    | 0.0600    | 1.05  |
| large  | 256 → 64  | 0.2779    | 0.2827    | 0.98  |
| xlarge | 128 → 128 | 0.2289    | 0.2211    | 1.04  |

Register pressure identical to Tin2:

| Size   | Slang regs | Slang spill S/L | Tin2 regs | Tin2 spill S/L |
|--------|-----------|----------------|-----------|----------------|
| tiny   | 40        | 0 / 0          | 40        | 0 / 0          |
| small  | 72        | 0 / 0          | 64        | 0 / 0          |
| medium | 166       | 0 / 0          | 166       | 0 / 0          |
| large  | 255       | 928 / 856      | 255       | 952 / 796      |
| xlarge | 255       | 572 / 564      | 255       | 548 / 496      |

## What was fixed in this session

### 18. CUDA float backward: float→half conversion in `backward()` (`mma-tiled-cuda.slang`)

`backward()` was missing the float→half conversion before `fromArrayPacked`, unlike the forward path. For T=float, each uint held a float bit pattern but `fromArrayPacked` interprets each uint as two packed halves. Added the same `sizeof(T)` branch as forward — converts float elements to half via `__realCast<half>` before packing.

### 19. biasReduce float readback fix (`mma-tiled-cuda.slang`)

The intermediate reduction stores `half2` pairs as `uint`, but the float readback was using `smemLoad<float>` which reinterprets half2 bits as float. Unified both paths to always read `vector<half, 2>` from shmem. The `sizeof(U)` branch now only affects the final atomicAdd: half uses vector atomicAdd, float converts each half via `__realCast<U>` before individual atomicAdd.

### 20. BytesForCrossWarpReduction scaling (`shared-memory-pool.slang`)

The outer product butterfly reduction has G/2 warps storing simultaneously in the first pass. The formula previously only allocated one matrix (correct for G=2 but would under-allocate for G>=4). Now uses `Max(SubgroupCount/2, 1)` to scale correctly. Updated `shared-memory-size.slang` test to verify all four `SharedMemorySize0` terms.

### 21. Shared memory size analysis

Full analysis confirmed that `SharedMemorySize0` (computed for linear layout) always provides sufficient shmem for the tiled layout:
- CUDA tiled forward/mmaTranspose use ZERO shmem (shuffles + loadWeightRaw)
- Cross-warp reduction shmem is layout-independent and already correctly computed
- BytesForMMA (linear layout overhead) provides headroom for small sizes where Vulkan tile loading would otherwise exceed reduction
- No Layout generic parameter needed in SharedMemorySize

### 22. Rebase onto upstream master

Successfully rebased all 49 commits onto upstream `1de4aea20` with:
- `hlsl.meta.slang` intrinsics (`clear`, `fragmentWrite`, `fragmentRead`, `getPackedFragmentCount`, `ChangeMajor`, `StoreRaw`, `LoadRaw`, `storeNative`, `loadNative`) using `default:` branches for HLSL/SPIR-V compatibility
- `accelerate-vector-coopmat.slang` signatures matching upstream: `[HasTrivialForwardDerivative]`, `[require(cuda_spirv)]`, `IPointerLikeAddress<T>` (not `T.Differential`), `Differential = WaveTangledVector<T,...>` (not `T.Differential`)
- All `[require(cooperative_matrix, subgroup_basic)]` expanded to include `cuda_spirv` to exclude HLSL target
- Dead code removed: `mma-new-helper.slang`, `mma-new-helper-v2.slang`

## Files modified

| File | Changes |
|------|---------|
| `WaveMatrix.slang` | `toArrayPacked` float fix, `LanesAreRows` param, tile-row-at-a-time shmem, `fromVectorViaShMem` float stride + bounds, `toVectorViaShMem` bounds guard |
| `mma-tiled-cuda.slang` | Float store in outerProduct, Uint4Aligned packed sizes, `Bias` flag, float→half conversion in backward, biasReduce float readback fix |
| `mma-tiled-vulkan.slang` | Renamed `mma`→`mmaTranspose`, per-warp shmem, pass `.data`, biasReduce `dOutA×onesB`, reuse dOutA, `Bias` flag |
| `mma-tiled-layout-helper.slang` | `Bias` template param on backward |
| `accelerate-vector-coopmat.slang` | Pass `Bias` from `linearTransformBwdOnTarget`, OptimalLayout dispatch |
| `shared-memory-pool.slang` | `BytesForCrossWarpReduction` scales with `SubgroupCount` |
| `hlsl.meta.slang` | `clear`, `fragmentWrite`/`fragmentRead`, `getPackedFragmentCount`, `ChangeMajor`, `StoreRaw`/`LoadRaw`, `storeNative`/`loadNative` |
| `prelude/slang-cuda-prelude.h` | MMA m16n8k16 PTX support, fragment helpers |
| `istorages.slang` | `BufferType`, `getBuffer()`, `getBaseIndex()` on `IPointerLikeAddress` |
| `bindless-storage.slang` | Implement `BufferType`/`getBuffer`/`getBaseIndex` on all address types |
| `tests/neural/common.slang` | Native layout helpers, tiled weight fill functions |
| `tests/neural/*.slang` | Forward, transpose, backward, outer product, FFLayer, shared memory size tests |

## How to run tests

```bash
# Build
cmake --build --preset release -j12
touch source/standard-modules/neural/*.slang
cmake --build --preset release -j12 --target slang-neural-module

# All tiled layout tests (CUDA + Vulkan)
./build/Release/bin/slang-test tests/neural/mma-tiled-layout-test-*.slang -use-test-server -server-count 8

# Outer product tests
./build/Release/bin/slang-test tests/neural/outerproduct-accumulate-tiled-test*.slang -use-test-server -server-count 4

# basic-coopmat + fflayer tests
./build/Release/bin/slang-test tests/neural/basic-coopmat-vector-tiled-layout-test.slang tests/neural/fflayer-wavetangled-vector-tiled-test.slang -use-test-server -server-count 4

# Register pressure check
for cfg in "32 16 tiny" "64 16 small" "128 32 medium" "256 64 large" "128 128 xlarge"; do
  set -- $cfg
  ./build/Release/bin/slangc benchmarks/benchmark_single_layer_backward.slang \
    -target ptx -entry compute_backward -stage compute \
    -o /tmp/backward_${3}.ptx \
    -I build/Release/lib/slang-standard-module-2026.3.1 \
    -DINPUT_SIZE=$1 -DOUTPUT_SIZE=$2 -DSUBGROUP_COUNT=2 -experimental-feature
  head -c -1 /tmp/backward_${3}.ptx > /tmp/backward_${3}_fixed.ptx
  printf "%-7s " "$3:"
  /usr/local/cuda/bin/ptxas -v --gpu-name sm_89 /tmp/backward_${3}_fixed.ptx -o /dev/null 2>&1 | grep -oE "Used [0-9]+ registers|[0-9]+ bytes spill (stores|loads)"
done

# Runtime benchmark (uses ncu_launcher, run 5 times per size for stable averages)
# Build launcher if needed:
#   /usr/local/cuda/bin/nvcc -o benchmarks/ncu_launcher_new_mma \
#     benchmarks/ncu_launcher_new_mma.cu -lcuda -arch=sm_89
for cfg in "32 16 tiny" "64 16 small" "128 32 medium" "256 64 large" "128 128 xlarge"; do
  set -- $cfg
  ./build/Release/bin/slangc benchmarks/benchmark_single_layer_backward.slang \
    -target ptx -entry compute_backward -stage compute \
    -o /tmp/backward_${3}.ptx \
    -I build/Release/lib/slang-standard-module-2026.3.1 \
    -DINPUT_SIZE=$1 -DOUTPUT_SIZE=$2 -DSUBGROUP_COUNT=2 -experimental-feature
  head -c -1 /tmp/backward_${3}.ptx > /tmp/backward_${3}_fixed.ptx
  echo "=== $3 (${1}x${2}) ==="
  for run in 1 2 3 4 5; do
    ./benchmarks/ncu_launcher_new_mma /tmp/backward_${3}_fixed.ptx \
      --input-size $1 --output-size $2 --batch-size 8192 --warps 2 --mode backward
  done
done
```
