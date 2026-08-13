# EDM1 model format

All integers are little-endian. Packed structures have no implicit padding.
Offsets are absolute file offsets.

The versioned header stores the `EDM1` magic, endian tag, architecture and
precision IDs, model flags, tensor/class counts, metadata length, table and
payload offsets, total file length, payload and upstream-source SHA-256
digests, and a header CRC32. The frozen-backbone flag records the adaptation
scope.

Each tensor descriptor contains a fixed-width name, dtype, rank, four
dimensions, payload offset/length, CRC32, and trainability flags. Class names
are fixed-width UTF-8 strings. UTF-8 metadata records architecture,
normalization, source URL, license and source-to-target mapping.

FP32 PicoDet tensors use NHWC-compatible native layouts: dense convolution
weights are OHWI and depthwise convolution weights are KKC. The loader rejects
unsupported architecture/precision IDs, malformed regions, header/tensor CRC
failures, and payload SHA-256 mismatches.

Tensor dtype 4 stores IEEE-754 binary16 values in the external artifact. The
loader validates their CRCs and expands them deterministically to FP32, so the
same inference and training kernels can use a roughly half-sized pretrained
file. `edcompact` creates this representation; saving normally emits FP32.

Tensor dtype 5 is a hybrid INT8 storage representation for convolution
weights. Its payload contains a channel count, one FP32 symmetric scale per
output channel, then signed INT8 values. Sensitive rank-3 depthwise weights
remain binary16 and small rank-1 bias/constant tensors remain FP32 in the same
artifact. The loader dequantizes all tensors to FP32 before execution. Create
it with `edcompact --precision int8`; this reduces transfer/storage size but is
not an INT8 compute kernel.

Tensor dtype 6 packs signed symmetric INT4 values two per byte, with one FP32
scale per output channel. `edcompact --precision int4` is deliberately a
mixed-precision mode: only pointwise tensors with at least 131,072 elements
use W4, smaller pointwise tensors use dtype 5, depthwise tensors stay FP16,
and biases stay FP32. Quantizing every pointwise layer to W4 was measured and
rejected because it reduced detection mAP sharply. The loader verifies the
record CRC and payload digest before unpacking and dequantizing to FP32.

Saving writes a new complete file and regenerates the per-tensor CRCs, payload
digest and header CRC. Pretrained and adapted models remain ordinary external
files.
