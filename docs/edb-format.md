# EDB1 dataset format

All integers are little-endian. Packed structures have no implicit padding.
Offsets are absolute file offsets and loaders reject overflow or out-of-range
regions.

The header contains `EDB1`, version, endian tag, header size, record/class
counts, offsets for the record table, fixed-width class-name table and data,
total file size, and a header CRC32.

Each record descriptor contains the compressed image offset and byte length,
image CRC32, original width and height, annotation offset/count, and an
annotation-record CRC32. Images remain their original JPEG/PNG bytes; packing
performs no feature extraction.

Each annotation is four FP32 coordinates followed by a 16-bit class ID and
16-bit flags. Flag bit 0 means ignored. VOC objects outside the requested
target class table and objects marked `difficult` are stored as ignored boxes,
so training does not treat those regions as background.

The loader memory-maps the file, validates its structure immediately, and
validates each record checksum when that record is accessed.
