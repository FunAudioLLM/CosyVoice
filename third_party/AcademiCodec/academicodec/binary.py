# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Raw binary format for Encodec compressed audio. Actual compression API is in `encodec.compress`."""
import io
import json
import struct
import typing as tp

# format is `ECDC` magic code, followed by the header size as uint32.
# Then an uint8 indicates the protocol version (0.)
# The header is then provided as json and should contain all required
# informations for decoding. A raw stream of bytes is then provided
# and should be interpretable using the json header.
_encodec_header_struct = struct.Struct('!4sBI')
_ENCODEC_MAGIC = b'ECDC'


def write_ecdc_header(fo: tp.IO[bytes], metadata: tp.Any):
    meta_dumped = json.dumps(metadata).encode('utf-8')
    version = 0
    header = _encodec_header_struct.pack(_ENCODEC_MAGIC, version,
                                         len(meta_dumped))
    fo.write(header)
    fo.write(meta_dumped)
    fo.flush()


def _read_exactly(fo: tp.IO[bytes], size: int) -> bytes:
    buf = b""
    while len(buf) < size:
        new_buf = fo.read(size)
        if not new_buf:
            raise EOFError("Impossible to read enough data from the stream, "
                           f"{size} bytes remaining.")
        buf += new_buf
        size -= len(new_buf)
    return buf


def read_ecdc_header(fo: tp.IO[bytes]):
    header_bytes = _read_exactly(fo, _encodec_header_struct.size)
    magic, version, meta_size = _encodec_header_struct.unpack(header_bytes)
    if magic != _ENCODEC_MAGIC:
        raise ValueError("File is not in ECDC format.")
    if version != 0:
        raise ValueError("Version not supported.")
    meta_bytes = _read_exactly(fo, meta_size)
    return json.loads(meta_bytes.decode('utf-8'))


class BitPacker:
    """Simple bit packer to handle ints with a non standard width, e.g. 10 bits.
    Note that for some bandwidth (1.5, 3), the codebook representation
    will not cover an integer number of bytes.

    Args:
        bits (int): number of bits per value that will be pushed.
        fo (IO[bytes]): file-object to push the bytes to.
    """

    def __init__(self, bits: int, fo: tp.IO[bytes]):
        self._current_value = 0
        self._current_bits = 0
        self.bits = bits
        self.fo = fo

    def push(self, value: int):
        """Push a new value to the stream. This will immediately
        write as many uint8 as possible to the underlying file-object."""
        self._current_value += (value << self._current_bits)
        self._current_bits += self.bits
        while self._current_bits >= 8:
            lower_8bits = self._current_value & 0xff
            self._current_bits -= 8
            self._current_value >>= 8
            self.fo.write(bytes([lower_8bits]))

    def flush(self):
        """Flushes the remaining partial uint8, call this at the end
        of the stream to encode."""
        if self._current_bits:
            self.fo.write(bytes([self._current_value]))
            self._current_value = 0
            self._current_bits = 0
        self.fo.flush()


class BitUnpacker:
    """BitUnpacker does the opposite of `BitPacker`.

    Args:
        bits (int): number of bits of the values to decode.
        fo (IO[bytes]): file-object to push the bytes to.
        """

    def __init__(self, bits: int, fo: tp.IO[bytes]):
        self.bits = bits
        self.fo = fo
        self._mask = (1 << bits) - 1
        self._current_value = 0
        self._current_bits = 0

    def pull(self) -> tp.Optional[int]:
        """
        Pull a single value from the stream, potentially reading some
        extra bytes from the underlying file-object.
        Returns `None` when reaching the end of the stream.
        """
        while self._current_bits < self.bits:
            buf = self.fo.read(1)
            if not buf:
                return None
            character = buf[0]
            self._current_value += character << self._current_bits
            self._current_bits += 8

        out = self._current_value & self._mask
        self._current_value >>= self.bits
        self._current_bits -= self.bits
        return out


def test():
    import torch
    torch.manual_seed(1234)
    for rep in range(4):
        length: int = torch.randint(10, 2_000, (1, )).item()
        bits: int = torch.randint(1, 16, (1, )).item()
        tokens: tp.List[int] = torch.randint(2**bits, (length, )).tolist()
        rebuilt: tp.List[int] = []
        buf = io.BytesIO()
        packer = BitPacker(bits, buf)
        for token in tokens:
            packer.push(token)
        packer.flush()
        buf.seek(0)
        unpacker = BitUnpacker(bits, buf)
        while True:
            value = unpacker.pull()
            if value is None:
                break
            rebuilt.append(value)
        assert len(rebuilt) >= len(tokens), (len(rebuilt), len(tokens))
        # The flushing mechanism might lead to "ghost" values at the end of the stream.
        assert len(rebuilt) <= len(tokens) + 8 // bits, (len(rebuilt),
                                                         len(tokens), bits)
        for idx, (a, b) in enumerate(zip(tokens, rebuilt)):
            assert a == b, (idx, a, b)


if __name__ == '__main__':
    test()
