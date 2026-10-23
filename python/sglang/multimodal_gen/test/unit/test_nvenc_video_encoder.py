# SPDX-License-Identifier: Apache-2.0
"""A warm output and a different resolution must keep separate NVENC sessions."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(
    torch.cuda.is_available() and importlib.util.find_spec("PyNvVideoCodec"),
    "requires CUDA and PyNvVideoCodec",
)
class TestNvencVideoEncoder(CustomTestCase):
    def test_warmup_cache_hits_by_resolution(self):
        import av

        from sglang.multimodal_gen.runtime.entrypoints import (
            nvenc_video_encoder as nvenc,
        )

        with nvenc._cache_lock:
            nvenc._idle_sessions.clear()
        device = torch.device("cuda", torch.cuda.current_device())
        cases = (
            (832, 480, False),
            (640, 360, False),
            (832, 480, True),
            (640, 360, True),
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                for index, (width, height, expected_hit) in enumerate(cases):
                    path = Path(directory, f"video_{index}.mp4")
                    frames = torch.full(
                        (3, 4, height, width),
                        index / 5,
                        dtype=torch.float32,
                        device=device,
                    )
                    encoder = nvenc.NvencVideoEncoder.open(
                        str(path),
                        device=device,
                        height=height,
                        width=width,
                        fps=24,
                        num_frames=4,
                    )
                    self.assertIsNotNone(encoder)
                    self.assertEqual(encoder.cache_hit, expected_hit)
                    try:
                        encoder.write(frames)
                        encoder.close()
                    except BaseException:
                        encoder.abort()
                        raise
                    with av.open(str(path)) as container:
                        stream = container.streams.video[0]
                        decoded = list(container.decode(video=0))
                        self.assertEqual((stream.width, stream.height), (width, height))
                        self.assertEqual(len(decoded), 4)
                        self.assertEqual(decoded[0].pts, 0)
                        self.assertTrue(decoded[0].key_frame)
                        self.assertAlmostEqual(
                            float(stream.duration * stream.time_base), 4 / 24
                        )
        finally:
            with nvenc._cache_lock:
                nvenc._idle_sessions.clear()


if __name__ == "__main__":
    unittest.main()
