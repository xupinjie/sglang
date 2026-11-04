# SPDX-License-Identifier: Apache-2.0
"""NVENC output, session reuse, and CPU fallback tests."""

import importlib.util
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sglang.test.test_utils import CustomTestCase


class TestNvencAvailability(CustomTestCase):
    def test_h264_encoder_capability_controls_nvenc(self):
        """A GPU without an H.264 encoder must fall back even when CUDA is available."""
        from sglang.multimodal_gen.runtime.entrypoints import (
            nvenc_video_encoder as nvenc,
        )

        native = types.ModuleType("PyNvVideoCodec")
        native.GetEncoderCaps = mock.Mock(return_value={"num_encoder_engines": 0})
        nvenc._device_has_nvenc.cache_clear()
        try:
            with (
                mock.patch.dict(sys.modules, {"PyNvVideoCodec": native}),
                mock.patch.object(torch.cuda, "is_available", return_value=True),
            ):
                self.assertFalse(nvenc.supports_nvenc(torch.device("cuda:0")))
                native.GetEncoderCaps.assert_called_once_with(0, "h264")
                native.GetEncoderCaps.return_value = {"num_encoder_engines": 1}
                nvenc._device_has_nvenc.cache_clear()
                self.assertTrue(nvenc.supports_nvenc(torch.device("cuda:0")))
                native.GetEncoderCaps.side_effect = RuntimeError()
                nvenc._device_has_nvenc.cache_clear()
                self.assertFalse(nvenc.supports_nvenc(torch.device("cuda:0")))
        finally:
            nvenc._device_has_nvenc.cache_clear()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_writer_remains_available_without_nvenc(self):
        from sglang.multimodal_gen.runtime.entrypoints import (
            nvenc_video_encoder as nvenc,
        )
        from sglang.multimodal_gen.runtime.entrypoints import (
            utils,
        )
        from sglang.srt.environ import envs

        frames = torch.zeros((3, 2, 8, 8), device="cuda")
        audio = np.zeros(800, np.float32)
        encoder = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            with (
                envs.SGLANG_USE_NVENC.override(True),
                mock.patch.object(
                    nvenc.NvencVideoEncoder, "open", return_value=None
                ) as open_nvenc,
                mock.patch.object(
                    utils.CudaVideoEncoder, "open", return_value=encoder
                ) as open_cuda,
            ):
                self.assertTrue(
                    utils._try_save_cuda_video_direct(
                        save_file_path=str(Path(directory, "fallback.mp4")),
                        sample=(frames, audio),
                        fps=24,
                        audio_sample_rate=16000,
                        output_compression=None,
                    )
                )
        self.assertIsNone(open_nvenc.call_args.kwargs["audio"])
        self.assertIs(open_cuda.call_args.kwargs["audio"], audio)
        self.assertIs(encoder.write.call_args.args[0], frames)
        encoder.close.assert_called_once()

    def test_streamed_nvenc_muxes_audio_after_video(self):
        from sglang.multimodal_gen.runtime.entrypoints import (
            nvenc_video_encoder as nvenc,
        )
        from sglang.multimodal_gen.runtime.entrypoints import (
            utils,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.encode_while_decoding import (
            EncodeWhileDecoding,
        )

        batch = SimpleNamespace(
            fps=24, num_frames=2, output_compression=None, x264_preset=None
        )
        encoder = mock.Mock()
        audio = np.zeros(800, np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory, "audio.mp4"))
            stream = EncodeWhileDecoding(
                path, batch, audio=audio, audio_sample_rate=16000, use_nvenc=True
            )
            with (
                mock.patch.object(
                    nvenc.NvencVideoEncoder, "open", return_value=encoder
                ) as open_nvenc,
                mock.patch.object(
                    utils.CudaVideoEncoder,
                    "open",
                    side_effect=AssertionError("unexpected CPU encoder"),
                ),
                mock.patch.object(utils, "_maybe_mux_audio_into_mp4") as mux_audio,
            ):
                stream(torch.zeros((1, 3, 2, 8, 8)))
                self.assertEqual(stream.finish(), [path])
        self.assertIsNone(open_nvenc.call_args.kwargs["audio"])
        encoder.write.assert_called_once()
        encoder.close.assert_called_once()
        self.assertEqual(mux_audio.call_args.kwargs["num_frames"], 2)
        self.assertIs(mux_audio.call_args.kwargs["audio"], audio)

    def test_two_frame_list_with_audio_and_cpu_fallback(self):
        from sglang.multimodal_gen.configs.sample.sampling_params import DataType
        from sglang.multimodal_gen.runtime.entrypoints import utils
        from sglang.srt.environ import envs

        frames = [np.full((8, 8, 3), 128, np.uint8) for _ in range(2)]
        audio = np.zeros(800, np.float32)
        sample = utils.attach_audio_to_video_sample(frames, audio, 0)
        video, selected_audio = utils._split_sample_audio(sample)
        self.assertIs(video, frames)
        self.assertIs(selected_audio, audio)

        materialized = utils.MaterializedOutput(
            sample=sample, frames=frames, audio=audio, fps=24
        )
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory, "fallback.mp4"))
            with (
                envs.SGLANG_USE_NVENC.override(True),
                mock.patch.object(
                    utils, "_try_save_cuda_video_direct", return_value=False
                ),
                mock.patch.object(
                    utils, "_try_save_video_with_audio", return_value=True
                ) as cpu_save,
            ):
                utils.save_materialized_output(materialized, DataType.VIDEO, path)
            cpu_save.assert_called_once()


@unittest.skipUnless(
    torch.cuda.is_available() and importlib.util.find_spec("PyNvVideoCodec"),
    "requires CUDA and PyNvVideoCodec",
)
class TestNvencVideoEncoder(CustomTestCase):
    def setUp(self):
        super().setUp()
        from sglang.multimodal_gen.runtime.entrypoints.nvenc_video_encoder import (
            supports_nvenc,
        )

        if not supports_nvenc(torch.device("cuda", torch.cuda.current_device())):
            self.skipTest("requires H.264 NVENC")

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

    def test_generic_video_output_uses_nvenc_with_audio(self):
        import av

        from sglang.multimodal_gen.configs.sample.sampling_params import DataType
        from sglang.multimodal_gen.runtime.entrypoints import utils
        from sglang.srt.environ import envs

        device = torch.device("cuda", torch.cuda.current_device())
        frames = torch.full((3, 4, 256, 256), 0.5, device=device)
        audio = torch.sin(torch.arange(3200) / 10).float()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "video_audio.mp4")
            with (
                envs.SGLANG_USE_NVENC.override(True),
                mock.patch.object(
                    utils,
                    "_save_video_ffmpeg",
                    side_effect=AssertionError("unexpected CPU encoding"),
                ),
            ):
                utils.save_outputs(
                    [frames],
                    DataType.VIDEO,
                    24,
                    True,
                    lambda _idx: str(path),
                    audio=audio,
                    audio_sample_rate=16000,
                )
            with av.open(str(path)) as container:
                self.assertEqual(len(list(container.decode(video=0))), 4)
                self.assertEqual(len(container.streams.audio), 1)

    def test_streamed_write_returns_while_nvenc_worker_is_busy(self):
        """Frame submission returns before the NVENC worker finishes."""
        import av

        from sglang.multimodal_gen.runtime.entrypoints.nvenc_video_encoder import (
            NvencVideoEncoder,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "overlapped.mp4")
            encoder = NvencVideoEncoder.open(
                str(path),
                device=device,
                height=256,
                width=256,
                fps=24,
                num_frames=4,
                max_queued_frames=4,
            )
            self.assertIsNotNone(encoder)
            started = threading.Event()
            release = threading.Event()
            original = encoder._write_frames

            def held_write(frames, ready):
                started.set()
                if not release.wait(5):
                    raise TimeoutError("NVENC worker was not released")
                return original(frames, ready)

            encoder._write_frames = held_write
            errors = []

            def submit_frames():
                try:
                    frames = torch.full((3, 4, 256, 256), 0.5, device=device)
                    encoder.write(frames)
                except BaseException as exc:
                    errors.append(exc)

            producer = threading.Thread(target=submit_frames)
            producer.start()
            try:
                self.assertTrue(started.wait(5))
                producer.join(timeout=0.25)
                self.assertFalse(producer.is_alive())
            finally:
                release.set()
                producer.join(timeout=5)
            self.assertEqual(errors, [])
            encoder.close()
            with av.open(str(path)) as container:
                frames = list(container.decode(video=0))
                self.assertEqual(len(frames), 4)
                self.assertAlmostEqual(
                    frames[0].to_ndarray(format="rgb24").mean(), 128, delta=12
                )

    def test_separate_audio_mux_timing(self):
        """Measure the extra audio pass while checking both saved MP4s retain A/V."""
        import json
        import statistics
        import time

        import av
        import numpy as np

        from sglang.multimodal_gen.runtime.entrypoints import utils
        from sglang.multimodal_gen.runtime.entrypoints.nvenc_video_encoder import (
            NvencVideoEncoder,
        )

        if utils.scipy_wavfile is None:
            self.skipTest("requires scipy to mux audio")

        device = torch.device("cuda", torch.cuda.current_device())
        width, height, fps, num_frames = 640, 352, 30, 90
        x = torch.linspace(0, 1, width, device=device)[None, None, None, :]
        y = torch.linspace(0, 1, height, device=device)[None, None, :, None]
        t = torch.arange(num_frames, device=device)[None, :, None, None] / num_frames
        frames = torch.cat(
            [
                ((x + t) % 1).expand(1, num_frames, height, width),
                ((y + t / 2) % 1).expand(1, num_frames, height, width),
                (((x + y) / 2 + t / 3) % 1),
            ],
            dim=0,
        ).contiguous()
        sample_rate = 24000
        samples = np.arange(sample_rate * num_frames // fps, dtype=np.float32)
        audio = (0.2 * np.sin(2 * np.pi * 440 * samples / sample_rate)).astype(
            np.float32
        )

        def cpu_save(path):
            start = time.perf_counter()
            self.assertTrue(
                utils._try_save_cuda_video_direct(
                    save_file_path=str(path),
                    sample=(frames, audio),
                    fps=fps,
                    audio_sample_rate=sample_rate,
                    output_compression=None,
                )
            )
            torch.cuda.synchronize(device)
            return {"total_s": time.perf_counter() - start}

        def nvenc_save(path):
            start = time.perf_counter()
            encoder = NvencVideoEncoder.open(
                str(path),
                device=device,
                height=height,
                width=width,
                fps=fps,
                num_frames=num_frames,
            )
            self.assertIsNotNone(encoder)
            try:
                encoder.write(frames)
                encoder.close()
            except BaseException:
                encoder.abort()
                raise
            torch.cuda.synchronize(device)
            encoded_s = time.perf_counter() - start
            utils._maybe_mux_audio_into_mp4(
                save_file_path=str(path),
                audio=audio,
                num_frames=num_frames,
                fps=fps,
                audio_sample_rate=sample_rate,
            )
            return {
                "encode_s": encoded_s,
                "mux_s": time.perf_counter() - start - encoded_s,
                "total_s": time.perf_counter() - start,
            }

        with tempfile.TemporaryDirectory() as directory:
            for name, save in (("cpu", cpu_save), ("nvenc", nvenc_save)):
                save(Path(directory, f"{name}_warm.mp4"))
            timings = {"cpu": [], "nvenc": []}
            for round_index in range(3):
                order = ("cpu", "nvenc") if round_index != 1 else ("nvenc", "cpu")
                for name in order:
                    path = Path(directory, f"{name}_{round_index}.mp4")
                    timing = (cpu_save if name == "cpu" else nvenc_save)(path)
                    timings[name].append(timing)
                    with av.open(str(path)) as container:
                        self.assertEqual(len(container.streams.video), 1)
                        self.assertEqual(len(container.streams.audio), 1)
                        self.assertEqual(
                            sum(1 for _ in container.decode(video=0)), num_frames
                        )
                    with av.open(str(path)) as container:
                        self.assertGreater(
                            sum(frame.samples for frame in container.decode(audio=0)),
                            sample_rate * (num_frames / fps - 0.1),
                        )
            print(
                "AUDIO_MUX_TIMING "
                + json.dumps(
                    {
                        "shape": [num_frames, height, width],
                        "cpu_total_median_s": statistics.median(
                            sample["total_s"] for sample in timings["cpu"]
                        ),
                        "nvenc_encode_median_s": statistics.median(
                            sample["encode_s"] for sample in timings["nvenc"]
                        ),
                        "separate_mux_median_s": statistics.median(
                            sample["mux_s"] for sample in timings["nvenc"]
                        ),
                        "nvenc_total_median_s": statistics.median(
                            sample["total_s"] for sample in timings["nvenc"]
                        ),
                    },
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
