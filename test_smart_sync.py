import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import soundfile as sf

from app.pipeline.recap_smart_sync import (
    assign_full_coverage_boundaries,
    build_mouth_sync_plan,
    build_source_timed_narration,
    render_smart_sync,
)
from app.pipeline.tts_engine import TTSEngine


def _duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def test_recap_boundaries_start_at_source_speech_and_keep_intro_separate():
    units = [
        {"speech_start": 0.45, "speech_end": 1.2},
        {"speech_start": 2.1, "speech_end": 2.8},
        {"speech_start": 3.4, "speech_end": 4.0},
    ]

    aligned = assign_full_coverage_boundaries(units, 5.0)

    assert aligned[0]["video_start"] == 0.45
    assert aligned[0]["video_end"] == 2.1
    assert aligned[1]["video_start"] == aligned[0]["video_end"]
    assert aligned[1]["video_end"] == 3.4
    assert aligned[2]["video_start"] == aligned[1]["video_end"]
    assert aligned[2]["video_end"] == 5.0
    # The caller can render [0, 0.45] as an intro while the TTS track is silent.
    assert aligned[0]["video_start"] > 0.0


def test_tts_lead_in_is_silence_and_shifts_caption_timing():
    with TemporaryDirectory() as temp:
        output_dir = Path(temp)
        engine = TTSEngine()

        def fake_edge_tts(text, voice, output_path, pitch_hz=20, rate_pct=20):
            samples = np.full(4410, 0.15, dtype=np.float32)
            sf.write(str(output_path), samples, 44100, subtype="PCM_16")

        with patch.object(engine, "_generate_edge_segment", side_effect=fake_edge_tts):
            audio_path, synced = engine.generate(
                segments=[{"id": 7, "start": 1.0, "end": 2.0, "text": "hello"}],
                output_dir=output_dir,
                engine="edge_tts",
                pause_duration=0.1,
                leading_silence_duration=0.5,
                tuning={"edge_pitch_hz": 20, "edge_rate_pct": 20},
            )

        audio, sample_rate = sf.read(str(audio_path))
        assert sample_rate == 44100
        assert synced[0]["start"] == 0.5
        assert synced[0]["slot_start"] == 0.5
        assert abs((synced[0]["end"] - synced[0]["start"]) - 0.1) < 0.002
        assert np.max(np.abs(audio[: int(0.45 * sample_rate)])) == 0
        assert np.max(np.abs(audio[int(0.51 * sample_rate):])) > 0.05


def test_mouth_sync_plan_preserves_all_non_speech_footage_at_source_duration():
    synced = [
        {
            "id": 1, "source_start": 1.0, "source_end": 2.0,
            "start": 0.5, "end": 1.8, "slot_duration": 1.5,
        },
        {
            "id": 2, "source_start": 3.0, "source_end": 4.2,
            "start": 2.0, "end": 2.7, "slot_duration": 0.7,
        },
    ]

    plan = build_mouth_sync_plan(synced, 5.0)

    assert len(plan) == 5  # intro, speech, source gap, speech, trailing silence
    assert plan[0]["slot_duration"] == 1.0
    assert plan[0]["audio_type"] == "silence"
    assert abs(plan[1]["slot_duration"] - 1.3) < 0.001
    assert plan[1]["audio_type"] == "speech"
    assert plan[1]["audio_start"] == 0.5 and plan[1]["audio_end"] == 1.8
    assert abs(plan[2]["slot_duration"] - 1.0) < 0.001
    assert plan[2]["audio_type"] == "silence"
    assert abs(plan[3]["slot_duration"] - 0.7) < 0.001
    assert abs(plan[4]["slot_duration"] - 0.8) < 0.001
    # Caption timestamps move with the rebuilt track; spoken audio duration stays unchanged.
    assert synced[0]["start"] == 1.0 and synced[0]["end"] == 2.3
    assert abs(synced[1]["start"] - 3.3) < 0.001
    assert abs(synced[1]["end"] - 4.0) < 0.001


def test_source_timed_narration_rebuilds_silence_without_speeding_up_voice():
    with TemporaryDirectory() as temp:
        work_dir = Path(temp)
        sample_rate = 44100
        raw_tts = work_dir / "raw_tts.wav"
        aligned = work_dir / "aligned.wav"
        source_video = work_dir / "source.mp4"
        output_video = work_dir / "synced.mp4"
        samples = np.zeros(int(2.7 * sample_rate), dtype=np.float32)
        tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(int(1.3 * sample_rate)) / sample_rate)
        samples[int(0.5 * sample_rate):int(1.8 * sample_rate)] = tone
        second_start = int(2.0 * sample_rate)
        second_end = int(2.7 * sample_rate)
        tone2 = 0.2 * np.sin(2 * np.pi * 660 * np.arange(second_end - second_start) / sample_rate)
        samples[second_start:second_end] = tone2
        sf.write(str(raw_tts), samples, sample_rate)

        synced = [
            {"id": 1, "source_start": 1.0, "source_end": 2.0,
             "start": 0.5, "end": 1.8, "slot_duration": 1.5},
            {"id": 2, "source_start": 3.0, "source_end": 4.2,
             "start": 2.0, "end": 2.7, "slot_duration": 0.7},
        ]
        plan = build_mouth_sync_plan(synced, 5.0)
        build_source_timed_narration(raw_tts, plan, aligned)
        audio, _ = sf.read(str(aligned))

        assert abs(_duration(aligned) - 4.8) < 0.02
        assert np.max(np.abs(audio[:int(0.95 * sample_rate)])) < 0.001
        assert np.max(np.abs(audio[int(1.1 * sample_rate):int(2.2 * sample_rate)])) > 0.1
        assert np.max(np.abs(audio[int(2.4 * sample_rate):int(3.2 * sample_rate)])) < 0.001
        assert np.max(np.abs(audio[int(3.4 * sample_rate):int(3.9 * sample_rate)])) > 0.1
        assert np.max(np.abs(audio[int(4.1 * sample_rate):])) < 0.001

        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=navy:s=320x240:r=30:d=5",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source_video),
            ],
            check=True,
        )
        render_smart_sync(
            video_path=source_video,
            tts_audio_path=aligned,
            segments=plan,
            output_path=output_video,
            work_dir=work_dir,
            resolution="tiktok540",
        )
        assert abs(_duration(output_video) - _duration(aligned)) <= 1 / 30 + 0.01


def test_render_smart_sync_keeps_video_and_tts_within_one_frame():
    with TemporaryDirectory() as temp:
        work_dir = Path(temp)
        source = work_dir / "source.mp4"
        narration = work_dir / "narration.wav"
        output = work_dir / "synced.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=navy:s=320x240:r=30:d=3",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
            ],
            check=True,
        )
        sf.write(str(narration), np.zeros(int(2.2 * 44100), dtype=np.float32), 44100)

        render_smart_sync(
            video_path=source,
            tts_audio_path=narration,
            segments=[
                {"video_start": 0.0, "video_end": 0.4, "slot_duration": 0.4},
                {"video_start": 0.4, "video_end": 2.0, "slot_duration": 1.1},
                {"video_start": 2.0, "video_end": 3.0, "slot_duration": 0.7},
            ],
            output_path=output,
            work_dir=work_dir,
            resolution="tiktok540",
        )

        assert output.exists() and output.stat().st_size > 0
        assert abs(_duration(output) - _duration(narration)) <= 1 / 30 + 0.01
