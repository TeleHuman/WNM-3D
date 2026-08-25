#!/usr/bin/env python3
"""Convert GN0-rendered InteriorGS trajectories to the WNM-3D LeRobot layout.

The converter joins two public GN0 artifacts by ``(scene, trajectory_id)``:

* rendered videos: ``<render-root>/<scene>/<trajectory_id>.mp4``;
* trajectory annotations: ``InteriorGS_train_seen.parquet``.

Language instructions and world-space reference paths are read from the
annotation parquet. Camera extrinsics are reconstructed from the reference path
and aligned frame-by-frame with the rendered video, so no private instruction
JSON or per-frame camera sidecars are required.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import itertools
import json
import math
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


VIDEO_KEY = "observation.images.rgb"
EXTRINSIC_KEY = "observation.camera_extrinsic"
LANGUAGE_KEY = "annotation.language.language_instruction"
CHUNK_SIZE = 1000
DEFAULT_ANNOTATIONS = Path(
    "../GN0/data/datasets/GN_Matrix/InteriorGS/InteriorGS_train_seen.parquet"
)
DEFAULT_OUTPUT_ROOT = Path("data/interiorgs_lerobot_seen")
DEFAULT_VIDEO_CODEC = "libx264"
H264_VIDEO_CODECS = frozenset({"h264", "x264", "avc1", "libx264"})


@dataclass(frozen=True)
class TrajectoryRecord:
    scene: str
    trajectory: str
    instruction: str
    path_xy: tuple[tuple[float, float], ...]
    start_heading_degrees: float


@dataclass(frozen=True)
class VideoProbe:
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class ConversionJob:
    episode_index: int
    task_index: int
    record: TrajectoryRecord
    video_path: Path
    video_probe: VideoProbe


@dataclass(frozen=True)
class ConversionSettings:
    output_root: Path
    frame_stride: int
    resize_scale: float
    camera_height: float
    video_codec: str
    pose_source: str


def normalize_trajectory_id(value: object) -> str:
    """Return a filename-compatible trajectory id without numeric padding."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)) or not float(value).is_integer():
            raise ValueError(f"Invalid trajectory_id: {value!r}")
        return str(int(value))

    text = str(value).strip()
    if not text:
        raise ValueError("Empty trajectory_id")
    try:
        return str(int(text))
    except ValueError:
        return text


def trajectory_sort_key(record: TrajectoryRecord) -> tuple[str, int, int | str]:
    try:
        return record.scene, 0, int(record.trajectory)
    except ValueError:
        return record.scene, 1, record.trajectory


def load_trajectory_records(
    annotations_path: Path,
    instruction_column: str,
) -> list[TrajectoryRecord]:
    required_columns = {
        "scene",
        "trajectory_id",
        instruction_column,
        "path_raster_world_x",
        "path_raster_world_y",
        "start_facing_heading_degrees",
    }
    available_columns = set(pq.ParquetFile(annotations_path).schema_arrow.names)
    missing = sorted(required_columns - available_columns)
    if missing:
        raise ValueError(
            f"Annotation parquet is missing required columns: {', '.join(missing)}"
        )

    columns = sorted(required_columns)
    table = pd.read_parquet(annotations_path, columns=columns)
    records: list[TrajectoryRecord] = []
    seen_keys: set[tuple[str, str]] = set()

    for row_number, row in enumerate(table.itertuples(index=False), start=1):
        values = row._asdict()
        scene = str(values["scene"]).strip()
        trajectory = normalize_trajectory_id(values["trajectory_id"])
        key = (scene, trajectory)
        if not scene:
            raise ValueError(f"Row {row_number} has an empty scene")
        if key in seen_keys:
            raise ValueError(
                "Annotation parquet contains a duplicate (scene, trajectory_id): "
                f"{scene}/{trajectory}"
            )
        seen_keys.add(key)

        instruction_value = values[instruction_column]
        instruction = (
            "" if pd.isna(instruction_value) else str(instruction_value).strip()
        )
        if not instruction:
            raise ValueError(
                f"Row {row_number} has an empty {instruction_column!r}: "
                f"{scene}/{trajectory}"
            )

        path_x = np.asarray(values["path_raster_world_x"], dtype=np.float64)
        path_y = np.asarray(values["path_raster_world_y"], dtype=np.float64)
        if path_x.ndim != 1 or path_y.ndim != 1 or len(path_x) != len(path_y):
            raise ValueError(f"Invalid raster path arrays for {scene}/{trajectory}")
        if len(path_x) == 0:
            raise ValueError(f"Empty raster path for {scene}/{trajectory}")
        path_xy = np.column_stack([path_x, path_y])
        if not np.isfinite(path_xy).all():
            raise ValueError(f"Non-finite raster path for {scene}/{trajectory}")

        heading = float(values["start_facing_heading_degrees"])
        if not math.isfinite(heading):
            raise ValueError(f"Invalid start heading for {scene}/{trajectory}")

        records.append(
            TrajectoryRecord(
                scene=scene,
                trajectory=trajectory,
                instruction=instruction,
                path_xy=tuple((float(x), float(y)) for x, y in path_xy),
                start_heading_degrees=heading,
            )
        )

    return sorted(records, key=trajectory_sort_key)


def probe_video(video_path: Path, fallback_fps: float) -> VideoProbe:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open rendered video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()

    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Rendered video has invalid metadata: {video_path}")
    if fps <= 0:
        fps = fallback_fps
    return VideoProbe(frame_count=frame_count, fps=fps, width=width, height=height)


def build_jobs(
    records: Iterable[TrajectoryRecord],
    render_root: Path,
    *,
    frame_stride: int,
    resize_scale: float,
    fallback_fps: float,
    min_output_frames: int,
    first_n: int | None,
    require_all_videos: bool,
) -> tuple[list[ConversionJob], dict[str, int], int, int]:
    matched: list[tuple[TrajectoryRecord, Path, VideoProbe]] = []
    missing_videos = 0
    short_videos = 0

    for record in records:
        video_path = render_root / record.scene / f"{record.trajectory}.mp4"
        if not video_path.is_file():
            missing_videos += 1
            continue

        video_probe = probe_video(video_path, fallback_fps=fallback_fps)
        output_frames = (video_probe.frame_count + frame_stride - 1) // frame_stride
        if output_frames < min_output_frames:
            short_videos += 1
            continue

        matched.append((record, video_path, video_probe))
        if first_n is not None and len(matched) >= first_n:
            break

    if require_all_videos and missing_videos:
        raise FileNotFoundError(
            f"Missing {missing_videos} rendered videos under {render_root}"
        )
    if not matched:
        raise RuntimeError(
            "No convertible trajectories were found. Expected rendered videos at "
            f"{render_root}/<scene>/<trajectory_id>.mp4."
        )

    output_shapes = {
        (
            max(1, int(round(probe.width * resize_scale))),
            max(1, int(round(probe.height * resize_scale))),
        )
        for _, _, probe in matched
    }
    output_fps_values = {round(probe.fps / frame_stride, 6) for _, _, probe in matched}
    if len(output_shapes) != 1:
        raise ValueError(
            "All rendered videos must have the same resolution; found output shapes "
            f"{sorted(output_shapes)}"
        )
    if len(output_fps_values) != 1:
        raise ValueError(
            "All rendered videos must have the same frame rate; found output FPS values "
            f"{sorted(output_fps_values)}"
        )

    task_to_index: dict[str, int] = {}
    jobs: list[ConversionJob] = []
    for episode_index, (record, video_path, video_probe) in enumerate(matched):
        task_index = task_to_index.setdefault(record.instruction, len(task_to_index))
        jobs.append(
            ConversionJob(
                episode_index=episode_index,
                task_index=task_index,
                record=record,
                video_path=video_path,
                video_probe=video_probe,
            )
        )
    return jobs, task_to_index, missing_videos, short_videos


def resample_path(path_xy: np.ndarray, count: int) -> np.ndarray:
    """Arc-length-resample a world-space path to one position per video frame."""
    if count <= 0:
        raise ValueError(f"Path sample count must be positive, got {count}")
    points = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        raise ValueError("Cannot resample an empty path")

    if len(points) > 1:
        keep = np.concatenate(
            [[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
        )
        points = points[keep]
    if len(points) == 1:
        return np.repeat(points, count, axis=0)

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    targets = np.linspace(0.0, float(cumulative[-1]), count)
    return np.column_stack(
        [
            np.interp(targets, cumulative, points[:, 0]),
            np.interp(targets, cumulative, points[:, 1]),
        ]
    )


def path_headings(path_xy: np.ndarray, start_heading_degrees: float) -> np.ndarray:
    count = len(path_xy)
    headings = np.full(count, math.radians(start_heading_degrees), dtype=np.float64)
    if count <= 1:
        return headings

    deltas = np.diff(path_xy, axis=0)
    last_heading = headings[0]
    for index, delta in enumerate(deltas, start=1):
        if float(np.linalg.norm(delta)) > 1e-9:
            last_heading = math.atan2(float(delta[1]), float(delta[0]))
        headings[index] = last_heading
    headings[0] = math.radians(start_heading_degrees)
    return np.unwrap(headings)


def build_camera_extrinsics(
    path_xy: np.ndarray,
    start_heading_degrees: float,
    camera_height: float,
) -> np.ndarray:
    """Construct camera-to-world transforms in the convention used by WNM-3D."""
    headings = path_headings(path_xy, start_heading_degrees)
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], len(path_xy), axis=0)

    for index, ((x, y), heading) in enumerate(zip(path_xy, headings)):
        forward = np.array(
            [math.cos(float(heading)), math.sin(float(heading)), 0.0],
            dtype=np.float32,
        )
        lateral = np.array([-forward[1], forward[0], 0.0], dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        extrinsics[index, :3, :3] = np.column_stack([lateral, up, forward])
        extrinsics[index, :3, 3] = [float(x), float(y), camera_height]
    return extrinsics


def load_rendered_camera_extrinsics(
    video_path: Path,
    source_indices: list[int],
) -> np.ndarray | None:
    """Load exact renderer poses when frame camera sidecars are available."""
    pose_directory = video_path.with_suffix("")
    if not pose_directory.is_dir():
        return None

    extrinsics: list[np.ndarray] = []
    for source_index in source_indices:
        camera_path = pose_directory / f"frame_{source_index:04d}_camera.json"
        if not camera_path.is_file():
            raise FileNotFoundError(
                f"Incomplete rendered camera metadata: missing {camera_path}"
            )
        with camera_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        matrix = np.asarray(payload.get("camera_to_world"), dtype=np.float32)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"camera_to_world must be 4x4 in {camera_path}, got {matrix.shape}"
            )
        # GN0 camera sidecars store row-vector transforms. WNM-3D consumes
        # conventional column-vector camera-to-world matrices.
        extrinsics.append(matrix.T.copy())
    return np.stack(extrinsics, axis=0)


def read_sampled_frames(
    video_path: Path,
    *,
    frame_count: int,
    frame_stride: int,
    output_width: int,
    output_height: int,
) -> tuple[list[int], list[np.ndarray]]:
    target_indices = list(range(0, frame_count, frame_stride))
    target_set = set(target_indices)
    frames: list[np.ndarray] = []
    decoded_indices: list[int] = []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open rendered video: {video_path}")
    try:
        source_index = 0
        while source_index <= target_indices[-1]:
            ok, frame = capture.read()
            if not ok:
                break
            if source_index in target_set:
                if frame.shape[1] != output_width or frame.shape[0] != output_height:
                    frame = cv2.resize(
                        frame,
                        (output_width, output_height),
                        interpolation=cv2.INTER_AREA,
                    )
                decoded_indices.append(source_index)
                frames.append(np.ascontiguousarray(frame))
            source_index += 1
    finally:
        capture.release()

    if len(decoded_indices) != len(target_indices):
        raise RuntimeError(
            f"Decoded {len(decoded_indices)} of {len(target_indices)} requested frames "
            f"from {video_path}"
        )
    return decoded_indices, frames


def episode_output_paths(output_root: Path, episode_index: int) -> tuple[Path, Path]:
    chunk_index = episode_index // CHUNK_SIZE
    parquet_path = (
        output_root
        / "data"
        / f"chunk-{chunk_index:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    video_path = (
        output_root
        / "videos"
        / f"chunk-{chunk_index:03d}"
        / VIDEO_KEY
        / f"episode_{episode_index:06d}.mp4"
    )
    return parquet_path, video_path


def is_h264_video_codec(codec: str) -> bool:
    return codec.lower() in H264_VIDEO_CODECS


def canonical_video_codec(codec: str) -> str:
    return "h264" if is_h264_video_codec(codec) else codec


def write_video_atomic(
    output_path: Path,
    frames: list[np.ndarray],
    *,
    fps: float,
    codec: str,
) -> None:
    if not frames:
        raise ValueError(f"No frames to write for {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.mp4")
    height, width = frames[0].shape[:2]

    if is_h264_video_codec(codec):
        if width % 2 != 0 or height % 2 != 0:
            raise ValueError(
                "H.264 yuv420p output requires even frame dimensions; "
                f"got {width}x{height} for {output_path}"
            )

        import imageio.v2 as imageio

        writer = imageio.get_writer(
            str(temporary_path),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            pixelformat="yuv420p",
        )
        try:
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    raise ValueError(
                        f"Inconsistent frame shape while writing {output_path}"
                    )
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            writer.close()
        os.replace(temporary_path, output_path)
        return

    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to initialize codec {codec!r} for {temporary_path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Inconsistent frame shape while writing {output_path}"
                )
            writer.write(frame)
    finally:
        writer.release()
    os.replace(temporary_path, output_path)


def convert_job(job: ConversionJob, settings: ConversionSettings) -> dict:
    cv2.setNumThreads(0)
    output_width = max(1, int(round(job.video_probe.width * settings.resize_scale)))
    output_height = max(1, int(round(job.video_probe.height * settings.resize_scale)))
    source_indices, frames = read_sampled_frames(
        job.video_path,
        frame_count=job.video_probe.frame_count,
        frame_stride=settings.frame_stride,
        output_width=output_width,
        output_height=output_height,
    )

    rendered_extrinsics = None
    if settings.pose_source != "annotations":
        rendered_extrinsics = load_rendered_camera_extrinsics(
            job.video_path, source_indices
        )
    if rendered_extrinsics is not None:
        extrinsics = rendered_extrinsics
        resolved_pose_source = "rendered_camera_json"
    else:
        if settings.pose_source == "rendered":
            raise FileNotFoundError(
                "Rendered camera metadata was requested but not found beside "
                f"{job.video_path}"
            )
        full_path = resample_path(
            np.asarray(job.record.path_xy, dtype=np.float64),
            job.video_probe.frame_count,
        )
        sampled_path = full_path[np.asarray(source_indices, dtype=np.int64)]
        extrinsics = build_camera_extrinsics(
            sampled_path,
            start_heading_degrees=job.record.start_heading_degrees,
            camera_height=settings.camera_height,
        )
        resolved_pose_source = "annotation_parquet.path_raster_world"

    length = len(frames)
    task_indices = np.full(length, job.task_index, dtype=np.int64)
    dataframe = pd.DataFrame(
        {
            EXTRINSIC_KEY: [matrix.tolist() for matrix in extrinsics],
            "episode_index": np.full(length, job.episode_index, dtype=np.int64),
            "frame_index": np.arange(length, dtype=np.int64),
            "task_index": task_indices,
            LANGUAGE_KEY: task_indices,
        }
    )

    parquet_path, video_path = episode_output_paths(
        settings.output_root, job.episode_index
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_parquet = parquet_path.with_name(
        f".{parquet_path.stem}.{os.getpid()}.tmp.parquet"
    )
    dataframe.to_parquet(temporary_parquet, index=False)
    os.replace(temporary_parquet, parquet_path)

    output_fps = job.video_probe.fps / settings.frame_stride
    write_video_atomic(
        video_path,
        frames,
        fps=output_fps,
        codec=settings.video_codec,
    )
    return {
        "episode_index": job.episode_index,
        "tasks": [job.record.instruction],
        "length": length,
        "scene": job.record.scene,
        "trajectory": job.record.trajectory,
        "source_video": str(job.video_path),
        "source_frame_stride": settings.frame_stride,
        "source_video_fps": job.video_probe.fps,
        "pose_source": resolved_pose_source,
        "video_fps": output_fps,
        "video_height": output_height,
        "video_width": output_width,
    }


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary_path, path)


def load_resume_prefix(output_root: Path, jobs: list[ConversionJob]) -> list[dict]:
    episodes_path = output_root / "meta/episodes.jsonl"
    rows = sorted(
        read_jsonl(episodes_path), key=lambda row: int(row.get("episode_index", -1))
    )
    completed: list[dict] = []
    for expected_index, row in enumerate(rows):
        if expected_index >= len(jobs):
            break
        if int(row.get("episode_index", -1)) != expected_index:
            break
        job = jobs[expected_index]
        if (
            str(row.get("scene")) != job.record.scene
            or normalize_trajectory_id(row.get("trajectory")) != job.record.trajectory
        ):
            raise ValueError(
                "Cannot resume because the annotation/video ordering changed at "
                f"episode {expected_index}. Use a new output directory."
            )
        parquet_path, video_path = episode_output_paths(output_root, expected_index)
        if not parquet_path.is_file() or not video_path.is_file():
            break
        completed.append(row)
    return completed


def output_contains_data(output_root: Path) -> bool:
    for relative in ("meta/episodes.jsonl", "meta/info.json"):
        if (output_root / relative).exists():
            return True
    for directory in (output_root / "data", output_root / "videos"):
        if directory.is_dir() and next(directory.rglob("*"), None) is not None:
            return True
    return False


def write_metadata(
    *,
    output_root: Path,
    episodes: list[dict],
    task_to_index: dict[str, int],
    render_root: Path,
    annotations_path: Path,
    instruction_column: str,
    resize_scale: float,
    camera_height: float,
    video_codec: str,
) -> None:
    if not episodes:
        raise RuntimeError("Cannot write metadata for an empty conversion")

    tasks = [
        {"task_index": index, "task": task}
        for task, index in sorted(task_to_index.items(), key=lambda item: item[1])
    ]
    meta_dir = output_root / "meta"
    write_jsonl_atomic(meta_dir / "tasks.jsonl", tasks)
    write_jsonl_atomic(meta_dir / "episodes.jsonl", episodes)

    first = episodes[0]
    if any(
        episode["video_height"] != first["video_height"]
        or episode["video_width"] != first["video_width"]
        or not math.isclose(
            float(episode["video_fps"]),
            float(first["video_fps"]),
            rel_tol=1e-6,
        )
        for episode in episodes[1:]
    ):
        raise ValueError("Converted episodes do not share one video shape and FPS")

    total_episodes = len(episodes)
    info = {
        "codebase_version": "v2.0",
        "robot_type": "interiorgs",
        "total_episodes": total_episodes,
        "total_frames": sum(int(episode["length"]) for episode in episodes),
        "total_tasks": len(tasks),
        "total_videos": 1,
        "total_chunks": (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": float(first["video_fps"]),
        "splits": {"train": "0:100"},
        "data_path": (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [first["video_height"], first["video_width"], 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(first["video_fps"]),
                    "video.codec": canonical_video_codec(video_codec),
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            EXTRINSIC_KEY: {
                "dtype": "float32",
                "shape": [4, 4],
                "names": ["row", "column"],
            },
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            LANGUAGE_KEY: {"dtype": "int64", "shape": [1]},
        },
        "conversion": {
            "render_root": str(render_root.resolve()),
            "annotations": str(annotations_path.resolve()),
            "instruction_column": instruction_column,
            "pose_sources": sorted(
                {str(episode.get("pose_source", "unknown")) for episode in episodes}
            ),
            "resize_scale": resize_scale,
            "camera_height": camera_height,
        },
    }
    modality = {
        "video": {"rgb": {"original_key": VIDEO_KEY}},
        "camera": {"extrinsic": {"original_key": EXTRINSIC_KEY}},
        "annotation": {"language.language_instruction": {}},
    }

    for name, payload in (("info.json", info), ("modality.json", modality)):
        path = meta_dir / name
        temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert GN0-rendered InteriorGS videos and GN-Matrix annotations "
            "to the LeRobot layout consumed by WNM-3D."
        )
    )
    parser.add_argument(
        "--render-root",
        type=Path,
        required=True,
        help="GN0 rendering root containing <scene>/<trajectory_id>.mp4.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="InteriorGS_train_seen.parquet from GN0.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination LeRobot dataset root.",
    )
    parser.add_argument(
        "--instruction-column",
        default="instruction",
        help="Parquet column used as the language instruction.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=3,
        help="Keep every Nth source video frame (default: 3).",
    )
    parser.add_argument(
        "--resize-scale",
        type=float,
        default=1.0,
        help="Spatial scale applied while re-encoding videos (default: 1.0).",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=1.0,
        help="Constant z coordinate used in reconstructed extrinsics.",
    )
    parser.add_argument(
        "--pose-source",
        choices=("auto", "rendered", "annotations"),
        default="auto",
        help=(
            "Camera pose source. 'auto' prefers rendered frame camera JSON and "
            "otherwise reconstructs poses from the annotation path."
        ),
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=10.0,
        help="Source FPS used only when a video does not report one.",
    )
    parser.add_argument(
        "--min-output-frames",
        type=int,
        default=9,
        help="Skip trajectories shorter than this after temporal subsampling.",
    )
    parser.add_argument(
        "--video-codec",
        default=DEFAULT_VIDEO_CODEC,
        help=(
            "Output video encoder. H.264 aliases h264/x264/avc1/libx264 use "
            "FFmpeg with yuv420p; other values must be a four-character "
            "OpenCV FourCC (default: libx264)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of conversion worker processes (default: 4).",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=None,
        help="Convert only the first N matched trajectories for validation.",
    )
    parser.add_argument(
        "--require-all-videos",
        action="store_true",
        help="Fail if any parquet trajectory is missing its rendered video.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the contiguous prefix recorded in meta/episodes.jsonl.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.render_root.is_dir():
        raise NotADirectoryError(args.render_root)
    if not args.annotations.is_file():
        raise FileNotFoundError(args.annotations)
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.resize_scale <= 0:
        raise ValueError("--resize-scale must be positive")
    if args.fallback_fps <= 0:
        raise ValueError("--fallback-fps must be positive")
    if args.min_output_frames <= 0:
        raise ValueError("--min-output-frames must be positive")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if args.first_n is not None and args.first_n <= 0:
        raise ValueError("--first-n must be positive")
    if not is_h264_video_codec(args.video_codec) and len(args.video_codec) != 4:
        raise ValueError(
            "--video-codec must be an H.264 alias "
            "(h264/x264/avc1/libx264) or a four-character OpenCV FourCC"
        )


def main() -> int:
    args = parse_args()
    validate_args(args)
    records = load_trajectory_records(args.annotations, args.instruction_column)
    jobs, task_to_index, missing_videos, short_videos = build_jobs(
        records,
        args.render_root,
        frame_stride=args.frame_stride,
        resize_scale=args.resize_scale,
        fallback_fps=args.fallback_fps,
        min_output_frames=args.min_output_frames,
        first_n=args.first_n,
        require_all_videos=args.require_all_videos,
    )

    if output_contains_data(args.output_root) and not args.resume:
        raise FileExistsError(
            f"Output root already contains a dataset: {args.output_root}. "
            "Choose a new directory or pass --resume."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    completed = load_resume_prefix(args.output_root, jobs) if args.resume else []
    write_jsonl_atomic(args.output_root / "meta/episodes.jsonl", completed)

    settings = ConversionSettings(
        output_root=args.output_root,
        frame_stride=args.frame_stride,
        resize_scale=args.resize_scale,
        camera_height=args.camera_height,
        video_codec=args.video_codec,
        pose_source=args.pose_source,
    )
    remaining_jobs = jobs[len(completed) :]
    episodes = list(completed)

    print(
        f"[INFO] annotations={len(records)} matched={len(jobs)} "
        f"missing_videos={missing_videos} short_videos={short_videos} "
        f"resumed={len(completed)} workers={args.num_workers}"
    )
    episodes_path = args.output_root / "meta/episodes.jsonl"
    if args.num_workers == 1:
        results = (convert_job(job, settings) for job in remaining_jobs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.num_workers)
        results = executor.map(
            convert_job,
            remaining_jobs,
            itertools.repeat(settings),
            chunksize=1,
        )

    try:
        for result in results:
            episodes.append(result)
            with episodes_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"[OK] episode={result['episode_index']:06d} "
                f"source={result['scene']}/{result['trajectory']} "
                f"frames={result['length']}"
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    write_metadata(
        output_root=args.output_root,
        episodes=episodes,
        task_to_index=task_to_index,
        render_root=args.render_root,
        annotations_path=args.annotations,
        instruction_column=args.instruction_column,
        resize_scale=args.resize_scale,
        camera_height=args.camera_height,
        video_codec=args.video_codec,
    )
    print(
        f"[DONE] episodes={len(episodes)} tasks={len(task_to_index)} "
        f"output={args.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
