#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import mediapy as media
import numpy as np
import trimesh, pyrender
import torch
import tyro
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from typing_extensions import Literal, assert_never

from nerfstudio.cameras.camera_paths import (
    generate_ellipse_path,
    get_path_from_json,
    get_spiral_path,
)
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.base_datamanager import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.utils import install_checks
from nerfstudio.utils.rich_utils import ItersPerSecColumn
import time

CONSOLE = Console(width=120)


def _interpolate_trajectory(cameras: Cameras, num_views: int = 300):
    """calculate interpolate path"""

    c2ws = np.stack(cameras.camera_to_worlds.cpu().numpy())

    key_rots = Rotation.from_matrix(c2ws[:, :3, :3])
    key_times = list(range(len(c2ws)))
    slerp = Slerp(key_times, key_rots)
    interp = interp1d(key_times, c2ws[:, :3, 3], axis=0)
    render_c2ws = []
    for i in range(num_views):
        time = float(i) / num_views * (len(c2ws) - 1)
        cam_location = interp(time)
        cam_rot = slerp(time).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = cam_rot
        c2w[:3, 3] = cam_location
        render_c2ws.append(c2w)
    render_c2ws = torch.from_numpy(np.stack(render_c2ws, axis=0))

    # use intrinsic of first camera
    camera_path = Cameras(
        fx=cameras[0].fx,
        fy=cameras[0].fy,
        cx=cameras[0].cx,
        cy=cameras[0].cy,
        height=cameras[0].height,
        width=cameras[0].width,
        camera_to_worlds=render_c2ws[:, :3, :4],
        camera_type=cameras[0].camera_type,
    )
    return camera_path


def _render_trajectory_video(
    meshfile: Path,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: str,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
    merge_type: Literal["half", "concat"] = "half",
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
    """
    CONSOLE.print("[bold green]Creating trajectory video")
    images = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    # cameras = cameras.to(pipeline.device)
    width = cameras[0].width[0].item()
    height = cameras[0].height[0].item()

    # Initialize a new mesh set
    mesh = trimesh.load(str(meshfile))

    # Setup a Pyrender scene
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    # Create an offscreen renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    # Set the camera parameters (optional)
    camera = pyrender.IntrinsicsCamera(fx=cameras.fx[0], fy=cameras.fy[0], cx=cameras.cx[0], cy=cameras.cy[0])
    cam_node = scene.add(camera, pose=np.eye(4))  # Add camera to the scene

    # Generate a list of images for the video by rotating the mesh
    frames = []

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(
            text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
            show_speed=True,
        ),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=False, compact=False),
        TimeElapsedColumn(),
    )

    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):
            pose=cameras[camera_idx].camera_to_worlds.numpy()
            pose = np.vstack((pose,[0,0,0,1]))
            # Update camera position around the object
            scene.set_pose(cam_node, pose=pose)

            # Render the scene from the current camera angle
            color, _ = renderer.render(scene)

            # Convert the color array from RGB to BGR for OpenCV
            color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            # Store the frame
            frames.append(color_bgr)

    # Save the frames as a video using mediapy
    fps = len(frames) / seconds
    media.write_video(output_filename, frames, fps=fps)
   


@dataclass
class RenderTrajectory:
    """Load a checkpoint, render a trajectory, and save to a video file."""

    # Path to config YAML file.
    meshfile: Path
    # Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb", "normal"])
    #  Trajectory to render.
    traj: Literal["spiral", "filename", "interpolate", "ellipse"] = "spiral"
    # Scaling factor to apply to the camera image resolution.
    downscale_factor: int = 1
    # Filename of the camera path to render.
    camera_path_filename: Path = Path("camera_path.json")
    # Name of the output file.
    output_path: Path = Path("renders/output.mp4")
    # How long the video should be.
    seconds: float = 5.0
    # pfs of the video
    fps: int = 24
    # How to save output data.
    output_format: Literal["images", "video"] = "video"
    merge_type: Literal["half", "concat"] = "half"

    data: AnnotatedDataParserUnion = SDFStudioDataParserConfig()
    num_views: int = 300

    def main(self) -> None:
        """Main function."""

        install_checks.check_ffmpeg_installed()
        seconds = self.seconds
        if self.output_format == "video":
            assert str(self.output_path)[-4:] == ".mp4"

        if self.traj == "filename":
            with open(self.camera_path_filename, "r", encoding="utf-8") as f:
                camera_path = json.load(f)
            seconds = camera_path["seconds"]
            camera_path = get_path_from_json(camera_path)
        elif self.traj == "interpolate":
            # load training data and interpolate path
            outputs = self.data.setup()._generate_dataparser_outputs()
            camera_path = _interpolate_trajectory(cameras=outputs.cameras, num_views=self.num_views)
            seconds = camera_path.size / 24
        elif self.traj == "spiral":
            outputs = self.data.setup()._generate_dataparser_outputs()
            camera_path = get_spiral_path(camera=outputs.cameras, steps=self.num_views, radius=1.0)
            seconds = camera_path.size / 24
        elif self.traj == "ellipse":
            outputs = self.data.setup()._generate_dataparser_outputs()
            camera_path = generate_ellipse_path(cameras=outputs.cameras, n_frames=self.num_views, const_speed=False)
            seconds = camera_path.size / self.fps
        else:
            assert_never(self.traj)

        _render_trajectory_video(
            self.meshfile,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
            output_format=self.output_format,
            merge_type=self.merge_type,
        )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderTrajectory).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RenderTrajectory)  # noqa
