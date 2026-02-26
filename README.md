# SensorDataSim

Offline sensor simulation on Toronto-3D point cloud data (`L002`), exported as an MCAP file for Foxglove. The output contains a VLP-16 style LiDAR stream, a pinhole RGB camera stream, and a TF tree, with two dynamic cars added to the static scene.

Final demo video (LiDAR + camera side-by-side):
- https://www.youtube.com/watch?v=cVbzTud8lWQ

## What this project does

This project takes a real MLS point cloud (Toronto-3D) and turns it into a replayable sensor simulation. I used tile `L002` because it has a long straight road, enough parked cars, and a good mix of background structure (buildings, trees, poles, wires).

The final output is a 15-second MCAP that can be opened directly in Foxglove. It includes TF, LiDAR, and camera data. The camera stream in the final config is compressed (`/camera/image_raw/compressed`).

## Quick run

I developed this in a `conda` environment named `sensorsim` (Python 3.11). Main dependencies are `numpy`, `scipy`, `numba`, `open3d`, `opencv-python`, `plyfile`, `pyyaml`, and `mcap-ros2` / `mcap_ros2`.

Put the Toronto-3D files under `data/raw/Toronto_3D/` (at minimum `L002.ply`, `Mavericks_classes_9.txt`, `Colors.xml`), then run:

```powershell
conda run -n sensorsim python scripts/preprocess.py
conda run -n sensorsim python run_sim.py --output data/output/sim_output_submission_15s_compressed_20260226.mcap
```

Open in Foxglove:

```powershell
foxglove-studio data/output/sim_output_submission_15s_compressed_20260226.mcap
```

Foxglove notes: use `/camera/image_raw/compressed` for the image panel. For LiDAR, use `/velodyne_points` and set `Color By = intensity`.

## Configuration (sim.yaml)

Everything is configured in `sim.yaml`. The final run uses a 15 second timeline with `10 Hz` LiDAR, `10 Hz` camera, and `50 Hz` TF. The camera is a pinhole model at `1280x720` with `fx=fy=600`, and the final camera renderer runs with Gaussian depth-aware splatting plus `2x` supersampling. Camera output is written as compressed PNG images in a ROS `CompressedImage` topic.

Waypoints in the YAML are stored as `[Y, X]` (not `[X, Y]`). Sensor mounts are attached to `base_link` (`velodyne` at `z=1.8`, camera at `x=0.5, z=1.5`).

## Preprocessing (Toronto-3D L002)

The preprocessing script (`scripts/preprocess.py`) builds a runtime-friendly static scene from the raw Toronto-3D point cloud. In practical terms: it loads the tile and semantic labels, crops a road-centered ROI, fits a local ground plane, extracts cars, removes cars from the static background scene, and voxel-downsamples the remaining static geometry.

One important design choice is that I do not treat cars the same way as the rest of the scene. MLS cars are already sparse and often incomplete. If I downsample them aggressively, they look much worse in the camera view. So I keep a downsampled global static scene for performance and a separate full-resolution static-car subset for the camera renderer.

The current processed static scene is about `5M` points after downsampling (`0.05 m` voxel size), down from about `10M` points after ROI filtering.

## Dynamic cars

I built a small car asset pipeline (`scripts/build_car_asset_library.py`) using semantic car labels + DBSCAN clustering + simple geometry checks. The final dynamic asset is based on a car cluster from `L003`. It had one side more complete than the other, so I mirrored/fixed it to make a better symmetric dynamic model.

The final scene uses two dynamic cars:
- one oncoming car
- one same-direction car that passes the ego vehicle

## Trajectories (ego and agents)

Trajectories are spline-based (`sim/trajectory.py`). Waypoints are fit with cubic splines, parameterized by arc length, and then evaluated at constant speed (`s(t) = v * t`). Yaw is computed from the spline tangent.

A single global ground plane was not enough for vehicle Z placement. It made cars look sunk into the road in some segments and made the ego camera look too low. To fix this, I added a local road-height model (`sim/road_surface.py`) using a KD-tree over road-labeled points (`Ground` + `Road_markings`), with robust estimation (median + MAD clipping + Gaussian distance weighting).

I also smooth the road-height profile along the trajectory and estimate pitch from local road grade. This keeps vehicles visually attached to the road while avoiding camera shake from raw point noise.

## Simulation design (why these methods)

The simulator is event-driven (`run_sim.py` + `sim/timeline.py`) using integer nanosecond ticks to avoid timing drift between TF, LiDAR, and camera. At each timestamp, the code composes the scene once (static background + full-res static cars + dynamic agents) and reuses it for whichever sensors fire on that tick.

For LiDAR, I use a VLP-16 style "scatter-min" renderer (`sim/lidar.py` + `sim/numba_kernels.py`). Instead of ray casting, each point is transformed into LiDAR frame, assigned to a ring/azimuth cell, and the nearest point wins for that cell. This matches the structure of a spinning LiDAR scan and works directly on point cloud input without meshing.

For the camera, I use a pinhole point-splat renderer (`sim/camera.py`) rather than ray casting or voxelization. The source is already a colored point cloud, so splatting is a good fit and avoids mesh generation artifacts.

## Camera image improvements

The final camera renderer is not the original hard-circle splat renderer. It now uses a two-pass depth-aware Gaussian splat pipeline:

- Pass A builds a front depth buffer (nearest depth per pixel)
- Pass B accumulates Gaussian-weighted color only near that front surface

This reduces harsh splat edges and background bleed-through. I also added adaptive splat radii, a density-based radius boost in sparse regions, hybrid rendering for full-resolution static cars, and `2x` supersampling followed by downsampling (`INTER_AREA`).

Before/after camera comparison video (old renderer vs Gaussian + supersampling):
- https://www.youtube.com/watch?v=CS0ytRj9hzo

## Performance and optimization

I first made the simulation correct, then optimized it enough to iterate comfortably. The biggest speedups came from Numba kernels (`sim/numba_kernels.py`) for LiDAR scatter-min and camera projection/splatting. I also use KD-tree culling plus a small cull cache in `sim/scene_compose.py` to avoid re-querying the static scene from scratch every frame, and I reuse LiDAR/camera buffers to reduce allocation churn.

The final quality-focused submission run (Gaussian camera + 2x supersampling + compressed camera topic) takes about `5.8 min` for a `15s` simulation on my machine. A profiled run (`cProfile`) was used for hotspot analysis only. In the current design, camera Gaussian accumulation is the main runtime cost, followed by scene composition/culling. LiDAR is relatively cheap after the Numba scatter-min optimization.

I did not implement multithreading or a GPU renderer. For this project, CPU + Numba already reached a usable runtime, and GPU/parallel paths would add a lot of complexity (atomics/conflict handling for splats, data transfer/setup, portability issues for reviewers).

## Output and Foxglove playback

The final compressed-camera MCAP (`data/output/sim_output_submission_15s_compressed_20260226.mcap`) contains:
- `/tf_static`
- `/tf`
- `/velodyne_points`
- `/camera/image_raw/compressed`
- `/camera/camera_info`

LiDAR points are exported as ROS `PointCloud2` with `x, y, z, intensity, ring` (plus padding). I remap intensity for visualization so static points stay dark and dynamic cars are bright. In Foxglove this makes the moving cars stand out immediately when using `Color By = intensity`.

## Code structure (short version)

The core runtime lives in `sim/` (`camera.py`, `lidar.py`, `scene_compose.py`, `trajectory.py`, `tf_tree.py`, `timeline.py`, `mcap_writer.py`). `run_sim.py` is the orchestration entry point. The `scripts/` folder contains preprocessing and helper tools (preprocess, car asset extraction, ego fitting, agent planning, video generation).

## Limitations

This is a point-cloud simulation pipeline, not a full physical sensor simulator. I did not model motion blur, rolling shutter, weather, lighting changes, or complex agent behavior. Dynamic agents in the final config are based on a small car asset set (one mirrored car asset reused). These are reasonable next steps if more time is available.
