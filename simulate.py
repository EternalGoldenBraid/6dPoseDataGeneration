import logging
import numpy as np
import kubric as kb
from kubric.renderer.blender import Blender as KubricBlender
from kubric.simulator.pybullet import PyBullet as KubricSimulator

logging.basicConfig(level="INFO")  # < CRITICAL, ERROR, WARNING, INFO, DEBUG

# --- create scene and attach a renderer and simulator
scene = kb.Scene(resolution=(256, 256))
#scene = kb.Scene(resolution=(64, 64))

scene.frame_end = 30   # < numbers of frames to render
#scene.frame_end = 2   # < numbers of frames to render

scene.frame_rate = 6  # < rendering framerate

scene.step_rate = 240  # < simulation framerate
#scene.step_rate = 24  # < simulation framerate
renderer = KubricBlender(scene)
simulator = KubricSimulator(scene)

# --- populate the scene with objects, lights, cameras
#scene += kb.Cube(name="floor", scale=(30, 30, 0.1), position=(0, 0, -1.),
scene += kb.Cube(name="floor", scale=(300, 300, 0.1), position=(0, 0, -1.),
                 static=True)
scene += kb.DirectionalLight(name="sun", position=(-10, -0.5, 30),
                             look_at=(0, 0, 0), intensity=1.5)
#scene.camera = kb.PerspectiveCamera(name="camera", position=(2, -0.5, 90),
scene.camera = kb.PerspectiveCamera(name="camera", position=(100, -55., 70),
                                    look_at=(0, 0, 0))

# --- generates spheres randomly within a spawn region
scene_name = "engine_parts"

obj_source = kb.AssetSource.from_manifest(manifest_path=f"{scene_name}_manifest.json")
asset_names = list(obj_source._assets.keys())
rng = np.random.default_rng()
n_objects = 2
spawn_region = [[0, 0, 15], [1, 1, 20]]
for i in range(n_objects):
    velocity = rng.uniform([-1, -1, 0], [1, 1, 0])
    material = kb.PrincipledBSDFMaterial(color=kb.random_hue_color(rng=rng))
    
    new_obj = obj_source.create(asset_id=asset_names[rng.integers(0,len(asset_names))],
            material=material,
            #velocity=velocity,
            #scale=0.1
            )

    scene += new_obj
    kb.move_until_no_overlap(new_obj, simulator, spawn_region=spawn_region)
    

# --- executes the simulation (and store keyframes)
collision, animation = simulator.run()

# --- renders the output
from pathlib import Path
output_path = Path("output","simulation",scene_name)
output_path.mkdir(exist_ok=True, parents=True)
renderer.save_state(str(output_path/"simulator.blend"))
#renderer.save_state("output/simulation/scene_name/simulator.blend")
frames_dict = renderer.render()
#kb.write_image_dict(frames_dict, "output/simulation")
kb.write_image_dict(frames_dict, str(output_path))
