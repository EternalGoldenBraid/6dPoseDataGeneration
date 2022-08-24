import logging
import numpy as np
import kubric as kb
from kubric.renderer.blender import Blender as KubricBlender
from kubric.simulator.pybullet import PyBullet as KubricSimulator

logging.basicConfig(level="INFO")  # < CRITICAL, ERROR, WARNING, INFO, DEBUG

# --- create scene and attach a renderer and simulator
scene = kb.Scene(resolution=(256, 256))
scene.frame_end = 200   # < numbers of frames to render
#scene.frame_end = 2   # < numbers of frames to render
scene.frame_rate = 6  # < rendering framerate
scene.step_rate = 240  # < simulation framerate
renderer = KubricBlender(scene)
simulator = KubricSimulator(scene)

# --- populate the scene with objects, lights, cameras
scene += kb.Cube(name="floor", scale=(30, 30, 0.1), position=(0, 0, -1.),
                 static=True)
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3),
                             look_at=(0, 0, 0), intensity=1.5)
#scene.camera = kb.PerspectiveCamera(name="camera", position=(2, -0.5, 90),
scene.camera = kb.PerspectiveCamera(name="camera", position=(100, -25., 70),
                                    look_at=(0, 0, 0))

# --- generates spheres randomly within a spawn region
spawn_region = [[-1, -1, 15], [1, 1, 20]]
rng = np.random.default_rng()
for i in range(1):
    velocity = rng.uniform([-1, -1, 0], [1, 1, 0])
    material = kb.PrincipledBSDFMaterial(color=kb.random_hue_color(rng=rng))
    
    #if i % 2 == 0:
    if True:
        obj_source = kb.AssetSource.from_manifest(manifest_path="test_manifest.json")
        new_obj = obj_source.create(asset_id="gear_assembled")

        scene += new_obj
        kb.move_until_no_overlap(new_obj, simulator, spawn_region=spawn_region)
    
    else:
        sphere = kb.Sphere(scale=1, velocity=velocity, material=material)
        scene += sphere
        kb.move_until_no_overlap(sphere, simulator, spawn_region=spawn_region)


# --- executes the simulation (and store keyframes)
simulator.run()

# --- renders the output
renderer.save_state("output/simulation/simulator.blend")
frames_dict = renderer.render()
kb.write_image_dict(frames_dict, "output/simulation")
