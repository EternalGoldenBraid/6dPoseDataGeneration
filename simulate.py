import os
import sys
import logging
import numpy as np
import kubric as kb
from kubric.renderer.blender import Blender as KubricBlender
from kubric.simulator.pybullet import PyBullet as KubricSimulator

import argparse

parser = argparse.ArgumentParser(prog='demo',
        description='Simulate.')

parser.add_argument('-s','--scene', dest='scene_name', default='robo_gears',
                    type=str, help='Scene name defined by assets/<scene_name>/<scene_name>.glb')
parser.add_argument('--framerate', dest='framerate', default='robo_gears',
                    type=int, help='Scene name defined by assets/<scene_name>/<scene_name>.glb')
parser.add_argument('--n_objects', dest='n_objects', default=3,
                    type=int, help='Number of objects to render from scene')
parser.add_argument('--n_frames', dest='n_frames', default=3,
                    type=int, help='Number of objects to render from scene')
parser.add_argument('--size', dest='img_size', default='640',
                    type=str, help='')

# Camera params
parser.add_argument("--camera", choices=["fixed_random", "linear_movement"],
                    default="fixed_random")
parser.add_argument("--max_camera_movement", type=float, default=0.0)
parser.add_argument("--max_motion_blur", type=float, default=1.0)

# Configuration for the source of the assets
parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")

args = parser.parse_args()

logging.basicConfig(level="INFO")  # < CRITICAL, ERROR, WARNING, INFO, DEBUG

rng = np.random.default_rng()
# --- create scene and attach a renderer and simulator

IMG_SIZE = {'640': (640,480), '256': (256,256), '64': (64, 64)}
scene = kb.Scene(resolution=IMG_SIZE[args.img_size])

scene.frame_end = args.n_frames   # < numbers of frames to render
#scene.frame_end = 60   # < numbers of frames to render
#scene.frame_end = 2   # < numbers of frames to render

scene.frame_rate = args.framerate  # < rendering framerate

#scene.step_rate = 240  # < simulation framerate
#scene.step_rate = 240  # < simulation framerate
scene.step_rate = 10*args.framerate  # < simulation framerate

motion_blur = rng.uniform(0, args.max_motion_blur)
renderer = KubricBlender(scene, motion_blur=motion_blur)
simulator = KubricSimulator(scene)

# --- background 
import bpy
kubasic = kb.AssetSource.from_manifest(args.kubasic_assets)
hdri_source = kb.AssetSource.from_manifest(args.hdri_assets)
backgrounds = list(hdri_source._assets.keys())
#
hdri_id = rng.choice(backgrounds)
background_hdri = hdri_source.create(asset_id=hdri_id)
background_name = background_hdri.filename.split(".")[0].split("/")[-2]
##background_hdri_filename = "assets/backgrounds/provence_studio_4k.exr"
#
# Add Dome object
dome = kubasic.create(asset_id="dome", name="dome", static=True, background=True,
        scale=0.1)
scene += dome
## Set the texture 
dome_blender = dome.linked_objects[renderer]
texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
texture_node.image = bpy.data.images.load(background_hdri.filename)

# --- populate the scene with objects, lights, cameras
floor_x = 1.0
floor_y = 1.0
floor_z = 0.1
#scene += kb.Cube(name="floor", scale=(30, 30, 0.1), position=(0, 0, -1.),
#scene += kb.Cube(name="floor", scale=(100, 100, 0.1), position=(0, 0, -1.),
scene += kb.Cube(name="floor", scale=(floor_x, floor_y, floor_z), position=(0, 0, -floor_z),
                 static=True)
#light_spawn_region = [[0, 0, 0], [0.1, 0.1, 0.3]]
#scene += kb.DirectionalLight(name="sun", position=(50, 50, 30),
#                             look_at=(0, 0, 0), intensity=1.5)
renderer._set_ambient_light_hdri(background_hdri.filename)
#scene.camera = kb.PerspectiveCamera(name="camera", position=(200, 100, 50),
scene.camera = kb.PerspectiveCamera(name="camera", position=(0.4, -0.3, 0.3),
                                    look_at=(0, 0, 0),
                                    #focal_length=5., sensor_width=8,
                                    )

# --- generates spheres randomly within a spawn region
#scene_name = "engine_parts"
scene_name = "robo_gears"
scene_name = args.scene_name

obj_source = kb.AssetSource.from_manifest(manifest_path=f"{scene_name}_manifest.json")
asset_names = list(obj_source._assets.keys())
n_objects = args.n_objects
#spawn_region = [[0, 0, 0], [floor_x, floor_y, 1]]
spawn_region = [[0, 0, 0], [0.1, 0.1, 0.1]]
#spawn_region = [[0, 0, 0], [1, 1, 1]]
material_name = rng.choice(["metal", "rubber", "other"])
for i in range(n_objects):
    #velocity = rng.uniform([-1, -1, 0], [1, 1, 0])
    new_obj = obj_source.create(asset_id=asset_names[rng.integers(0,len(asset_names))],
            #material=material,
            #velocity=velocity,
            scale=np.array([1/1000, 1/1000, 1/1000])
            )
    color_label, random_color = kb.randomness.sample_color("uniform_hue", rng)
    size_label, size = kb.randomness.sample_sizes("uniform", rng)

    if material_name == "metal":
        new_obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=1.0,
            roughness=0.2, ior=2.5)
        new_obj.friction = 0.4
        new_obj.restitution = 0.3
        new_obj.mass *= 2.7 * size**3

    elif material_name == "rubber":
        new_obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=0.,
            ior=1.25, roughness=0.7, specular=0.33)
        new_obj.friction = 0.8
        new_obj.restitution = 0.7
        new_obj.mass *= 1.1 * size**3
    else: 
        #new_obj.material= kb.PrincipledBSDFMaterial(color=kb.random_hue_color(rng=rng))
        new_obj.material= kb.PrincipledBSDFMaterial(color=random_color)

    scene += new_obj
    kb.move_until_no_overlap(new_obj, simulator, spawn_region=spawn_region)
    

# --- executes the simulation (and store keyframes)
collisions, animation = simulator.run()

#print("n_keys:", len(list(collisions.keys())))
#print(list(collisions.keys()))
#print(collisions[list(collisions.keys())[0]])
#import pdb; pdb.set_trace()

# --- renders the output
from pathlib import Path
import scripts.kubric_to_BOP as kb_to_bop
scene_path = Path("output","simulation",scene_name)
existing_file_idx = np.array(list(map(int,os.listdir(scene_path))))
if len(existing_file_idx) == 0:
    next_data_name = f"{0:05d}"
else:
    next_data_name = f"{existing_file_idx.max()+1:05d}"

output_path = scene_path / next_data_name
output_path.mkdir(exist_ok=True, parents=True)
renderer.save_state(str(output_path/"simulator.blend"))

# frames_dict['segmentation'] -> (n_frames, img_shape, 1)
frames_dict = renderer.render( return_layers = (
    "rgb", 
    #"backward_flow",
    #"forward_flow",
    "depth",
    "normal",
    "object_coordinates",
    "segmentation"),
    )

# --- Visibility?
kb.compute_visibility(frames_dict["segmentation"], scene.assets)
visible_foreground_assets = [asset for asset in scene.foreground_assets
        if np.max(asset.metadata["visibility"]) >0]

visible_foreground_assets = sorted(  # sort assets by their visibility
    visible_foreground_assets,
    key=lambda asset: np.sum(asset.metadata["visibility"]),
    reverse=True)

#kb.write_image_dict(frames_dict, str(output_path))
kb_to_bop.write_image_dict_bop(frames_dict, str(output_path))

# --- Reindex segmentation maps for bbox computation
frames_dict["segmentation"] = kb.adjust_segmentation_idxs(
            frames_dict["segmentation"], scene.assets, visible_foreground_assets)
scene.metadata["num_instances"] = len(visible_foreground_assets) 

# --- Bounding boxes
#kb.post_processing.compute_bboxes(frames_dict["segmentation"],
kb_to_bop.compute_bboxes(frames_dict["segmentation"],
        visible_foreground_assets)

#np.save(output_path/'data_masks', frames_dict['segmentation'])

# --- Output BOP style
to_BOP = True
if to_BOP:
    import json
    gt, gt_info = kb_to_bop.get_BOP_info(scene=scene, assets_subset=visible_foreground_assets)
    with open(output_path/"scene_gt.json", 'w') as f:
        json.dump(gt, f, indent=4)
    with open(output_path/"scene_gt_info.json", 'w') as f:
        json.dump(gt_info, f, indent=4)
    
    cam_info = kb.get_camera_info(scene.camera)
    scene_camera = {}
    depth_scale = 1.0 # TODO Where is this?
    for frame in list(gt.keys()):
        scene_camera[frame] = {'cam_K': cam_info['K'].tolist(), 'depth_scale': depth_scale}

    with open(output_path/'scene_camera.json', 'w') as f:
        json.dump(scene_camera, f, indent=4)
    
    #kb.write_json(filename=output_path / "camera.json", 
    #    data=kb_to_bop.camera_to_dense(cam_info['K']))

# --- Metadata
logging.info("Collecting and storing metadata for each object.")
kb.write_json(filename=output_path / "metadata.json", data={
    "flags": vars(args),
    "metadata": kb.get_scene_metadata(scene),
    "camera": kb.get_camera_info(scene.camera),
    "instances": kb.get_instance_info(scene, visible_foreground_assets),
})
#kb.write_json(filename=output_path / "events.json", data={
#    "collisions":  kb.process_collisions(
#        collisions, scene, assets_subset=visible_foreground_assets),
#})

#print >> sys.stderr, str("Saved outputs to:", output_path)
print(background_name)

kb.done()
