import os
import sys
import logging
import numpy as np
import kubric as kb
from kubric.renderer.blender import Blender as KubricBlender
from kubric.simulator.pybullet import PyBullet as KubricSimulator

from utils.utils import sample_point_in_half_sphere_shell

import argparse

INDOOR_BGS = [ 
"abandoned_workshop_02", "photo_studio_loft_hall",
"cayley_interior", "peppermint_powerplant_2", "pump_house",
"artist_workshop", "fireplace", "peppermint_powerplant",
"vintage_measuring_lab", "carpentry_shop_02", "glass_passage"
]


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
parser.add_argument('--scale', dest='obj_scale', default=1.,
                    type=float, help='Scale of objects ing glb file.')

# Camera params
parser.add_argument("--camera", choices=["fixed_random", "linear_movement"],
                    default="fixed_random")
parser.add_argument("--max_camera_movement", type=float, default=0.0)
parser.add_argument("--max_motion_blur", type=float, default=0.1)

# Configuration for the source of the assets
parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")


args = parser.parse_args()

logging.basicConfig(level="INFO")  # < CRITICAL, ERROR, WARNING, INFO, DEBUG

rng = np.random.default_rng()
compute_specular = lambda ior: ((ior - 1)/(ior + 1))**2/0.08
IMG_SIZE = {'640': (640,480), '256': (256,256), '64': (64, 64)}

# --- create scene and attach a renderer and simulator

scene = kb.Scene(resolution=IMG_SIZE[args.img_size])

scene.frame_end = args.n_frames   # < numbers of frames to render
scene.frame_start = 1
#scene.frame_end = 60   # < numbers of frames to render
#scene.frame_end = 2   # < numbers of frames to render

scene.frame_rate = args.framerate  # < rendering framerate

#scene.step_rate = 240  # < simulation framerate
#scene.step_rate = 240  # < simulation framerate
scene.step_rate = 100*args.framerate  # < simulation framerate

if args.max_motion_blur != 0.0:
    motion_blur = rng.uniform(0, args.max_motion_blur)
    #motion_blur = 0.1

renderer = KubricBlender(scene,
        motion_blur=motion_blur,
        use_denoising=True,
        adaptive_sampling=True,
        background_transparency=False,
        )
simulator = KubricSimulator(scene)

# --- background 

import bpy
kubasic = kb.AssetSource.from_manifest(args.kubasic_assets)

hdri_source = kb.AssetSource.from_manifest(args.hdri_assets)

backgrounds = list(hdri_source._assets.keys())
hdri_id = rng.choice(INDOOR_BGS)

use_indoor = True
if use_indoor:
    while hdri_id not in backgrounds:
        hdri_id = rng.choice(INDOOR_BGS)
else:
    hdri_id = rng.choice(backgrounds)

background_hdri = hdri_source.create(asset_id=hdri_id)
background_name = background_hdri.filename.split(".")[0].split("/")[-2]
##background_hdri_filename = "assets/backgrounds/provence_studio_4k.exr"
 
# Add Dome object
dome = kubasic.create(asset_id="dome", name="dome", static=True, background=True,
        scale=0.1)
scene += dome
dome.scale = 0.1

## Set the texture 
dome_blender = dome.linked_objects[renderer]
texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
texture_node.image = bpy.data.images.load(background_hdri.filename)

##hdri_source = kb.TextureSource('hdri_assets')
#hdri_source = kb.AssetSource('hdri_assets')
#breakpoint()
#backgrounds = list(hdri_source._assets.keys())
#dome = kb.assets.utils.add_hdri_dome(hdri_source, scene, None)


# --- populate the scene with objects, lights, cameras
#floor_x = 0.2
#floor_y = 0.2
floor_x = args.obj_scale*20
floor_y = args.obj_scale*20
floor_z = 0.03
#color_label, random_color = kb.randomness.sample_color("uniform_hue", rng)
color_label, random_color = kb.randomness.sample_color("clevr", rng)
ior = 1.3
floor_material = kb.PrincipledBSDFMaterial( 
        color=random_color, metallic=0.1, roughness=0.8, 
        ior=ior, specular=compute_specular(ior))
scene += kb.Cube(name="floor", static=True, 
        scale=(floor_x, floor_y, floor_z), position=(0, 0, floor_z), 
        material=floor_material)
#scene += kb.DirectionalLight(name="sun", position=(50, 50, 30),
#                             look_at=(0, 0, 0), intensity=1.5)
renderer._set_ambient_light_hdri(background_hdri.filename)


# --- generates objects randomly within a spawn region
#scene_name = "engine_parts"
scene_name = "robo_gears"
scene_name = args.scene_name

obj_source = kb.AssetSource.from_manifest(manifest_path=f"{scene_name}_manifest.json")
asset_names = list(obj_source._assets.keys())
n_objects = args.n_objects
#spawn_region = [[0, 0, 0], [floor_x, floor_y, 1]]
#spawn_region = [[0, 0, floor_z+0.01], [0.1, 0.1, floor_z+0.4]]
spawn_region = [[0, 0, floor_z+0.01], [floor_x+0.1, floor_y+0.1, floor_z+0.4]]
#material_name = rng.choice(["metal", "rubber", "plastic", "other"])
material_name = rng.choice([
    #"plastic",
    #"metal",
    "rubber",
    #"other"
    ])

for i in range(n_objects):
    #velocity = rng.uniform([-1, -1, 0], [1, 1, 0])
    new_obj = obj_source.create(asset_id=asset_names[rng.integers(0,len(asset_names))],
            #velocity=velocity,
            scale=np.ones(3)*args.obj_scale
            )
    obj_scale = np.linalg.norm(new_obj.aabbox[1] - new_obj.aabbox[0])
    #color_label, random_color = kb.randomness.sample_color("uniform_hue", rng)
    color_label, random_color = kb.randomness.sample_color("clevr", rng)
    #color_label, random_color = kb.randomness.sample_color("gray", rng)
    size_label, size = kb.randomness.sample_sizes("uniform", rng)

    if material_name == "metal":
        ior = 1.5
        specular = compute_specular(ior) 
        new_obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=1.0,
            roughness=0.2, ior=ior, specular=specular)
        new_obj.friction = 0.4
        new_obj.restitution = 0.3
        new_obj.mass *= 2.7 * size**3

    elif material_name == "rubber":
        ior = 1.25
        specular = compute_specular(ior) 
        new_obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=0.,
            ior=ior, roughness=0.7, 
            #specular=0.33,
            specular=specular,
            )
        new_obj.friction = 0.8
        new_obj.restitution = 0.7
        new_obj.mass *= 1.1 * size**3
    elif material_name == "plastic": 
        #ior = 1.25
        #specular = compute_specular(ior)
        new_obj.material= kb.PrincipledBSDFMaterial(color=random_color)
        new_obj.friction = 0.8
        new_obj.restitution = 0.7
        new_obj.mass *= 1.1 * size**3
    else:
        #new_obj.material= kb.PrincipledBSDFMaterial(color=kb.random_hue_color(rng=rng))
        new_obj.material = kb.FlatMaterial(color=random_color, 
                holdout=True, 
                #indirect_visibility=True
                )
        new_obj.friction = 0.8
        new_obj.restitution = 0.7
        new_obj.mass *= 1.1 * size**3

    scene += new_obj
    kb.move_until_no_overlap(new_obj, simulator, spawn_region=spawn_region)

# --- executes the simulation (and store keyframes)
animation, collisions = simulator.run()
obj_ids = animation.keys()

cam_pos_array = np.zeros([args.n_frames+1, 3])

for obj in obj_ids:
    if not hasattr(obj, 'asset_id'): continue
    if obj.asset_id == 'dome': continue
    cam_pos_array = cam_pos_array + np.array(animation[obj]["position"])
cam_pos_array = cam_pos_array / len(obj_ids)

# --- Keyframe the camera
scene.camera = kb.PerspectiveCamera(name='camera',
            #focal_length=5., sensor_width=8,
                                    )

#for frame in range(0, args.n_frames):
cam_lowest_z = floor_z+0.2
for frame in range(1, args.n_frames + 1):
    # scene.camera.position = (1, 1, 1)  #< frozen camera
    pos = sample_point_in_half_sphere_shell(
        inner_radius=0.1, outer_radius=0.6, rng=rng) # meters
    pos[-1] = rng.uniform(cam_lowest_z,cam_lowest_z+1.0)
    scene.camera.position = pos
    #breakpoint()
    scene.camera.look_at(cam_pos_array[frame-1])
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)

# --- renders the output
from pathlib import Path
import scripts.kubric_to_BOP as kb_to_bop
scene_path = Path("output","simulation",scene_name)
scene_path.mkdir(exist_ok=True, parents=True)
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
    "segmentation",
    #"backward_flow",
    #"forward_flow",
    #"depth",
    #"normal",
    #"object_coordinates",
    ),)

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
    "background": background_name,
})
#kb.write_json(filename=output_path / "events.json", data={
#    "collisions":  kb.process_collisions(
#        collisions, scene, assets_subset=visible_foreground_assets),
#})

#print >> sys.stderr, str("Saved outputs to:", output_path)
print(background_name)

kb.done()
