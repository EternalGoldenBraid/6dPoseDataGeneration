import os
import sys
import json
from pathlib import Path
import logging
import bpy
import numpy as np
import kubric as kb
from kubric.renderer.blender import Blender as KubricBlender
from kubric.simulator.pybullet import PyBullet as KubricSimulator

import scripts.kubric_to_BOP as kb_to_bop

from utils.utils import sample_point_in_half_sphere_shell

import argparse

INDOOR_BGS = [ 
#abandoned_workshop_02",
#photo_studio_loft_hall",
#cayley_interior",
#peppermint_powerplant_2",
#pump_house",
#"artist_workshop",
#"fireplace",
#"peppermint_powerplant",
"vintage_measuring_lab",
#"carpentry_shop_02",
#"glass_passage"
]


def get_simulator_renderer(args, rng, 
        use_denoising=True, 
        adaptive_sampling=True,
        background_transparency=False):

    IMG_SIZE = {'640': (640,480), '256': (256,256), '64': (64, 64)}

    # --- create scene and attach a renderer and simulator
    
    frame_rate = int(args.n_frames/args.simulation_time)# < rendering framerate
    
    #scene.frame_end = args.n_frames   # < numbers of frames to render
    #scene.frame_start = 0
    ##scene.frame_rate = args.framerate  # < rendering framerate
    ## < simulation framerate. Needs to be multiple of framerate. Default 240
    #breakpoint()
    #scene.step_rate = 1*frame_rate  
    #scene.frame_rate = framerate# < rendering framerate

    scene = kb.Scene(resolution=IMG_SIZE[args.img_size],
            frame_start=0, frame_end=args.n_frames,
            frame_rate=frame_rate, step_rate=5*frame_rate)
    
    
    if args.max_motion_blur != 0.0:
        motion_blur = rng.uniform(0, args.max_motion_blur)
    
    renderer = KubricBlender(scene,
            motion_blur=motion_blur,
            use_denoising=use_denoising,
            adaptive_sampling=adaptive_sampling,
            background_transparency=background_transparency
            )
    simulator = KubricSimulator(scene)

    return simulator, renderer, scene

def add_background(renderer, scene, args, rng, use_indoor=True):

    hdri_source = kb.AssetSource.from_manifest(args.hdri_assets) 
    background_names = list(hdri_source._assets.keys())
    hdri_id = rng.choice(INDOOR_BGS)
    #hdri_id = rng.choice(["artist_workshop", 
    #    "fireplace", 
    #    "peppermint_powerplant", 
    #    "photo_studio_loft_hall",
    #    ])
    
    use_indoor = True
    if use_indoor:
        while hdri_id not in background_names:
            hdri_id = rng.choice(INDOOR_BGS)
    else:
        hdri_id = rng.choice(background_names)
    
    background_hdri = hdri_source.create(asset_id=hdri_id)
    scene.background_name = background_hdri.filename.split(".")[0].split("/")[-2]
     
    # Add Dome object
    kubasic = kb.AssetSource.from_manifest(args.kubasic_assets)
    dome = kubasic.create(asset_id="dome", name="dome", static=True, background=True,
            scale=0.04,
            #scale=0.05,
            #scale=0.20,
            position=[0.0,0.0,-0.01]
            )

    scene += dome
    #dome.scale = 0.01
    print("Dome scale:",dome.scale)
    print("object scale:",args.obj_scale)
    
    ## Set dome texture 
    dome_blender = dome.linked_objects[renderer]
    texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
    texture_node.image = bpy.data.images.load(background_hdri.filename)

    #scene += kb.DirectionalLight(name="sun", position=(50, 50, 30),
    #                             look_at=(0, 0, 0), intensity=1.5)
    renderer._set_ambient_light_hdri(background_hdri.filename)

def populate_scene(scene, simulator, rng, args, scene_name="robo_gears"):

    compute_specular = lambda ior: ((ior - 1)/(ior + 1))**2/0.08
    # --- populate the scene with objects, lights, cameras
    floor_x = 0.0
    floor_y = 0.0
    #floor_x = args.obj_scale*100
    #floor_y = args.obj_scale*100
    floor_z = 0.00
    #color_label, random_color = kb.randomness.sample_color("uniform_hue", rng)
    color_label, random_color = kb.randomness.sample_color("clevr", rng)
    ior = 1.3

    #floor_material = kb.PrincipledBSDFMaterial( 
    #        color=random_color, metallic=0.1, roughness=0.8, 
    #        ior=ior, specular=compute_specular(ior))
    ##floor_material = kb.Texture
    #scene += kb.Cube(name="floor", static=True, 
    #        scale=(floor_x, floor_y, floor_z), position=(0, 0, floor_z), 
    #        material=floor_material)
    
    # --- generates objects randomly within a spawn region
    scene_name = args.scene_name
    obj_source = kb.AssetSource.from_manifest(manifest_path=f"{scene_name}_manifest.json")
    asset_names = list(obj_source._assets.keys())
    n_objects = args.n_objects
    #spawn_region = [[0, 0, 0], [floor_x, floor_y, 1]]
    #spawn_region = [[0, 0, floor_z+0.01], [0.1, 0.1, floor_z+0.4]]
    spawn_region = [[0, 0, floor_z+0.01], [floor_x+0.2, floor_y+0.2, floor_z+0.2]]
    material_name = rng.choice([
        #"plastic",
        "metal",
        "rubber",
        #"other"
        ])
    
    for i in range(n_objects):
        #velocity = rng.uniform([-1, -1, 0], [1, 1, 0])
        #angular_velocity = rng.uniform([0, 0, 0], [10, 10, 10])
        angular_velocity = [0, 20, 20]
        #new_obj = obj_source.create(asset_id=asset_names[rng.integers(0,len(asset_names))],
        new_obj = obj_source.create(asset_id="Top_casing",
                #velocity=velocity,
                angular_velocity = angular_velocity,
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
            new_obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=0.7,
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
        new_obj.position = np.array([0, 0, 0.3])
        #kb.move_until_no_overlap(new_obj, simulator, spawn_region=spawn_region)

    scene.cam_lowest_z = floor_z+0.1

def store_keyframes(animation, args, rng, scene, renderer, to_BOP=True):
    """
    Keyframe the camera based on simulated animation and save
    results to disk.
    to_BOP: Output also BOP format json files see 
        https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.mdBOP
        for details.
    """

    # TODO DEPTH OF FIELD
    # https://docs.blender.org/api/current/bpy.types.CameraDOFSettings.html

    # Set camera to point to mean of object positions in each frame.
    obj_ids = animation.keys()
    obj_names = []
    cam_pos_array = np.zeros([args.n_frames+1, 3])
    for obj in obj_ids:
        if not hasattr(obj, 'asset_id'): continue
        if obj.asset_id == 'dome': continue
        obj_names.append(obj.asset_id)
        cam_pos_array = cam_pos_array + np.array(animation[obj]["position"])
    cam_pos_array = cam_pos_array / len(obj_names)
    print(cam_pos_array)

    # --- Keyframe the camera
    scene.camera = kb.PerspectiveCamera(name='camera')
    
    #cam_lowest_z = floor_z+0.2
    cam_lowest_z = scene.cam_lowest_z
    for frame in range(scene.frame_start, scene.frame_end+1):
        # scene.camera.position = (1, 1, 1)  #< frozen camera
        pos = sample_point_in_half_sphere_shell(
            inner_radius=0.1, outer_radius=0.5, rng=rng) # meters

        pos[-1] = rng.uniform(cam_lowest_z,cam_lowest_z+0.5)
        scene.camera.position = pos
        #scene.camera.position = np.array([0+frame, -0.5, 0.5])
        scene.camera.look_at(cam_pos_array[frame])
        scene.camera.keyframe_insert("position", frame)
        scene.camera.keyframe_insert("quaternion", frame)

    # --- renders the output
    scene_path = Path("output","simulation",scene.scene_name)
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
        "background": scene.background_name,
        "obj_ids": obj_names,
    })
    #kb.write_json(filename=output_path / "events.json", data={
    #    "collisions":  kb.process_collisions(
    #        collisions, scene, assets_subset=visible_foreground_assets),
    #})
    
    #print >> sys.stderr, str("Saved outputs to:", output_path)

    from PIL import Image
    print("Saving gif to:", output_path)
    #import cv2
    #imgs = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mode='RGB') for frame in frames_dict['rgb']]
    imgs = [Image.fromarray(frame) for frame in frames_dict['rgb']]
    # duration is the number of milliseconds between frames; 
    imgs[0].save(output_path/'rgb.gif', save_all=True, append_images=imgs[1:], duration=int(1000/scene.frame_rate), loop=0)
    print("Done")

    return output_path

def main(raw_args=None):

    parser = argparse.ArgumentParser(prog='demo',
            description='Simulate.')
    
    parser.add_argument('-s','--scene', dest='scene_name', default='robo_gears',
                        type=str, help='Scene name defined by assets/<scene_name>/<scene_name>.glb')
    #parser.add_argument('--framerate', dest='framerate', default='robo_gears',
    #                    type=int, help='Scene name defined by assets/<scene_name>/<scene_name>.glb')
    parser.add_argument('--n_objects', dest='n_objects', default=3,
                        type=int, help='Number of objects to render from scene')
    parser.add_argument('--n_frames', dest='n_frames', default=3,
                        type=int, help='Number of objects to render from scene')
    parser.add_argument('--size', dest='img_size', default='640',
                        type=str, help='')
    parser.add_argument('--simulation_time', dest='simulation_time', default=5,
                        type=int, help='')
    parser.add_argument('--sim_step_multiplier', dest='sim_step', default=1,
                        type=int, help='')
    parser.add_argument('--scale', dest='obj_scale', default=1.,
                        type=float, help='Scale of objects ing glb file.')

    # Camera params
    parser.add_argument("--camera", choices=["fixed_random", "linear_movement"],
                        default="fixed_random")
    parser.add_argument("--max_camera_movement", type=float, default=0.0)
    parser.add_argument("--max_motion_blur", type=float, default=0.05)
    
    # Configuration for the source of the assets
    parser.add_argument("--kubasic_assets", type=str,
                        default="gs://kubric-public/assets/KuBasic/KuBasic.json")
    parser.add_argument("--hdri_assets", type=str,
                        default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
    
    args = parser.parse_args()

    logging.basicConfig(level="INFO")  # < CRITICAL, ERROR, WARNING, INFO, DEBUG

    rng = np.random.default_rng()

    print("Initializing simulator.")
    simulator, renderer, scene = get_simulator_renderer(args=args, rng=rng,
        adaptive_sampling=True, use_denoising=True, background_transparency=False)
    scene.scene_name = args.scene_name
    print("Adding background.")
    add_background(renderer=renderer, args=args, scene=scene, use_indoor=True, rng=rng)
    print("Populating scene.")
    populate_scene(scene=scene, simulator=simulator, 
            rng=rng, args=args, scene_name=args.scene_name)

    # --- executes the simulation (and store keyframes)
    print("Simulating.")
    animation, collisions = simulator.run()

    print("Rendering keyframes.")
    store_keyframes(animation=animation, args=args, renderer=renderer,
            scene=scene, to_BOP=True, rng=rng)

    kb.done()

    #https://stackoverflow.com/questions/44734858/python-calling-a-module-that-uses-argparser

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
        main()
