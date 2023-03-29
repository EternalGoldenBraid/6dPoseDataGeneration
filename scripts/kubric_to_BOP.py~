import os
import numpy as np
import copy
import threading

from pathlib import Path
from typing import Sequence
from kubric import core
from kubric.kubric_typing import ArrayLike, PathLike
from typing import Any, Dict

import trimesh.transformations as tritrans

def compute_bboxes(segmentation: ArrayLike, asset_list: Sequence[core.Asset]):
    for k, asset in enumerate(asset_list, start=1):
        asset.metadata["bboxes"] = []
        asset.metadata["bbox_frames"] = []
        for t in range(segmentation.shape[0]):
            seg = segmentation[t, ..., 0]
            idxs = np.array(np.where(seg == k), dtype=np.float32)
            if idxs.size > 0:
                y_min = float(idxs[0].min() / seg.shape[0])
                x_min = float(idxs[1].min() / seg.shape[1])
                y_max = float((idxs[0].max() + 1) / seg.shape[0])
                x_max = float((idxs[1].max() + 1) / seg.shape[1])
                asset.metadata["bboxes"].append((y_min, x_min, y_max, x_max))
                asset.metadata["bbox_frames"].append(t)
            else:
                asset.metadata["bboxes"].append((-1, -1, -1, -1))
                asset.metadata["bbox_frames"].append(t)

def extract_frame_data(instance, frame_range: range, scene,
        image_shape: (int, int) = (640,480)):
    """
    Extract frame per frame information for an instance of an object.
    return:
        gt_frames_data [dict{rotation, translation}]: object pose per frame as list of dicts
        gt_info_frames_data [dict{bbox, ***}]: object bbox data per frame as list of dicts
    """
    import bpy

    info = copy.copy(instance.metadata)
    assert len(info['bboxes']) == len(frame_range)

    gt_frames_data = []
    gt_info_frames_data = []
    w = image_shape[0]
    h = image_shape[1]

    for frame_idx, frame in enumerate(frame_range):
        #if hasattr(instance, "asset_id"):
        with scene.camera.at_frame(frame):
            cam_t = scene.camera.position
            cam_R = tritrans.quaternion_matrix(
                    scene.camera.quaternion)[:3,:3]

            # Points as rows, world matrix columns are subspace axis.
            # Map from world coordinates to camera coordinates.
            """
            W = [
                [Rxx, Rxy, Rxz, Tx]
                [Ryx, Ryy, Ryz, Ty]
                [Rzx, Rzy, Rzz, Tz]
                [0,   0,   0,   1 ]
                ]
            """
            W_inv = np.linalg.inv(scene.camera.matrix_world) 
            #print("cam world:", scene.camera.matrix_world)
            #print("Cam_POS:", scene.camera.position)
            #print("homo:", W_inv)

            assert len(W_inv.shape) == 2 and W_inv.shape[0] == 4 and W_inv.shape[1] == 4

        with instance.at_frame(frame):

            R = tritrans.quaternion_matrix(instance.quaternion)[:3,:3]

            # TODO
            # Can't get change in camera pose for this one. 
            # Only estimates with final cam pose.
            #cam = bpy.data.objects['camera']
            #model_pose_cam_local = np.array(
            #        cam.convert_space(matrix=instance.matrix_world, to_space='LOCAL'))
            #print("LOCAL")
            #print(model_pose_cam_local)

            # A circumvention of the above problem.
            # Map object world pose to camera coordinates.
            obj_pose = np.eye(4, dtype=float)
            obj_pose[:3,:3] = R
            obj_pose[:3,3] = instance.position
            obj_in_cam_coords = W_inv @ obj_pose

            #print("OLD T:", cam_R.T.dot((-cam_t + instance.position)))
            #print("NEW T:", obj_in_cam_coords[:3,3])
            #print("OLD R:", cam_R.T.dot(R))
            #print("NEW R:", obj_in_cam_coords[:3,:3])

            ### Coordinate transformation from kubric camera frame to BOP camera frame.
            kub_2_bop_frame = np.array([
                [ 1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0, -1],
            ], dtype=np.float32)

            object_gt = {
                    "cam_R_w2m": R.tolist(),
                    "cam_t_w2m": instance.position.tolist(),

                    # Both 3x3 and 4x4 transformations produce same results.
                    # Mapping with 4x4 transformation and homogenours coords.
                    "cam_R_m2c": (kub_2_bop_frame @ obj_in_cam_coords[:3,:3]).tolist(),
                    "cam_t_m2c": (kub_2_bop_frame @ obj_in_cam_coords[:3,3]).tolist(),
                    #"cam_R_m2c": (obj_in_cam_coords[:3,:3]).tolist(),
                    #"cam_t_m2c": (obj_in_cam_coords[:3,3]).tolist(),
                    
                    # Mapping with 3x3 transformation
                    #"cam_R_m2c": (cam_R.T.dot(R)).tolist(), # Rotation is right
                    #"cam_t_m2c": (cam_R.T.dot((-cam_t + instance.position))).tolist(),

                    #"cam_R_m2c": (R.dot(cam_R.T).tolist(),
                    #"cam_t_m2c": (cam_t - instance.position).tolist(),
                    "obj_id": instance.asset_id,
                    #"image_positions": np.array([scene.camera.project_point(point3d=p, frame=f)[:2]
                    #        for f, p in zip(frame_range, info["positions"])], dtype=np.float32),
                    }

        # Scale bounding boxes to image scale if found.
        y_min, x_min, y_max, x_max = info['bboxes'][frame_idx]
        if y_min == -1 and x_min == -1 and y_max == -1 and x_max == -1:
            bb = [-1, -1, -1, -1]
        else:
            bb = [y_min*h, x_min*w, y_max*h, x_max*w]
            bb = list(map(int, bb))
        
        object_gt_info = {
                #"description": info['description'],
                #"bbox_obj": info['bboxes'], # List of bbox
                "bbox_visib": bb,
                #"bbox_obj": None, 
                #"px_count_all": None,
                #"px_count_valid": None,
                #"px_count_visib": None,
                #"visib_fract": None
                }
        gt_frames_data.append(object_gt)
        gt_info_frames_data.append(object_gt_info)

    return gt_frames_data, gt_info_frames_data

# --- Output BOP style
def get_BOP_info(scene, assets_subset=None):
    """
    Scene defined here to be a sequence of frames.
    """
    frame_range = range(scene.frame_start, scene.frame_end+1)

    object_gt = {"cam_R_m2c": "rotation", "cam_t_m2c": "translation", "obj_id": 1}

    scene_gt_info: list[dict] = [None]*len(frame_range)

    # extract the framewise position, quaternion, and velocity for each object
    assets_subset = scene.foreground_assets if assets_subset is None else assets_subset

    from itertools import starmap
    #fun = lambda i, f=frame_range: extract_frame_data(instance=i,frame_range=f)
    def fun(instance, frame_range=frame_range, scene=scene):
        if hasattr(instance, 'asset_id'):
            data = extract_frame_data(instance=instance,frame_range=frame_range, scene=scene)
            return data

    data = list(map(fun, assets_subset))
    n_objects = len(data)

    gt_info = {}; gt = {}
    for frame_idx, frame in enumerate(frame_range):
        #gt[frame] = np.empty(n_objects,dtype=object)
        gt[frame] = [None]*n_objects
        #gt_info[frame] = np.empty(n_objects,dtype=object)
        gt_info[frame] = [None]*n_objects
        for instance_idx, instance_data in enumerate(data):

            # Skip instances with no <'asset_id'>. i.e. static background/distractions objects.
            if instance_data == None:
                continue
            gt_frames_data, gt_info_frames_data = instance_data
            gt[frame][instance_idx] = gt_frames_data[frame_idx]
            gt_info[frame][instance_idx] = gt_info_frames_data[frame_idx]

    return gt, gt_info

from kubric.file_io import (write_rgb_batch,
                    write_rgba_batch,
                    write_depth_batch,
                    write_uv_batch,
                    write_normal_batch,
                    write_flow_batch,
                    write_forward_flow_batch,
                    write_backward_flow_batch,
                    write_segmentation_batch,
                    write_coordinates_batch,
    )
#from kubric.file_io import write_png, as_path, multi_write_image
#def write_depth_batch(data, directory, file_template="depth_{:05d}.png", max_write_threads=16):
#    assert data.ndim == 4 and data.shape[-1] == 1, data.shape
#    if type(data) != np.uint8:
#        breakpoint()
#        info_type = np.iinfo(data.dtype)
#        print("Depth not type uint8, normalizing and coverting. Found type:", info_type)
#        data = data / info_type.max # Normalize to [0,1]
#        data *= 255
#        data = data.astype(np.uint8)
#    path_template = str(as_path(directory) / file_template)
#    multi_write_image(data, path_template, write_fn=write_png, max_write_threads=max_write_threads)


DEFAULT_WRITERS = {
    "rgb": write_rgb_batch,
    "rgba": write_rgba_batch,
    "depth": write_depth_batch,
    "uv": write_uv_batch,
    "normal": write_normal_batch,
    "flow": write_flow_batch,
    "forward_flow": write_forward_flow_batch,
    "backward_flow": write_backward_flow_batch,
    "segmentation": write_segmentation_batch,
    "object_coordinates": write_coordinates_batch,
}

#def write_image_dict_bop(data_dict: Dict[str, np.ndarray], directory: PathLike,
def write_image_dict_bop(data_dict, directory,
    file_templates = (), max_write_threads=16):
    conversions = {
        'rgba' : 'rgba',
        'backward_flow' : 'backward_flow', 
        'forward_flow' : 'forward_flow',
        'depth' : 'depth',
        'normal' : 'normal',
        'object_coordinates' : 'object_coordinates',
        'segmentation' : 'mask_visib'
        }
    for key, data in data_dict.items():
        save_path = Path(directory,key)
        save_path.mkdir(exist_ok=True, parents=True)
        print("Saving:",save_path)
        if key in file_templates:
            DEFAULT_WRITERS[key](data, save_path, file_template=file_templates[key],
            max_write_threads=max_write_threads)
        else:
            DEFAULT_WRITERS[key](data, save_path, max_write_threads=max_write_threads)


def camera_to_dense(cam_K: [3,3]):
    """
    Sparse camera intrinsic matrix to dense dictionary.
    param:
        cam_K: np.array
    """
    data = {
            'fx':cam_K[0,0], 'fy':cam_K[1,1],
            'cx':cam_K[0,2], 'cy':cam_K[0,2]
            }

    return data

