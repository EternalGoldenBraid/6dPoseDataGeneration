"""
Place a *.glb or *glft file defining a scene that contains multiple objects.
Currently tested by importing multiple stl files in to blender and exporting the scene
as <name>.glb.
"""
# TODO write docs

import os
from pathlib import Path
import json

import numpy as np
import trimesh as tm

URDF_TEMPLATE = """
<robot name="{id}">
    <link name="base">
        <inertial>
            <origin xyz="{center_mass[0]} {center_mass[1]} {center_mass[2]}" />
            <mass value="{mass}" />
            <inertia ixx="{inertia[0][0]}" ixy="{inertia[0][1]}" 
                     ixz="{inertia[0][2]}" iyy="{inertia[1][1]}" 
                     iyz="{inertia[1][2]}" izz="{inertia[2][2]}" />
        </inertial>
        <visual>
            <origin xyz="0 0 0" />
            <geometry>
                <mesh filename="assets/gear_assembled/gear_assembled.obj" />
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" />
            <geometry>
                <mesh filename="assets/gear_assembled/gear_assembled.obj" />
            </geometry>
        </collision>
    </link>
</robot>
"""

def get_object_properties(tmesh, density=10.0):

  properties = {
      "nr_vertices": len(tmesh.vertices),
      "nr_faces": len(tmesh.faces),
      "bounds": tmesh.bounds.tolist(),
      "surface_area": tmesh.area,
      "volume": tmesh.volume,
      "center_mass": tmesh.center_mass.tolist(),
      "inertia": tmesh.moment_inertia.tolist(),
      "mass": tmesh.volume * density,
  }

  return properties

def create_objects_from_gltf(scene_dir):
    """
    Expects a <scene_name>.glb or <scene_name>.glft file as
    <scene_name>/<scene_name>.glb
    """

    print("Generating manifest for scene:", scene_dir.name)

    if scene_dir.name+".glb" in os.listdir(scene_dir):
        file_type = ".glb"
    elif scene_dir.name+".gltf" in os.listdir(scene_dir):
        file_type = ".gltf"
    else:
        print(scene_dir.name, "not found in", os.listdir(scene_dir))
        return None

#scene = tm.load(scene_dir/scene_name/".gltf")
    scene = tm.load(scene_dir/(scene_dir.name+file_type))
    object_names = list(scene.geometry)
    geometries = list(scene.geometry.values())
    assets_list = []
    for object_name, geometry in zip(object_names, geometries):
        assert type(geometry) == tm.Trimesh

        object_mesh_path= scene_dir / object_name / (object_name+".obj")
        urdf_path = scene_dir/object_name/(object_name+".urdf")
        data_json_path = scene_dir/object_name/"data.json"

        properties = get_object_properties(tmesh=geometry)
        
        # Save .obj file for urdf simulation file.
        obj_output = tm.exchange.obj.export_obj(geometry) # TODO Docs say todo scenes with textured meshes.
        object_mesh_path.parent.mkdir(exist_ok=True, parents=True)
        object_mesh_path.write_text(obj_output)
        assert type(object_mesh_path.read_text()) == str

        # URDF
        urdf: str = URDF_TEMPLATE.format( id=object_name, **properties)
        urdf_path.parent.mkdir(exist_ok=True, parents=True)
        urdf_path.write_text(urdf)
        #with open(urdf_path, "w") as f:
        #    f.write(urdf)

        assert type(urdf_path.read_text()) == str
        print("Created urdf for",object_name)

        # Kubric JSON for object
        # TODO loop over objects

        category = "test_category"
        description = "test_description"
        asset_entry = {
            "path": None,
            "id": object_name,
            "asset_type": "FileBasedObject",
            "kwargs": {
                "bounds": properties["bounds"],
                "mass": properties["mass"],
                #"render_filename": object_mesh_path.name,
                #"simulation_filename": urdf_path.name,
                "render_filename": str(object_mesh_path),
                "simulation_filename": str(urdf_path),
            },
            "license": "CC BY-SA 4.0",
            "metadata": {
                "nr_faces": properties["nr_faces"],
                "nr_vertices": properties["nr_vertices"],
                "surface_area": properties["surface_area"],
                "volume": properties["volume"],
                "category": category,
                "description": description,
            },
        }

        with open(data_json_path, "w") as f:
            json.dump(asset_entry, f, indent=4, sort_keys=True)

        assets_list.append((object_name, asset_entry))
    return assets_list

#def create_urdf(object_mesh_path, collision_mesh_path,
def create_kubric_manifest(scene_dir, manifest_path):
    """ 
    Create urdf file given a *.glb or *.glft defining a scene
    with one or more objects.
    """

    # Create assets from scene defined in glb/gltf format.
    # Easy to store keyframs here (?)
    assets_list = create_objects_from_gltf(scene_dir=scene_dir)
    if not assets_list:
        print("Creation failed")
        return

    # Simulator manifest file
    manifest = {
        #"name": "test_manifest",
        "name": scene_dir.name,
        #"data_dir": ".",
        "version": "1.0",
        "assets": {k: v for k, v in assets_list}
        #"assets": {object_name: asset_entry}
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4, sort_keys=True)

if __name__ == '__main__':
    #object_name = "gear_assembled"
    #object_names = ["left_gear", "right_gear", "top_casing", "bottom_casing", "gear_assembled"]
    #create_urdf(
    #        object_mesh_path=Path('assets',object_name,object_name+".glb"),
    #        collision_mesh_path=None,
    #        object_names = object_names,
    #        )

    import argparse
    
    parser = argparse.ArgumentParser(prog='demo',
            description='Generate simulation files and <manifest>.json for kubric to run simulation.')

    parser.add_argument('-s','--scene', dest='scene_name', default='robo_gears',
                        type=str, help='Scene name defined by assets/<scene_name>/<scene_name>.glb')
    parser.add_argument('-t','--target', dest='target_name', default='robo_gears_manifest',
                        type=str, help='Target file_name created into current directory. TODO Add any path support.')

    args = parser.parse_args()
    #create_kubric_manifest(scene_dir=Path("assets","robo_gears"), manifest_path="robo_gears.json")
    create_kubric_manifest(scene_dir=Path("assets",args.scene_name), manifest_path=args.target_name+".json")
