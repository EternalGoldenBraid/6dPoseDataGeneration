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

def create_urdf(object_mesh_path, collision_mesh_path, object_name = 'gear_assembled'):

    assets_dir = Path('assets',object_name)
    urdf_path = assets_dir/(object_name+".urdf")
    data_json_path = assets_dir / "data.json"

    scene = tm.load(object_mesh_path)
    geometries = scene.geometry.values()
    geometry = list(geometries)[0]
    properties = get_object_properties(tmesh=geometry)

    inertia = np.eye(3)/1e3
    center_mass = np.zeros(3)
    mass = 0.01 # KG

    # URDF
    urdf: str = URDF_TEMPLATE.format( id=object_name, **properties)
    with open(urdf_path, "w") as f:
        f.write(urdf)

    print(urdf)


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

    print(asset_entry)

    with open(data_json_path, "w") as f:
        json.dump(asset_entry, f, indent=4, sort_keys=True)

    # Simulator manifest file
    manifest_path = "test_manifest.json"
    manifest = {
        "name": "test_manifest",
        #"data_dir": ".",
        "version": "1.0",
        #"assets": {k: v for k, v in assets_list}
        "assets": {object_name: asset_entry}
    }

    #import pdb; pdb.set_trace()

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4, sort_keys=True)

if __name__ == '__main__':
    object_name = "gear_assembled"
    create_urdf(object_mesh_path=Path('assets',object_name,object_name+".glb"), collision_mesh_path=None)

