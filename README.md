Pull from dockerhub
```
docker pull kubricdockerhub/kubruntu
```
Any Kubric worker file can be run in docker as:

```
docker run --rm --interactive \
    --user $(id -u):$(id -g) \
    --volume "$PWD:/kubric" \
    kubricdockerhub/kubruntu \
    python3 examples/<worker_file>.py
```

# Generate necessary files to run a simulation
Run `add_objects.py` to create `<manifest_name>.json` file 
that kubric utilizes to configure cad models for simulation.
``` 
python add_object.py --scene scene_name --target <target_manifest>
 ``` 


It also generates the `*.urdf` files to `assets/<object_in_scene>/<object_in_scene>.urdf` 
for each object in the scene.

After generating the simulation files, execute the worker file `<worker_file>.py` (in docker) to generate the contents of the `output` directory.

Before executing the worker file, make sure the line
```  
obj_source = kb.AssetSource.from_manifest(manifest_path="<target_manifest>.json")
 ```
contains the same `<target_manifest>` used with `add_object.py`. **TODO:** Simplify this.

An example run might be:

``` 
python add_object.py --scene rogo_gears --target robo_gears_manifest 
``` 

``` 
docker run --rm --interactive \
    --user $(id -u):$(id -g) \
    --volume "$PWD:/kubric" \
    kubricdockerhub/kubruntu \
    python3 simulate.py 
```
or for convenience I use the bash script
```
simulate.sh
 ```
 which clears the previos outputs and generates a gif.



This generates the following file structure where files/dirs marked marked with `*` are generated.
You only need to place the scene defined in the `*.glb` file into the assets directory.

```
├── add_object.py
├── assets
│   └── robo_gears
│       ├── robo_gears.glb
│       ├── * Bottom_casing
│       │   ├── * Bottom_casing.obj
│       │   ├── * Bottom_casing.urdf
│       │   └── * data.json
│       ├── * Helical_gear_toy_assembled
│       ├── * Left_gear
│       ├── * Right_gear
│       ├── * Top_casing
├── output
│   ├── simulation
│   │   ├── * data_ranges.json
│   │   └── * simulator.blend
│   └── * simulator.gif
├── * robo_gears.json
├── simulate.py
```

# TODO
- [] Add argparser to worker file
- [] Figure out how to initialize object positions. Currently crashes once certain number of objects added.
- [] Manifest files in their own dirs.

Based on: https://github.com/google-research/kubric