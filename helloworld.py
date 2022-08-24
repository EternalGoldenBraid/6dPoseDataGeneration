from pathlib import Path
import logging
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer

logging.basicConfig(level="INFO")

# --- create scene and attach a renderer to it
scene = kb.Scene(resolution=(256, 256))
renderer = KubricRenderer(scene)

# --- populate the scene with objects, lights, cameras
object_name = "gear_assembled"
scene += kb.Cube(name="floor", scale=(100, 100, 1), position=(0, 0, -23))
#scene += kb.Sphere(name="ball", scale=1, position=(0, 0, 1.))
new_obj = kb.FileBasedObject(
        asset_id=object_name,
        #render_filename="cad_models/gear_assembled.obj",
        render_filename=Path("cad_models",object_name,".glb"),
        bounds=((-1, -1, -1), (1, 1, 1)),
        simulation_filename=None
        )
scene += new_obj
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3),
                             look_at=(0, 0, 0), intensity=1.5)
scene += kb.PerspectiveCamera(name="camera", position=(3, -1, 90),
                              look_at=(0, 0, 1))


n_frames = 10
for frame_idx in range(n_frames):

    # Rotate camera
    scene.camera.position = kb.sample_point_in_half_sphere_shell(9., 10.)
    scene.camera.look_at((0, 0, 1))

    # --- render (and save the blender file)
    renderer.save_state(f"output/helloworld_{frame_idx}.blend")
    frame = renderer.render_still()
    
    # --- save the output as pngs
    kb.write_png(frame["rgba"], "output/helloworld_{frame_idx}.png")
    kb.write_palette_png(frame["segmentation"], "output/helloworld_{frame_idx}_segmentation.png")
    scale = kb.write_scaled_png(frame["depth"], "output/helloworld_{frame_idx}_depth.png")
    logging.info("Depth scale: %s", scale)
