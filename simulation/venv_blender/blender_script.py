import bpy
import numpy as np
import yaml
import sys
import os
import platform

argv = sys.argv
argv = argv[argv.index("--") + 1:]
EXPERIMENT_DIR = argv[0].strip()

print("Experiment directory:")
print(EXPERIMENT_DIR)
print()
print()

with open(os.path.join(EXPERIMENT_DIR, 'config.yaml')) as f:
    config = yaml.safe_load(f)

FRAME_DIM = config['FRAME_DIM']
ENV_DIM = config['ENV_DIM']
ENV_EPS = config['ENV_EPS']

print("Frame dimension:")
print(FRAME_DIM)
print()
print("Environment dimension:")
print(ENV_DIM)
print("Environment epsilon:")
print(ENV_EPS)
print()

pos = np.load(os.path.join(EXPERIMENT_DIR, 'riab_simulation/positions.npy'))
thetas = np.load(os.path.join(EXPERIMENT_DIR, 'riab_simulation/thetas.npy'))

def srgb_hex_to_linear(hex_color):
    """'#rrggbb' (sRGB) -> linear RGBA, as Blender expects for Base Color."""
    rgb = [int(hex_color.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb] + [1.0]

for i, obstacle in enumerate(config.get('OBSTACLES') or []):
    cx, cy = obstacle['CENTER']
    # SIZE is an edge length (square) or [width, depth].
    size, height = obstacle['SIZE'], obstacle['HEIGHT']
    width, depth = size if isinstance(size, list) else (size, size)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, height / 2))
    pillar = bpy.context.active_object
    pillar.name = f"obstacle{i}"
    pillar.scale = (width, depth, height)

    material = bpy.data.materials.new(name=f"obstacle{i}")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = srgb_hex_to_linear(obstacle['COLOR'])
    bsdf.inputs["Roughness"].default_value = 0.8
    pillar.data.materials.append(material)
    print(f"Added {pillar.name}: centre {obstacle['CENTER']}, size {size}, height {height}")

camera_center =  bpy.data.objects['Camera_center']

frame_count = 1
max_frame = len(pos)

for xy, theta in zip(pos, thetas):
    if frame_count%10_000==0:
        print(f"{frame_count/max_frame*100:.2f}% there.")
        
    x, y = xy[0], xy[1]
    camera_center.location = (x, y, 0.035)
    camera_center.keyframe_insert(data_path="location", frame=frame_count)
    
    camera_center.rotation_euler = (np.pi/2, 0, theta)
    camera_center.keyframe_insert(data_path="rotation_euler", frame=frame_count)
    
    frame_count += 1
    
print(f'number of frames: {frame_count}')

###
### RENDERING PARAMETERS
###

scene = bpy.context.scene

scene.render.engine = "CYCLES"

# Set the device_type
if 'mac' in platform.platform():
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "METAL"
else:
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # "CUDA" or "OPTIX"
    bpy.context.scene.cycles.denoiser = "OPTIX"

bpy.context.scene.cycles.device = "GPU"
bpy.context.preferences.addons["cycles"].preferences.refresh_devices()

# Resolution change
scene.render.resolution_x = FRAME_DIM[0]
scene.render.resolution_y = FRAME_DIM[1]
