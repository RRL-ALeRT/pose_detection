import bpy
import bpy_extras
import os
import random
import math
import mathutils

OUTPUT_IMAGES_DIR = os.path.expanduser("~/pose_detection/estop_detect/images/train")
OUTPUT_IMAGES_DIR_VAL = os.path.expanduser("~/pose_detection/estop_detect/images/val")
OUTPUT_LABELS_DIR = os.path.expanduser("~/pose_detection/estop_detect/labels/train")
OUTPUT_LABELS_DIR_VAL = os.path.expanduser("~/pose_detection/estop_detect/labels/val")
BACKGROUNDS_DIR = os.path.expanduser("~/pose_detection/backgrounds")

DATASET_SIZE = 5000  # Bumped to 5,000 for standard Object Detection
CLASS_ID = 0
SKIP_PARTIAL_OBJECTS = True

scene = bpy.context.scene
cam = bpy.data.objects['Camera']
target = bpy.data.objects['TargetObject']
light = bpy.data.objects['Light']
bg_plane = bpy.data.objects['BackgroundPlane']

scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
scene.cycles.samples = 64
scene.cycles.use_denoising = True 

prefs = bpy.context.preferences
cycles_prefs = prefs.addons['cycles'].preferences
cycles_prefs.compute_device_type = 'OPTIX' 
cycles_prefs.get_devices()

for device in cycles_prefs.devices:
    if device.type == 'CPU':
        device.use = False
    else:
        device.use = True

scene.render.resolution_x = 480
scene.render.resolution_y = 640
scene.render.resolution_percentage = 100

cam.data.type = 'PERSP'
cam.data.lens_unit = 'FOV'
cam.data.angle = math.radians(69.0)

def randomize_pose_and_scale(cam_obj, target_obj):
    radius = random.uniform(0.4, 1.5)
    theta = random.uniform(0, 2 * math.pi)
    phi = math.acos(random.uniform(0.1, 1.0)) 
    
    # --- OFF-CENTER FIX ---
    # We aim the camera at a random point in space near the E-Stop, 
    # rather than perfectly dead-center on it. 
    # This teaches YOLO to find it anywhere in the frame!
    offset_limit = 0.15  # Adjust this up to push it further to the edges
    target_center = mathutils.Vector((
        target_obj.location.x + random.uniform(-offset_limit, offset_limit),
        target_obj.location.y + random.uniform(-offset_limit, offset_limit),
        target_obj.location.z + random.uniform(-offset_limit, offset_limit)
    ))
    
    cam_obj.location = (
        target_center.x + radius * math.sin(phi) * math.cos(theta),
        target_center.y + radius * math.sin(phi) * math.sin(theta),
        target_center.z + radius * math.cos(phi)
    )
    
    cam_dir = target_center - cam_obj.location
    cam_quat = cam_dir.to_track_quat('-Z', 'Y')

    # Camera Roll to simulate odd mounting angles
    roll_euler = mathutils.Euler((0.0, 0.0, random.uniform(-math.pi, math.pi)), 'XYZ')
    cam_quat.rotate(roll_euler)
    cam_obj.rotation_euler = cam_quat.to_euler()

    # E-stop specific rotation (restrict backside to floor):
    spin_on_surface = random.uniform(-math.pi, math.pi)
    wobble = random.uniform(-0.15, 0.15) 
    
    target_obj.rotation_euler = (
        wobble, 
        (math.pi / 2) + wobble, 
        spin_on_surface
    )

    # Scale fix for FBX imports
    base_scale = 0.001
    scale_f = random.uniform(0.8, 1.2) * base_scale
    target_obj.scale = (scale_f, scale_f, scale_f)
    
    return target_center

def randomize_lighting(light_obj, cam_obj, target_center):
    if random.random() < 0.2:
        light_obj.data.energy = random.uniform(5000, 15000)
        dir_from_cam = target_center - cam_obj.location
        light_obj.location = target_center + (dir_from_cam.normalized() * random.uniform(1.0, 3.0))
    else:
        light_obj.data.energy = random.uniform(300, 1500)
        light_obj.location = (
            random.uniform(-3, 3),
            random.uniform(-3, 3),
            random.uniform(2, 6)
        )
    
def randomize_background(bg_obj, bg_dir):
    valid_exts = ('.png', '.jpg', '.jpeg')
    images = [f for f in os.listdir(bg_dir) if f.lower().endswith(valid_exts)]
    
    if not images:
        return
        
    random_img_name = random.choice(images)
    img_path = os.path.join(bg_dir, random_img_name)
    new_img = bpy.data.images.load(img_path)
    
    mat = bg_obj.data.materials[0]
    nodes = mat.node_tree.nodes
    
    if 'Mapping' not in nodes:
        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.name = 'Mapping'
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        mat.node_tree.links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
        for node in nodes:
            if node.type == 'TEX_IMAGE':
                mat.node_tree.links.new(mapping.outputs['Vector'], node.inputs['Vector'])
                break
    
    if 'Mapping' in nodes:
        nodes['Mapping'].inputs['Scale'].default_value[0] = 50.0
        nodes['Mapping'].inputs['Scale'].default_value[1] = 50.0

    for node in nodes:
        if node.type == 'TEX_IMAGE':
            old_img = node.image
            node.image = new_img
            if old_img:
                bpy.data.images.remove(old_img)
            break

def get_yolo_bbox(scene, cam, obj):
    verts_3d = []
    
    # Recursively gather vertices so FBX files work properly
    def collect_vertices(current_obj):
        if current_obj.type == 'MESH' and current_obj.data is not None:
            mat = current_obj.matrix_world
            for v in current_obj.data.vertices:
                verts_3d.append(mat @ v.co)
        for child in current_obj.children:
            collect_vertices(child)
            
    collect_vertices(obj)
    
    if not verts_3d:
        return None

    verts_2d = [bpy_extras.object_utils.world_to_camera_view(scene, cam, v) for v in verts_3d]

    if SKIP_PARTIAL_OBJECTS:
        for v in verts_2d:
            if v.z <= 0.0 or v.x < 0.0 or v.x > 1.0 or v.y < 0.0 or v.y > 1.0:
                return None

    x_coords = [v.x for v in verts_2d]
    y_coords = [v.y for v in verts_2d]

    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)

    xmin = max(0.0, min(1.0, xmin))
    xmax = max(0.0, min(1.0, xmax))
    ymin = max(0.0, min(1.0, ymin))
    ymax = max(0.0, min(1.0, ymax))

    if xmin == xmax or ymin == ymax:
        return None

    width = xmax - xmin
    height = ymax - ymin
    x_center = xmin + (width / 2.0)
    y_center = 1.0 - (ymin + (height / 2.0))

    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

TRAIN_RATIO = 0.8
train_cutoff = int(DATASET_SIZE * TRAIN_RATIO)
# 5% Negative Backgrounds 
MAX_NEGATIVES = int(DATASET_SIZE * 0.050) 
total_saved = 0
negatives_saved = 0

print(f"Starting bounding-box generation of {DATASET_SIZE} images...")

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_DIR_VAL, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR_VAL, exist_ok=True)

while total_saved < DATASET_SIZE:
    target_center = randomize_pose_and_scale(cam, target)
    randomize_lighting(light, cam, target_center)
    randomize_background(bg_plane, BACKGROUNDS_DIR)
    
    bpy.context.view_layer.update()

    # Strictly standard bounding box!
    yolo_label = get_yolo_bbox(scene, cam, target)

    if not yolo_label:
        if negatives_saved < MAX_NEGATIVES:
            negatives_saved += 1
        else:
            continue

    filename = f"estop_bbox_{total_saved:05d}" 

    if total_saved < train_cutoff:
        image_dir = OUTPUT_IMAGES_DIR
        label_dir = OUTPUT_LABELS_DIR
    else:
        image_dir = OUTPUT_IMAGES_DIR_VAL
        label_dir = OUTPUT_LABELS_DIR_VAL

    scene.render.filepath = os.path.join(image_dir, f"{filename}.jpg")
    scene.render.image_settings.file_format = 'JPEG'
    bpy.ops.render.render(write_still=True)

    with open(os.path.join(label_dir, f"{filename}.txt"), "w") as f:
        if yolo_label:
            f.write(yolo_label)

    if total_saved % 10 == 0:
        print(f"Rendered {total_saved}/{DATASET_SIZE} | Negatives: {negatives_saved}/{MAX_NEGATIVES}")

    total_saved += 1

print("Dataset complete.")