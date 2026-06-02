import bpy
import bpy_extras
import os
import random
import math

OUTPUT_IMAGES_DIR = os.path.expanduser("~/pose_detection/estop/images/train")
OUTPUT_IMAGES_DIR_VAL = os.path.expanduser("~/pose_detection/estop/images/val")
OUTPUT_LABELS_DIR = os.path.expanduser("~/pose_detection/estop/labels/train")
OUTPUT_LABELS_DIR_VAL = os.path.expanduser("~/pose_detection/estop/labels/val")
BACKGROUNDS_DIR = os.path.expanduser("~/pose_detection/backgrounds")
DATASET_SIZE = 1500
CLASS_ID = 0

SKIP_PARTIAL_OBJECTS = True
SKIP_OCCLUDED_OBJECTS = True
MIN_VISIBLE_RATIO = 0.55
OCCLUSION_SAMPLES = 120

scene = bpy.context.scene
cam = bpy.data.objects['Camera']
# TODO: Update this to match your E-Stop object name in Blender
target = bpy.data.objects['TargetObject'] 
light = bpy.data.objects['Light']
bg_plane = bpy.data.objects['BackgroundPlane']
# Keypoint empty objects should be prefixed with KP_ (e.g. KP_01_Top, KP_02_Base, etc.)
# Sorting ensures the order of keypoints is consistent in every label txt!
keypoints = sorted([obj for obj in bpy.data.objects if obj.name.startswith("KP_")], key=lambda k: k.name)

print(f"Found {len(keypoints)} keypoints: {[k.name for k in keypoints]}")

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
    radius = random.uniform(0.5, 1.5)
    theta = random.uniform(0, 2 * math.pi)
    phi = math.acos(random.uniform(0.1, 1.0)) 
    
    cam_obj.location = (
        target_obj.location.x + radius * math.sin(phi) * math.cos(theta),
        target_obj.location.y + radius * math.sin(phi) * math.sin(theta),
        target_obj.location.z + radius * math.cos(phi)
    )
    
    cam_dir = target_obj.location - cam_obj.location
    cam_obj.rotation_euler = cam_dir.to_track_quat('-Z', 'Y').to_euler()

    # E-stop specific rotation (restrict backside):
    # Since the backside is +X, we rotate Y by 90 degrees to point the backside DOWN (-Z) 
    # to rest flat on the Background Plane. The camera always spawns in the upper 
    # hemisphere (+Z), so it will never see the back!
    spin_on_surface = random.uniform(-math.pi, math.pi)
    wobble = random.uniform(-0.15, 0.15) # Tiny mounting imperfection
    
    target_obj.rotation_euler = (
        wobble, 
        (math.pi / 2) + wobble, 
        spin_on_surface
    )

    # CRITICAL FIX: The FBX was imported at 0.001 scale! 
    # If we set it to 1.0, it becomes 1000x larger and swallows the camera.
    # We must explicitly scale it relative to its imported base scale.
    base_scale = 0.001
    scale_f = random.uniform(0.8, 1.2) * base_scale
    target_obj.scale = (scale_f, scale_f, scale_f)

def randomize_lighting(light_obj, cam_obj, target_obj):
    if random.random() < 0.2:
        light_obj.data.energy = random.uniform(5000, 15000)
        dir_from_cam = target_obj.location - cam_obj.location
        light_obj.location = target_obj.location + (dir_from_cam.normalized() * random.uniform(1.0, 3.0))
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
    
    # Scale background texture appropriately
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
        # Scale the image texture up so we don't zoom in too much on massive Background planes
        nodes['Mapping'].inputs['Scale'].default_value[0] = 50.0
        nodes['Mapping'].inputs['Scale'].default_value[1] = 50.0
        
    for node in nodes:
        if node.type == 'TEX_IMAGE':
            old_img = node.image
            node.image = new_img
            if old_img:
                bpy.data.images.remove(old_img)
            break

def raycast_visibility(scene, depsgraph, cam_origin, target_point, min_dist_tolerance=0.01):
    """Checks if a 3D point is visible to the camera without being occluded by mesh geometry."""
    ray = target_point - cam_origin
    dist = ray.length
    direction = ray.normalized()
    
    # Push the ray a bit past the object to ensure we hit it if it's there
    hit, loc, normal, face_idx, hit_obj, matrix = scene.ray_cast(
        depsgraph,
        cam_origin,
        direction,
        distance=dist + 0.1
    )
    
    if hit:
        # If it hit geometry, see if the hit is physically close to our keypoint empty
        if (loc - target_point).length < min_dist_tolerance:
            return 2 # Visible! The hit was the target itself
        else:
            return 1 # Occluded! Hit some other face first
    # If raycast didn't hit anything, standardly it means it's visible (nothing in front).
    return 2


def get_yolo_pose(scene, cam, obj, kps):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    # 1. Bounding Box logic
    verts_3d = []
    
    def collect_vertices(current_obj):
        if current_obj.type == 'MESH' and current_obj.data is not None:
            mat = current_obj.matrix_world
            for v in current_obj.data.vertices:
                verts_3d.append(mat @ v.co)
        for child in current_obj.children:
            collect_vertices(child)
            
    collect_vertices(obj)
    
    if not verts_3d:
        print("Warning: No vertices found for target object!")
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
    
    bbox_str = f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    # 2. Keypoints logic
    kp_strs = []
    cam_origin = cam.matrix_world.translation
    
    for kp in kps:
        kp_world = kp.matrix_world.translation
        kp_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, kp_world)
        
        # Check if empty is outside frame or behind camera
        if kp_2d.z <= 0.0 or kp_2d.x < 0.0 or kp_2d.x > 1.0 or kp_2d.y < 0.0 or kp_2d.y > 1.0:
            vis = 0 # Out of frame
            px, py = 0.0, 0.0
        else:
            px = kp_2d.x
            py = 1.0 - kp_2d.y
            vis = raycast_visibility(scene, depsgraph, cam_origin, kp_world, min_dist_tolerance=0.05)
            
        kp_strs.append(f"{px:.6f} {py:.6f} {vis}")

    return bbox_str + " " + " ".join(kp_strs)


TRAIN_RATIO = 0.8 # Typical 80-20 train-val split
train_cutoff = int(DATASET_SIZE * TRAIN_RATIO)
MAX_NEGATIVES = int(DATASET_SIZE * 0.050) 
total_saved = 0
negatives_saved = 0

print(f"Starting generation of {DATASET_SIZE} images (Targeting {MAX_NEGATIVES} negative samples)...")

# Make dirs
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_DIR_VAL, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR_VAL, exist_ok=True)

while total_saved < DATASET_SIZE:
    randomize_pose_and_scale(cam, target)
    randomize_lighting(light, cam, target)
    randomize_background(bg_plane, BACKGROUNDS_DIR)
    
    bpy.context.view_layer.update()

    yolo_label = get_yolo_pose(scene, cam, target, keypoints)
    is_valid_estop = False

    if yolo_label:
        is_valid_estop = True

    if not is_valid_estop:
        if negatives_saved < MAX_NEGATIVES:
            negatives_saved += 1
        else:
            continue

    filename = f"estop_{total_saved:05d}" 

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

print(f"Dataset complete. Generated {DATASET_SIZE - MAX_NEGATIVES} labeled images and {MAX_NEGATIVES} bg images.")