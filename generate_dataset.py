import bpy
import bpy_extras
import os
import random
import math

OUTPUT_IMAGES_DIR = os.path.expanduser("~/yolo_dataset/linear_sense/images/train")
OUTPUT_IMAGES_DIR_VAL = os.path.expanduser("~/yolo_dataset/linear_sense/images/val")
OUTPUT_LABELS_DIR = os.path.expanduser("~/yolo_dataset/linear_sense/labels/train")
OUTPUT_LABELS_DIR_VAL = os.path.expanduser("~/yolo_dataset/linear_sense/labels/val")
BACKGROUNDS_DIR = os.path.expanduser("~/yolo_dataset/backgrounds")
DECALS_DIR = os.path.expanduser("~/yolo_dataset/decals")
DATASET_SIZE = 1500
CLASS_ID = 0
SKIP_PARTIAL_OBJECTS = True
SKIP_OCCLUDED_OBJECTS = True
MIN_VISIBLE_RATIO = 0.55
OCCLUSION_SAMPLES = 120

scene = bpy.context.scene
cam = bpy.data.objects['Camera']
target = bpy.data.objects['TargetObject']
light = bpy.data.objects['Light']
bg_plane = bpy.data.objects['BackgroundPlane']
base_plate = bpy.data.objects.get('BasePlate')

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
        print(f"Activating GPU: {device.name}")

scene.render.resolution_x = 480
scene.render.resolution_y = 640
scene.render.resolution_percentage = 100

cam.data.type = 'PERSP'
cam.data.lens_unit = 'FOV'
cam.data.angle = math.radians(69.0)

def randomize_pose_and_scale(cam_obj, target_obj):
    # 1. Place the camera in a random hemisphere
    radius = random.uniform(0.5, 2.5)
    theta = random.uniform(0, 2 * math.pi)
    phi = math.acos(random.uniform(0.1, 1.0)) 
    
    cam_obj.location = (
        target_obj.location.x + radius * math.sin(phi) * math.cos(theta),
        target_obj.location.y + radius * math.sin(phi) * math.sin(theta),
        target_obj.location.z + radius * math.cos(phi)
    )
    
    # 2. Point camera exactly at the target
    cam_dir = target_obj.location - cam_obj.location
    cam_obj.rotation_euler = cam_dir.to_track_quat('-Z', 'Y').to_euler()

    # 3. Orient the Object to face the camera
    obj_dir = cam_obj.location - target_obj.location
    
    # Force the object's -Y axis (the open end) to point at the camera
    base_rot = obj_dir.to_track_quat('Y', 'Z').to_euler()
    
    # 4. Add random "wobble"
    # Force ~15% of the dataset to be PERFECTLY front-facing 
    if random.random() < 0.15:
        wobble_x = 0.0
        wobble_z = 0.0
    else:
        # Otherwise, apply the normal severe angles
        wobble_x = random.uniform(-1.2, 1.2)
        wobble_z = random.uniform(-1.2, 1.2)
        
    # The pipe can still spin freely 360 degrees around its own central length axis (Y)
    spin_y = random.uniform(-math.pi, math.pi) 
    
    target_obj.rotation_euler = (
        base_rot[0] + wobble_x,
        base_rot[1] + spin_y,
        base_rot[2] + wobble_z
    )

    # 5. Apply Dynamic Length and Diameter
    diameter_scale = random.uniform(0.9, 1.1) 
    length_scale = random.uniform(0.5, 1.5)   
    
    target_obj.scale = (diameter_scale, length_scale, diameter_scale)

def randomize_baseplate(target_obj, decals_dir):
    if not base_plate:
        return
    
    valid_exts = ('.png', '.jpg', '.jpeg')
    decals = [f for f in os.listdir(decals_dir) if f.lower().endswith(valid_exts)]
    
    if decals:
        random_decal_name = random.choice(decals)
        img_path = os.path.join(decals_dir, random_decal_name)
        new_img = bpy.data.images.load(img_path)
        
        mat = base_plate.data.materials[0]
        nodes = mat.node_tree.nodes
        
        for node in nodes:
            if node.type == 'TEX_IMAGE':
                old_img = node.image
                node.image = new_img
                if old_img:
                    bpy.data.images.remove(old_img)
                break

    base_plate.rotation_euler[1] = random.uniform(0, 2 * math.pi)

def randomize_grime(target_obj):
    mat = target_obj.data.materials.get('PVCMaterial')
    if not mat or not mat.use_nodes:
        return
        
    nodes = mat.node_tree.nodes
    
    # Find the Mapping node and scramble its location vector
    for node in nodes:
        if node.type == 'MAPPING':
            # Shift the noise texture coordinates by massive random amounts
            node.inputs['Location'].default_value[0] = random.uniform(-1000, 1000)
            node.inputs['Location'].default_value[1] = random.uniform(-1000, 1000)
            node.inputs['Location'].default_value[2] = random.uniform(-1000, 1000)
            
            # Optionally rotate the noise so scratches go in different directions
            node.inputs['Rotation'].default_value[0] = random.uniform(0, 6.28)
            node.inputs['Rotation'].default_value[1] = random.uniform(0, 6.28)
            node.inputs['Rotation'].default_value[2] = random.uniform(0, 6.28)
            break

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
    
    for node in nodes:
        if node.type == 'TEX_IMAGE':
            old_img = node.image
            node.image = new_img
            if old_img:
                bpy.data.images.remove(old_img)
            break

def spawn_hard_negatives(scene):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in scene.objects:
        if obj.name.startswith("Distractor"):
            obj.select_set(True)
    bpy.ops.object.delete()

    mat = bpy.data.materials.new(name="WhiteGloss")
    mat.use_nodes = True
    if "Principled BSDF" in mat.node_tree.nodes:
        mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1)
        mat.node_tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.2

    for i in range(random.randint(3, 7)):
        shape_type = random.choice(['CUBE', 'PLANE', 'RECTANGLE'])
        if shape_type == 'CUBE':
            bpy.ops.mesh.primitive_cube_add(size=random.uniform(0.05, 2.0))
        elif shape_type == 'RECTANGLE':
            bpy.ops.mesh.primitive_cube_add(size=1)
            obj = bpy.context.active_object
            obj.scale.x = random.uniform(0.05, 3.0)
            obj.scale.y = random.uniform(0.1, 0.5)
            obj.scale.z = random.uniform(0.01, 0.05)
        else:
            bpy.ops.mesh.primitive_plane_add(size=random.uniform(1.0, 4.0))
            
        distractor = bpy.context.active_object
        distractor.name = f"Distractor_{i}"
        
        distractor.location = (
            random.uniform(-4, 4),
            random.uniform(-4, 4),
            random.uniform(-2, 5)
        )
        
        distractor.scale = (
            random.uniform(0.1, 3.0),
            random.uniform(0.1, 3.0),
            random.uniform(0.1, 3.0)
        )
        distractor.rotation_euler = (random.uniform(0, 3.14), random.uniform(0, 3.14), random.uniform(0, 3.14))
        distractor.data.materials.append(mat)

def get_yolo_bbox(scene, cam, obj):
    matrix = obj.matrix_world
    verts_3d = [matrix @ v.co for v in obj.data.vertices]
    verts_2d = [bpy_extras.object_utils.world_to_camera_view(scene, cam, v) for v in verts_3d]

    # If enabled: skip labels when object is partially out of frame (or behind camera)
    if SKIP_PARTIAL_OBJECTS:
        for v in verts_2d:
            if v.z <= 0.0 or v.x < 0.0 or v.x > 1.0 or v.y < 0.0 or v.y > 1.0:
                return None

    x_coords = [v.x for v in verts_2d]
    y_coords = [v.y for v in verts_2d]

    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)

    # Keep clamping for safety when flag is off
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

def _is_target_hit(hit_obj, target_obj):
    if not hit_obj:
        return False
    if hit_obj == target_obj:
        return True
    original = getattr(hit_obj, "original", None)
    return original == target_obj


def has_min_visibility(scene, cam, obj, min_visible_ratio=0.55, samples=120):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()

    if not mesh or len(mesh.vertices) == 0:
        if mesh:
            obj_eval.to_mesh_clear()
        return False

    cam_origin = cam.matrix_world.translation
    total = len(mesh.vertices)
    step = max(1, total // max(1, samples))

    tested = 0
    visible = 0

    for idx in range(0, total, step):
        v_local = mesh.vertices[idx].co
        v_world = obj_eval.matrix_world @ v_local

        # Ignore points not in front of camera or outside image frame.
        v2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, v_world)
        if v2d.z <= 0.0 or v2d.x < 0.0 or v2d.x > 1.0 or v2d.y < 0.0 or v2d.y > 1.0:
            continue

        tested += 1
        ray = v_world - cam_origin
        dist = ray.length
        if dist <= 1e-6:
            continue

        direction = ray.normalized()
        hit, loc, normal, face_idx, hit_obj, matrix = scene.ray_cast(
            depsgraph,
            cam_origin,
            direction,
            distance=dist + 1e-4
        )

        if hit and _is_target_hit(hit_obj, obj):
            visible += 1

    obj_eval.to_mesh_clear()

    if tested == 0:
        return False

    return (visible / tested) >= min_visible_ratio

TRAIN_RATIO = 0.001
train_cutoff = int(DATASET_SIZE * TRAIN_RATIO)

# The Quota System: 10% negative samples is perfect for YOLO
MAX_NEGATIVES = int(DATASET_SIZE * 0.050) 

total_saved = 0
negatives_saved = 0

print(f"Starting generation of {DATASET_SIZE} images (Targeting {MAX_NEGATIVES} negative samples)...")

while total_saved < DATASET_SIZE:
    # 1. Randomize the scene
    randomize_pose_and_scale(cam, target)
    randomize_lighting(light, cam, target)
    randomize_background(bg_plane, BACKGROUNDS_DIR)
    # randomize_baseplate(target, DECALS_DIR)
    # randomize_grime(target)
    spawn_hard_negatives(scene)

    bpy.context.view_layer.update()

    # 2. Evaluate the frame
    yolo_label = get_yolo_bbox(scene, cam, target)
    is_valid_pipe = False

    if yolo_label:
        # Check if it's blocked by distractors
        if SKIP_OCCLUDED_OBJECTS and not has_min_visibility(
            scene, cam, target,
            min_visible_ratio=MIN_VISIBLE_RATIO,
            samples=OCCLUSION_SAMPLES
        ):
            yolo_label = None # Erase label, treat as background
        else:
            is_valid_pipe = True # It passed all tests!

    # 3. The Gatekeeper: Decide if we keep this frame
    if not is_valid_pipe:
        if negatives_saved < MAX_NEGATIVES:
            negatives_saved += 1
        else:
            # We already have enough empty images. Skip and reroll!
            continue

    # 4. Determine file paths
    filename = f"linear_board_{total_saved + 6549:05d}" 

    if total_saved < train_cutoff:
        image_dir = OUTPUT_IMAGES_DIR
        label_dir = OUTPUT_LABELS_DIR
        split_name = "train"
    else:
        image_dir = OUTPUT_IMAGES_DIR_VAL
        label_dir = OUTPUT_LABELS_DIR_VAL
        split_name = "val"

    # 5. Render and save the image
    scene.render.filepath = os.path.join(image_dir, f"{filename}.jpg")
    scene.render.image_settings.file_format = 'JPEG'
    bpy.ops.render.render(write_still=True)

    # 6. Write the label file
    with open(os.path.join(label_dir, f"{filename}.txt"), "w") as f:
        if yolo_label:
            f.write(yolo_label)

    # 7. Print progress and increment master counter
    if total_saved % 10 == 0:
        print(f"Rendered {total_saved}/{DATASET_SIZE} | Negatives: {negatives_saved}/{MAX_NEGATIVES}")

    total_saved += 1

print(f"Dataset generation complete! Generated {DATASET_SIZE - MAX_NEGATIVES} labeled images and {MAX_NEGATIVES} negative samples.")