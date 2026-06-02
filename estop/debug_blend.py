import bpy

print("\n" + "="*50)
print("--- BLENDER SCENE DIAGNOSTICS ---")

# 1. Check Target Object
target = bpy.data.objects.get('TargetObject')
if not target:
    print("ERROR: TargetObject not found!")
else:
    print(f"TargetObject Location: {target.location}")
    print(f"TargetObject Scale: {target.scale}")
    
    verts = []
    def collect(current):
        if current.type == 'MESH' and current.data is not None:
            mat = current.matrix_world
            for v in current.data.vertices:
                verts.append(mat @ v.co)
        for child in current.children:
            collect(child)
    collect(target)
    
    if verts:
        x = [v.x for v in verts]
        y = [v.y for v in verts]
        z = [v.z for v in verts]
        
        dim_x = max(x) - min(x)
        dim_y = max(y) - min(y)
        dim_z = max(z) - min(z)
        center_x = (max(x) + min(x)) / 2
        center_y = (max(y) + min(y)) / 2
        center_z = (max(z) + min(z)) / 2
        
        print(f"TargetObject True Dimensions: X={dim_x:.4f}m, Y={dim_y:.4f}m, Z={dim_z:.4f}m")
        print(f"TargetObject True Center: X={center_x:.4f}, Y={center_y:.4f}, Z={center_z:.4f}")
    else:
        print("TargetObject has no mesh geometry!")

# 2. Check Background Plane
bg = bpy.data.objects.get('BackgroundPlane')
if not bg:
    print("ERROR: BackgroundPlane not found!")
else:
    print(f"BackgroundPlane Dimensions: {bg.dimensions}")
    print(f"BackgroundPlane Scale: {bg.scale}")
    print(f"BackgroundPlane Location: {bg.location}")

print("="*50 + "\n")
