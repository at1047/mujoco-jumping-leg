import mujoco

def get_link_lengths(model, debug = False):
    """Get the lengths of all links in the model"""
    link_lengths = {}
    
    if debug:
        print(f"Number of geoms: {model.ngeom}")
        for i in range(model.ngeom):
            geom_name = model.geom(i).name if model.geom(i).name else f"geom_{i}"
            geom_size = model.geom(i).size
            geom_type = model.geom(i).type
            print(f"Geom {i} ({geom_name}): type = {geom_type}, size = {geom_size}")


    for i in range(model.ngeom):
        geom = model.geom(i)
        geom_name = geom.name if geom.name else f"geom_{i}"
        geom_type = geom.type
        geom_size = geom.size
        
        # For cylinders, the length is typically 2 * size[1] (half-length)
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            length = 2 * geom_size[1]  # Full length of cylinder
            link_lengths[geom_name] = {
                'type': 'cylinder',
                'length': length,
                'radius': geom_size[0]
            }
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            length = 2 * geom_size[1]  # Full length of capsule
            link_lengths[geom_name] = {
                'type': 'capsule', 
                'length': length,
                'radius': geom_size[0]
            }
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            # For boxes, return dimensions
            link_lengths[geom_name] = {
                'type': 'box',
                'dimensions': geom_size
            }

    if debug:
        print(link_lengths)

    return link_lengths