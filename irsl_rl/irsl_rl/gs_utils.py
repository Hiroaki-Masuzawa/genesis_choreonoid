import genesis as gs
from genesis.sensors import RigidContactForceSensor

## functions
import numpy as np
from cnoid.IRSLCoords import coordinates
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import torch

## coordinates like functions
def transformation(coords_0, coords_1):
    """
    coords := ( pos, quat )
    coords_0.transformation(coords_1)
    """
    i0 = inverse_transformation(coords_0)
    return transform(i0, coords_1)

def inverse_transformation(coords):
    """
    coords := ( pos, quat )
    coords.inverse_transformation()
    """
    p, q = coords
    iq = inv_quat(q)
    return (-transform_by_quat(p, iq), iq)

def transform_vector(coords, vec):
    """
    coords := ( pos, quat )
    coords.transform_vector(vec)
    """
    p, q = coords
    return transform_by_quat(vec, q) + p

def transform(coords_0, coords_1):
    """
    coords_0 @ coords_1
    coords := ( pos, quat )

    coordinates(coords_0).transform(coords_1)
    """
    p0, q0 = coords_0
    p1, q1 = coords_1
    rq = transform_quat_by_quat(q1, q0)
    rp = transform_by_quat(p1, q0) + p0
    return (rp, rq)

def translate(coords, vec):
    """
    coords := ( pos, quat )
    coords.translate(vec)
    """
    p, q = coords
    return p + vec, q

def rotate(coords, rad, axis):
    """
    """
    pass

def irsl_coords(coords):
    """
    coords := ( pos, quat )
    """
    p, q = coords
    res = coordinates(p.cpu())
    res.quaternion_wxyz = q.cpu()
    return res

def pos_quat_from_irsl(irsl_coords):
    """
    Reterns:
        coords := ( pos, quat )
    """
    return (torch.tensor(irsl_coords.pos, device=gs.device, dtype=gs.tc_float),
            torch.tensor(irsl_coords.quaternion_wxyz, device=gs.device, dtype=gs.tc_float))

