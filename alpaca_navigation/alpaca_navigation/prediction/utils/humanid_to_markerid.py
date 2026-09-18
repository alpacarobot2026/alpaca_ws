from enum import Enum

from hst_infer.node_config import *

import numpy as np

# RViz Visualization (code errors)
class marker_id_offset(Enum):
    SKELETON_NS = "skeleton"
    KEYPOINT_ID_OFFSET = 0
    LINE_ID_OFFSET = 1
    GEO_CENTER_ID_OFFSET = 2
    HUMAN_MARKER_ID_VOL = 10

    HST_NS = HST_INFER_NODE
    HST_HUMAN_CURRENT_TRAJ = 0
    HST_HUMAN_FUTURE_TRAJ = 1

    DEFAULT_OFFSET = 100            # we have 100 slots for special id


## marker_id = human_id marker_offset

def get_marker_id(offset: int, ns: str, human_id=None):
    """
    human_id is a (1,) vector or int or None
    """
    if ns == marker_id_offset.SKELETON_NS.value:
        if type(human_id) == np.ndarray:
            assert human_id.shape == (1,)
            marker_id = int(human_id.item() * marker_id_offset.HUMAN_MARKER_ID_VOL.value + offset)
        elif type(human_id) == int:
            marker_id = int(human_id * marker_id_offset.HUMAN_MARKER_ID_VOL.value + offset)
        else:
            raise TypeError("human id should be (1,) numpy vector or an int")

    elif ns == marker_id_offset.HST_NS.value:
        marker_id = int(offset + marker_id_offset.DEFAULT_OFFSET.value)
    
    else:
        raise NameError(f"invalid namespace: {ns}")
    
    return marker_id