
import numpy as np

# Message type
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Header, ColorRGBA, MultiArrayDimension
from geometry_msgs.msg import PoseArray, Pose, Quaternion, Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge

from hst_infer.node_config import *
from hst_infer.utils.logger import logger
from hst_infer.utils.humanid_to_markerid import get_marker_id, marker_id_offset

import time

def add_multihuman_future_pos_markers(
        multi_human_pos_ATMD: np.ndarray,
        multi_human_mask_AT: np.ndarray,
        modes_prob: np.ndarray,
        present_idx: int = HISTORY_LENGTH,
        frame_id: str = STRETCH_BASE_FRAME,
        ns: str = marker_id_offset.HST_NS.value,
        color: dict = dict(r=0.5, g=0.5, b=0.0, a=0.7),
        heatmap: bool = True,
        ) -> Marker:
    """
    @multi_human_pos_ATMD: np.ndarray (A,T,M,2)
    @multi_human_mask_AT: np.ndarray (A,T)
    @frame_id: str, marker's tf2 name
    return (points_list, colors_list)
    """
    marker_id = get_marker_id(offset=marker_id_offset.HST_HUMAN_FUTURE_TRAJ.value,
                              ns=ns,
                              )
    
    # Marker building
    marker = Marker()
    marker.header.frame_id = frame_id

    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.POINTS
    marker.action = Marker.ADD

    # marker.pose = Pose()
    marker.scale.x = 0.03
    marker.scale.y = 0.03

    points_list: list[Point] = list()
    colors_list: list[ColorRGBA] = list()
    color_rgba = ColorRGBA(**color)
    color_heatmap = modes_prob_to_heatmap_color(modes_prob)

    A,T,M,D = multi_human_pos_ATMD.shape       # [A,T,2]
    print(A, T, M, D)
    print(time.time())
    for agent_idx in range(A):
        if multi_human_mask_AT[agent_idx,present_idx]:
            #print("INDICSE TRUE: ", agent_idx, present_idx)
            # # current
            # current_point = Point(
            #     x= multi_human_pos_ATMD[agent_idx,present_idx+1,2,0].item(),
            #     y= multi_human_pos_ATMD[agent_idx,present_idx+1,2,1].item(),
            #     z= float(0),
            #     )
            # points_list.append(current_point)
            # colors_list.append(current_color)

            # future
            for t_idx in range(present_idx+1,T):
                for m_idx in range(M):
                    future_point = Point(
                        x = multi_human_pos_ATMD[agent_idx,t_idx,m_idx,0].item(),
                        y = multi_human_pos_ATMD[agent_idx,t_idx,m_idx,1].item(),
                        z = float(),
                    )
                    #print("FUTURE POINT: ", agent_idx, t_idx, m_idx, future_point.x, future_point.y)
                    points_list.append(future_point)
                    if heatmap:
                        colors_list.append(color_heatmap[m_idx])
                    else:
                        colors_list.append(color_rgba)

    marker.points = points_list
    marker.colors = colors_list
    
    return marker


def add_multihuman_current_pos_markers(
        multi_human_history_pos_ATD: np.ndarray,
        multi_human_mask_AT: np.ndarray,
        present_idx: int = HISTORY_LENGTH,
        frame_id: str = STRETCH_BASE_FRAME,
        ns: str = marker_id_offset.HST_NS.value,
        color: dict = dict(r=1.0, g=0.5, b=0.5, a=1.0),
        ) -> Marker:
    
    marker_id = get_marker_id(offset=marker_id_offset.HST_HUMAN_CURRENT_TRAJ.value,
                              ns=ns,
                              )
    
    # Marker building
    marker = Marker()
    marker.header.frame_id = frame_id

    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.POINTS
    marker.action = Marker.ADD

    # marker.pose = Pose()
    marker.scale.x = 0.1
    marker.scale.y = 0.1

    points_list: list[Point] = list()
    colors_list: list[ColorRGBA] = list()
    color_rgba = ColorRGBA(**color)

    A,T,D = multi_human_history_pos_ATD.shape

    for agent_idx in range(A):
        if multi_human_mask_AT[agent_idx,present_idx]:

            current_point = Point(
                x = multi_human_history_pos_ATD[agent_idx,present_idx,0].item(),
                y = multi_human_history_pos_ATD[agent_idx,present_idx,1].item(),
                z = float(),
            )
            points_list.append(current_point)
            colors_list.append(color_rgba)
    
    marker.points = points_list
    marker.colors = colors_list
    
    return marker


def delete_multihuman_pos_markers(
        frame_id: str = STRETCH_BASE_FRAME,
        ns: str = marker_id_offset.HST_NS.value,
        ) -> MarkerArray:

    markers_list = list()
    for offset in (marker_id_offset.HST_HUMAN_FUTURE_TRAJ.value, 
                   marker_id_offset.HST_HUMAN_CURRENT_TRAJ.value,):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.ns = ns
        marker.type = Marker.DELETE

        marker.id = get_marker_id(offset, ns)
        markers_list.append(marker)

    markerarray = MarkerArray()
    markerarray.markers = markers_list
    return markerarray


def modes_prob_to_heatmap_color(prob: np.ndarray, 
                                min=(0.0,0.0,1.0), 
                                max=(1.0,0.0,0.0),
                                color_enhance = True,
                                ) -> list[ColorRGBA]:
    """
    Prob [0,1] -> RGB (1,1,1)
    @prob: (M,)
    """
    min, max = np.array(min), np.array(max)
    M = prob.shape[0]
    if color_enhance:
        thed = 3
        if M >= thed:
            prob_enhanced = prob / np.max(prob) * (M-2) / M
        else:
            prob_enhanced = prob / np.max(prob) * (thed-2) / (thed)
    else:
        prob_enhanced = prob.copy()

    prob_enhanced_mat = np.repeat(prob_enhanced[...,np.newaxis], 3, axis=-1)
    min = np.repeat(min[np.newaxis,...], M, axis=0)
    max = np.repeat(max[np.newaxis,...], M, axis=0)

    colors = min + prob_enhanced_mat * (max - min)
    colors = colors.astype(float)

    color_list: list[ColorRGBA] = list()
    for color, prob_ in zip(colors, prob_enhanced):
        rgb_dict = dict( r=color[0].item(), g=color[1].item(), b=color[2].item(), a=prob_.item())

        # color_list.append( ColorRGBA(r=0.7,g=0.0,b=0.3,a=1.0) )

        color_list.append( ColorRGBA(**rgb_dict) )
    
    return color_list