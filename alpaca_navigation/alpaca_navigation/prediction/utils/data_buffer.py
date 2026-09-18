
import numpy as np
from collections import deque

from geometry_msgs.msg import PoseArray, Pose, Quaternion, Point, Point32, PointStamped

from skeleton_interfaces.msg import MultiHumanSkeleton, HumanSkeleton
from hst_infer.utils import skeleton2msg
from hst_infer.utils.logger import logger
from hst_infer.node_config import *
from alpaca_navigation.prediction.human_scene_transformer.config import hst_config

class skeleton_buffer():
    def __init__(self, 
                 history_len:int = hst_config.hst_model_param.num_history_steps,
                 window_len: int = hst_config.hst_model_param.num_steps,
                 ):
        
        self.maxlen = history_len + 1   # history + presence
        self.present_idx = history_len + 1
        self.buffer: deque[tuple[list[HumanSkeleton], set[int]]] = deque(maxlen=self.maxlen)
        self.existing_id = dict()       # dict [ id: index ]
        self.window_len = window_len
        self.humanID_in_window: list[int] = list()
        # self.empty: bool = True
        # self.full: bool = False

    def receive_msg(self, data: MultiHumanSkeleton):
        """
        append (list[HumanSkeleton], id_set) to the deque

        if non-exsiting skeleton, put (list(),set()) into buffer
        """
        header = data.header

        multihuman_data: list[HumanSkeleton] = data.multi_human_skeleton
        multihumanID_set = set()

        if multihuman_data == list():
            self.buffer.append((multihuman_data, set()))
            return

        else:
            for human_data in multihuman_data:
                multihumanID_set.add(human_data.human_id)

            self.buffer.append((multihuman_data, multihumanID_set))
            return


    def get_current_multihumanID_set(self):
        """
        return a set of current human ID
        """
        if len(self.buffer) != 0:
            multihumanID_set = self.buffer[-1] [1]
        else:
            multihumanID_set = None
        return multihumanID_set


    def get_data_array(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        read the buffer and return keypoints_ATKD, human_center_ATD, keypoint_mask_ATK
        
        keypoints_ATKD: (A, T, K, D)
        human_center_ATD: (A, T, D)
        keypoint_mask_ATK: (A, T, K)

        A: agents, T: time, K: keypoints, D: dimensions
        """

        msg_seq: list[tuple[list[HumanSkeleton], set[int]]] = list(self.buffer)
        
        keypoints_ATKD: np.ndarray = np.zeros((MAX_AGENT_NUM, self.window_len, KEYPOINT_NUM, DIM_XYZ))
        center_ATD: np.ndarray = np.zeros((MAX_AGENT_NUM, self.window_len, DIM_XYZ))
        mask_ATK: np.ndarray = np.zeros((MAX_AGENT_NUM, self.window_len, KEYPOINT_NUM), dtype=bool)

        # TODO: convert msg into array, put the array in the large array
        # x x x x m m m 
        # x x x x m m m
        # x x x x x x x 

        # all the exsisting human in the buffer
        allhumanID_set = set()
        for _, multihumanID_set in msg_seq:
            allhumanID_set = allhumanID_set | multihumanID_set      # O(m+n)
        allhuamnID_list = list(allhumanID_set)

        # allocate the agent positions array
        id2idx = dict()
        for idx, id in enumerate(allhuamnID_list, start=0):
            id2idx[id] = idx

        t_start = self.maxlen - len(msg_seq)      # T axis
        # external loop for time
        for t_idx, (multihuman_data, _) in enumerate(msg_seq, start=t_start):
            # internal loop for agents, A axis
            for human_data in multihuman_data:
                id = human_data.human_id
                a_idx = id2idx[id]
                geo_center = skeleton2msg.point_to_np_vector(human_data.human_center) 
                keypoint_list: list[Point] = human_data.keypoint_data

                for k_idx in range(KEYPOINT_NUM):
                    keypoint_vector = skeleton2msg.point_to_np_vector(keypoint_list[k_idx])
                    
                    keypoints_ATKD[a_idx, t_idx, k_idx, :] = keypoint_vector
                    mask_ATK[a_idx, t_idx, k_idx] = human_data.keypoint_mask[k_idx]

                center_ATD[a_idx, t_idx, :] = geo_center
        
        # buffer global attributes
        self.humanID_in_window = allhuamnID_list
        self.id2idx_in_window = id2idx

        return keypoints_ATKD, center_ATD, mask_ATK


    def __len__(self):
        return len(self.buffer)
    


def skeleton_arrays_to_dict(keypoints_ATKD: np.ndarray, 
                            position_ATD: np.ndarray, 
                            keypoint_mask_ATK: np.ndarray, 
                            robot_position: np.ndarray) -> dict:
    """
    convert arrays into hst data structure
    """


    # TODO: 17 keypoints to 33 keypoints
    # TODO: put agent keypoints on orgin

    # TODO: 'agent/orientation' should be [1, T, 1] np.nan because it is not in the feature