import numpy as np
from hst_infer.node_config import KEYPOINT_INTERPOLATION

YOLO_POSE_DICT = {
    0: "Nose",
    1: "Left-eye",
    2: "Right-eye",
    3: "Left-ear",
    4: "Right-ear",
    5: "Left-shoulder",
    6: "Right-shoulder",
    7: "Left-elbow",
    8: "Right-elbow",
    9: "Left-wrist",
    10: "Right-wrist",
    11: "Left-hip",
    12: "Right-hip",
    13: "Left-knee",
    14: "Right-knee",
    15: "Left-ankle",
    16: "Right-ankle",
}

BLAZE_POSE_DICT = {
    0: "Nose",
    1: "Left-eye-inner",
    2: "Left-eye",
    3: "Left-eye-outer",
    4: "Right-eye-inner",
    5: 'Right-eye',
    6: "Right-eye outer",
    7: "Left-ear",
    8: "Right-ear",
    9:" Mouth-left",
    10: "Mouth-right",
    11: "Left-shoulder",
    12: "Right-shoulder",
    13: "Left-elbow",
    14: "Right-elbow",
    15: "Left-wrist",
    16: "Right-wrist",
    17: "Left-pinky", #1 knuckle
    18: "Right-pinky", #1 knuckle
    19: "Left-index", #1 knuckle
    20: "Right-index",#1 knuckle
    21: "Left-thumb", #2 knuckle
    22: "Right-thumb", #2 knuckle
    23: "Left-hip",
    24: "Right-hip",
    25: "Left-knee",
    26: "Right-knee",
    27: "Left-ankle",
    28: "Right-ankle",
    29: "Left-heel",
    30: "Right-heel",
    31: "Left-foot-index",
    32: "Right-foot-index",
    }

def get_keypoint_mapping(old_dict: dict = YOLO_POSE_DICT, 
                         new_dict: dict = BLAZE_POSE_DICT,
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    return tuple(old_idx, new_idx), where they share the same value in each's dict respectively
    """
    old_idx = np.arange(len(old_dict),dtype=int)
    new_idx = np.zeros_like(old_idx, dtype=int)

    for key1, val1 in old_dict.items():
        for key2, val2 in new_dict.items():
            if val1 == val2:
                new_idx[key1] = key2
                break
        assert Exception, "Could find a mapping from old_dict to new_dict"
    
    return old_idx, new_idx


def valid_interpolation(keypoint_mask_ATK: np.ndarray):
    """
    @keypoint_mask_ATK: [A,T,K]
    return valid_AT: [A,T]
    """
    main_yolo_keypoints = np.array([1,2,5,6,7,8,11,12,13,14],dtype=int)
    valid_AT = np.all(keypoint_mask_ATK[...,main_yolo_keypoints], axis=-1, keepdims=False)
    return valid_AT


class human_keypoints:

    old_idx, new_idx = get_keypoint_mapping(YOLO_POSE_DICT, BLAZE_POSE_DICT)

    @classmethod
    def map_yolo_to_hst_batch(cls, keypoints_ATKD: np.ndarray,
                        keypoint_mask_ATK: np.ndarray, 
                        keypoint_center_ATD: np.ndarray,
                        hst_K: int = len(BLAZE_POSE_DICT),
                        interpolation: bool = KEYPOINT_INTERPOLATION,
                        ) -> np.ndarray:
        """
        input: keypoints [A,T,17,3], mask [A,T,K]
        return: keypoinst [1,A,T,99]
        """
        A,T,K,D = keypoints_ATKD.shape 
        # Normalize keypoints
        keypoints_ATKD = keypoints_ATKD - keypoint_center_ATD[:,:,np.newaxis,:]

        keypoint_mask_ATKD = np.repeat(keypoint_mask_ATK[...,np.newaxis], 3, axis=3)
        assert keypoint_mask_ATKD.shape == (A,T,K,3)

        keypoints_ATKD = np.where(keypoint_mask_ATKD, keypoints_ATKD, np.nan)
        # init 33 keypoints with nan
        hst_keypoints_ATKD = np.full((A,T,hst_K,D), np.nan, dtype=float)

        if interpolation:
            # raise Exception("Interpolation Model to be implemented")
            valid_AT = valid_interpolation(keypoint_mask_ATK=keypoint_mask_ATK)
            for a_idx in range(A):
                for t_idx in range(T):
                    if valid_AT[a_idx,t_idx]:
                        hst_keypoints_ATKD[a_idx,t_idx,...] = interpolate_keypoints(
                            yolo_keypoints_KD=keypoints_ATKD[a_idx,t_idx,...],
                            yolo_keypoints_mask_K=keypoint_mask_ATK[a_idx,t_idx,...],
                            hst_K=hst_K
                            )        
        else:
            hst_keypoints_ATKD[:,:,cls.new_idx,:] = keypoints_ATKD[:,:,cls.old_idx,:]

        hst_keypoints_batch = hst_keypoints_ATKD.reshape((1,A,T,hst_K*D))
        assert hst_keypoints_batch.shape == (1,A,T,99)

        return hst_keypoints_batch
        

def interpolate_keypoints(yolo_keypoints_KD: np.ndarray, 
                          yolo_keypoints_mask_K: np.ndarray, 
                          hst_K: int) -> np.ndarray:
    """
    @yolo_keypoints_KD: [K,D]
    @yolo_keypoints_mask_K: [K,]
    
    return hst_pseodu_keypoints_KD: [K,D]
    """
    _, D = yolo_keypoints_KD.shape
    main_yolo_keypoints = np.array([0,1,2,5,6,7,8,11,12,13,14],dtype=int)
    hst_pseodu_keypoints_KD = np.full((hst_K,D), np.nan, dtype=float)

    # face
    hst_pseodu_keypoints_KD[0] = yolo_keypoints_KD[0]
    hst_pseodu_keypoints_KD[2] = yolo_keypoints_KD[1]
    hst_pseodu_keypoints_KD[5] = yolo_keypoints_KD[2]

    if yolo_keypoints_mask_K[3]:
        hst_pseodu_keypoints_KD[7] = yolo_keypoints_KD[3]
    else:
        hst_pseodu_keypoints_KD[7] = hst_pseodu_keypoints_KD[0] + 1.1*(hst_pseodu_keypoints_KD[2] - hst_pseodu_keypoints_KD[5])
    if yolo_keypoints_mask_K[4]:
        hst_pseodu_keypoints_KD[8] = yolo_keypoints_KD[4]
    else:
        hst_pseodu_keypoints_KD[8] = hst_pseodu_keypoints_KD[0] - 1.1*(hst_pseodu_keypoints_KD[2] - hst_pseodu_keypoints_KD[5])

    hst_pseodu_keypoints_KD[1] = hst_pseodu_keypoints_KD[2] + 0.15*(hst_pseodu_keypoints_KD[5]-hst_pseodu_keypoints_KD[2])
    hst_pseodu_keypoints_KD[3] = hst_pseodu_keypoints_KD[2] - 0.15*(hst_pseodu_keypoints_KD[5]-hst_pseodu_keypoints_KD[2])
    hst_pseodu_keypoints_KD[4] = hst_pseodu_keypoints_KD[5] - 0.15*(hst_pseodu_keypoints_KD[5]-hst_pseodu_keypoints_KD[2])
    hst_pseodu_keypoints_KD[6] = hst_pseodu_keypoints_KD[5] + 0.15*(hst_pseodu_keypoints_KD[5]-hst_pseodu_keypoints_KD[2])
    hst_pseodu_keypoints_KD[9] = hst_pseodu_keypoints_KD[0] + (hst_pseodu_keypoints_KD[0]-hst_pseodu_keypoints_KD[4])
    hst_pseodu_keypoints_KD[10] = hst_pseodu_keypoints_KD[0] + (hst_pseodu_keypoints_KD[0]-hst_pseodu_keypoints_KD[1])

    # body
    hst_pseodu_keypoints_KD[11] = yolo_keypoints_KD[5]
    hst_pseodu_keypoints_KD[12] = yolo_keypoints_KD[6]
    hst_pseodu_keypoints_KD[23] = yolo_keypoints_KD[11]
    hst_pseodu_keypoints_KD[24] = yolo_keypoints_KD[12]

    # arms
    hst_pseodu_keypoints_KD[13] = yolo_keypoints_KD[7]
    hst_pseodu_keypoints_KD[14] = yolo_keypoints_KD[8]

    if yolo_keypoints_mask_K[9]:
        hst_pseodu_keypoints_KD[15] = yolo_keypoints_KD[9]
    else:
        hst_pseodu_keypoints_KD[15] = hst_pseodu_keypoints_KD[13] + 0.5*(hst_pseodu_keypoints_KD[24]-hst_pseodu_keypoints_KD[12])
    if yolo_keypoints_mask_K[10]:
        hst_pseodu_keypoints_KD[16] = yolo_keypoints_KD[10]
    else:
        hst_pseodu_keypoints_KD[16] = hst_pseodu_keypoints_KD[14] + 0.5*(hst_pseodu_keypoints_KD[23]-hst_pseodu_keypoints_KD[11])
    
    hst_pseodu_keypoints_KD[17] = hst_pseodu_keypoints_KD[15] + 0.3*(hst_pseodu_keypoints_KD[13]-hst_pseodu_keypoints_KD[11])
    hst_pseodu_keypoints_KD[18] = hst_pseodu_keypoints_KD[16] + 0.3*(hst_pseodu_keypoints_KD[24]-hst_pseodu_keypoints_KD[12])
    hst_pseodu_keypoints_KD[19] = hst_pseodu_keypoints_KD[15] + 0.4*(hst_pseodu_keypoints_KD[15]-hst_pseodu_keypoints_KD[13])
    hst_pseodu_keypoints_KD[20] = hst_pseodu_keypoints_KD[16] + 0.4*(hst_pseodu_keypoints_KD[16]-hst_pseodu_keypoints_KD[14])
    hst_pseodu_keypoints_KD[21] = hst_pseodu_keypoints_KD[15] + 0.5*(hst_pseodu_keypoints_KD[15]-hst_pseodu_keypoints_KD[13])
    hst_pseodu_keypoints_KD[22] = hst_pseodu_keypoints_KD[16] + 0.5*(hst_pseodu_keypoints_KD[16]-hst_pseodu_keypoints_KD[14])

    # legs
    hst_pseodu_keypoints_KD[25] = yolo_keypoints_KD[13]
    hst_pseodu_keypoints_KD[26] = yolo_keypoints_KD[14]
    
    if yolo_keypoints_mask_K[15]:
        hst_pseodu_keypoints_KD[27] = yolo_keypoints_KD[15]
    else:
        hst_pseodu_keypoints_KD[27] = hst_pseodu_keypoints_KD[25] + 1.0*(hst_pseodu_keypoints_KD[26]-hst_pseodu_keypoints_KD[24])
    if yolo_keypoints_mask_K[16]:
        hst_pseodu_keypoints_KD[28] = yolo_keypoints_KD[16]
    else:
        hst_pseodu_keypoints_KD[28] = hst_pseodu_keypoints_KD[26] + 1.0*(hst_pseodu_keypoints_KD[25]-hst_pseodu_keypoints_KD[23])
    hst_pseodu_keypoints_KD[29] = hst_pseodu_keypoints_KD[27] + 0.28*(hst_pseodu_keypoints_KD[27]-hst_pseodu_keypoints_KD[25])
    hst_pseodu_keypoints_KD[30] = hst_pseodu_keypoints_KD[28] + 0.28*(hst_pseodu_keypoints_KD[28]-hst_pseodu_keypoints_KD[26])
    hst_pseodu_keypoints_KD[31] = hst_pseodu_keypoints_KD[29] + 1.2*(hst_pseodu_keypoints_KD[7]-hst_pseodu_keypoints_KD[0])
    hst_pseodu_keypoints_KD[32] = hst_pseodu_keypoints_KD[30] + 1.2*(hst_pseodu_keypoints_KD[8]-hst_pseodu_keypoints_KD[0])

    return hst_pseodu_keypoints_KD


if __name__ == "__main__":
    print('\n',human_keypoints.old_idx,'\n',human_keypoints.new_idx)     # good
    # print(len(BLAZE_POSE_DICT))
    A,T,K,D = 2,19,len(YOLO_POSE_DICT),3
    keypoints_ATKD = np.arange(A*T*K*D).reshape((A,T,K,D))
    keypoint_mask_ATK = np.random.randint(0,2, size=(A,T,K),dtype=bool)
    keypoint_center_ATD = np.random.random(size=(A,T,D))

    import time
    t1 = time.time()
    res = human_keypoints.map_yolo_to_hst_batch(keypoints_ATKD, keypoint_mask_ATK, keypoint_center_ATD, len(BLAZE_POSE_DICT), interpolation=True)
    t2 = time.time()

    print(t2-t1, res.shape)

    # print(keypoint_mask_ATK)
    # print(res)