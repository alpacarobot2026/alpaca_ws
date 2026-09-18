# from copy import deepcopy
# from . import tracklet
# from .redundancy import RedundancyModule
# from scipy.optimize import linear_sum_assignment
# from .frame_data import FrameData
# from .update_info_data import UpdateInfoData
# from .data_protos import BBox, Validity
# from .association import associate_dets_to_tracks
# from . import visualization
# from . import redundancy
# import pdb, os


# class MOTModel:
#     def __init__(self, configs):
#         self.trackers = list()         # tracker for each single tracklet
#         self.frame_count = 0           # record for the frames
#         self.count = 0                 # record the obj number to assign ids
#         self.time_stamp = None         # the previous time stamp
#         self.redundancy = RedundancyModule(configs) # module for no detection cases

#         non_key_redundancy_config = deepcopy(configs)
#         non_key_redundancy_config['redundancy'] = non_key_redundancy_config.get('non_key_redundancy', configs['redundancy'])
#         self.non_key_redundancy = RedundancyModule(non_key_redundancy_config)

#         self.configs = configs
#         self.match_type = configs['running']['match_type']
#         self.score_threshold = configs['running']['score_threshold']
#         self.asso = configs['running']['asso']
#         self.asso_thres = configs['running']['asso_thres'][self.asso]
#         self.motion_model = configs['running']['motion_model']

#         self.max_age = configs['running']['max_age_since_update']
#         self.min_hits = configs['running']['min_hits_to_birth']

#     @property
#     def has_velo(self):
#         return not (self.motion_model == 'kf' or self.motion_model == 'fbkf' or self.motion_model == 'ma')
    
#     def frame_mot(self, input_data: FrameData):
#         """ For each frame input, generate the latest mot results
#         Args:
#             input_data (FrameData): input data, including detection bboxes and ego information
#         Returns:
#             tracks on this frame: [(bbox0, id0), (bbox1, id1), ...]
#         """
#         self.frame_count += 1

#         # initialize the time stamp on frame 0
#         if self.time_stamp is None:
#             self.time_stamp = input_data.time_stamp

#         if input_data.aux_info is not None and not input_data.aux_info['is_key_frame']:
#             result = self.non_key_frame_mot(input_data)
#             return result
    
#         if 'kf' in self.motion_model:
#             matched, unmatched_dets, unmatched_trks = self.forward_step_trk(input_data)
        
#         time_lag = input_data.time_stamp - self.time_stamp
#         # update the matched tracks
#         for t, trk in enumerate(self.trackers):
#             if t not in unmatched_trks:
#                 for k in range(len(matched)):
#                     if matched[k][1] == t:
#                         d = matched[k][0]
#                         break
#                 if self.has_velo:
#                     aux_info = {
#                         'velo': list(input_data.aux_info['velos'][d]), 
#                         'is_key_frame': input_data.aux_info['is_key_frame']}
#                 else:
#                     aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
#                 update_info = UpdateInfoData(mode=1, bbox=input_data.dets[d], ego=input_data.ego, 
#                     frame_index=self.frame_count, pc=input_data.pc, 
#                     dets=input_data.dets, aux_info=aux_info)
#                 trk.update(update_info)
#             else:
#                 result_bbox, update_mode, aux_info = self.redundancy.infer(trk, input_data, time_lag)
#                 aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
#                 update_info = UpdateInfoData(mode=update_mode, bbox=result_bbox, 
#                     ego=input_data.ego, frame_index=self.frame_count, 
#                     pc=input_data.pc, dets=input_data.dets, aux_info=aux_info)
#                 trk.update(update_info)
        
#         # create new tracks for unmatched detections
#         for index in unmatched_dets:
#             if self.has_velo:
#                 aux_info = {
#                     'velo': list(input_data.aux_info['velos'][index]), 
#                     'is_key_frame': input_data.aux_info['is_key_frame']}
#             else:
#                 aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}


            
#             # if track is in wedge, do max_age_since_update = 30, else normal

#             track = tracklet.Tracklet(self.configs, self.count, input_data.dets[index], input_data.det_types[index], 
#                 self.frame_count, aux_info=aux_info, time_stamp=input_data.time_stamp)
            
            
#             self.trackers.append(track)
#             self.count += 1
        
#         # remove dead tracks
#         track_num = len(self.trackers)
#         for index, trk in enumerate(reversed(self.trackers)):
#             if trk.death(self.frame_count):
#                 self.trackers.pop(track_num - 1 - index)
        
#         # output the results
#         result = list()
#         for trk in self.trackers:
#             state_string = trk.state_string(self.frame_count)
#             result.append((trk.get_state(), trk.id, state_string, trk.det_type))
        
#         # wrap up and update the information about the mot trackers
#         self.time_stamp = input_data.time_stamp
#         for trk in self.trackers:
#             trk.sync_time_stamp(self.time_stamp)

#         return result
    
#     def forward_step_trk(self, input_data: FrameData):
#         dets = input_data.dets
#         det_indexes = [i for i, det in enumerate(dets) if det.s >= self.score_threshold]
#         dets = [dets[i] for i in det_indexes]

#         # prediction and association
#         trk_preds = list()
#         for trk in self.trackers:
#             trk_preds.append(trk.predict(input_data.time_stamp, input_data.aux_info['is_key_frame']))
        
#         # for m-distance association
#         trk_innovation_matrix = None
#         if self.asso == 'm_dis':
#             trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers] 

#         matched, unmatched_dets, unmatched_trks = associate_dets_to_tracks(dets, trk_preds, 
#             self.match_type, self.asso, self.asso_thres, trk_innovation_matrix)
        
#         for k in range(len(matched)):
#             matched[k][0] = det_indexes[matched[k][0]]
#         for k in range(len(unmatched_dets)):
#             unmatched_dets[k] = det_indexes[unmatched_dets[k]]
#         return matched, unmatched_dets, unmatched_trks
    
#     def non_key_forward_step_trk(self, input_data: FrameData):
#         """ tracking on non-key frames (for nuScenes)
#         """
#         dets = input_data.dets
#         det_indexes = [i for i, det in enumerate(dets) if det.s >= 0.5]
#         dets = [dets[i] for i in det_indexes]

#         # prediction and association
#         trk_preds = list()
#         for trk in self.trackers:
#             trk_preds.append(trk.predict(input_data.time_stamp, input_data.aux_info['is_key_frame']))
        
#         # for m-distance association
#         trk_innovation_matrix = None
#         if self.asso == 'm_dis':
#             trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers] 

#         matched, unmatched_dets, unmatched_trks = associate_dets_to_tracks(dets, trk_preds, 
#             self.match_type, self.asso, self.asso_thres, trk_innovation_matrix)
        
#         for k in range(len(matched)):
#             matched[k][0] = det_indexes[matched[k][0]]
#         for k in range(len(unmatched_dets)):
#             unmatched_dets[k] = det_indexes[unmatched_dets[k]]
#         return matched, unmatched_dets, unmatched_trks
    
#     def non_key_frame_mot(self, input_data: FrameData):
#         """ tracking on non-key frames (for nuScenes)
#         """

#         self.frame_count += 1
#         print (f'frame count is {self.frame_count}')
#         # initialize the time stamp on frame 0
#         if self.time_stamp is None:
#             self.time_stamp = input_data.time_stamp
        
#         if 'kf' in self.motion_model:
#             matched, unmatched_dets, unmatched_trks = self.non_key_forward_step_trk(input_data)
#         time_lag = input_data.time_stamp - self.time_stamp

#         redundancy_bboxes, update_modes = self.non_key_redundancy.bipartite_infer(input_data, self.trackers)
#         # update the matched tracks
#         for t, trk in enumerate(self.trackers):
#             print ("self.trackers length: ", len(self.trackers))
#             if t not in unmatched_trks:
#                 for k in range(len(matched)):
#                     if matched[k][1] == t:
#                         d = matched[k][0]
#                         break
#                 if self.has_velo:
#                     aux_info = {
#                         'velo': list(input_data.aux_info['velos'][d]), 
#                         'is_key_frame': input_data.aux_info['is_key_frame']}
#                 else:
#                     aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
#                 update_info = UpdateInfoData(mode=1, bbox=input_data.dets[d], ego=input_data.ego, 
#                     frame_index=self.frame_count, pc=input_data.pc, 
#                     dets=input_data.dets, aux_info=aux_info)
#                 trk.update(update_info)
#             else:
#                 aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
#                 update_info = UpdateInfoData(mode=update_modes[t], bbox=redundancy_bboxes[t], 
#                     ego=input_data.ego, frame_index=self.frame_count, 
#                     pc=input_data.pc, dets=input_data.dets, aux_info=aux_info)
#                 trk.update(update_info)
        
#         # output the results
#         result = list()
#         for trk in self.trackers:
#             state_string = trk.state_string(self.frame_count)
#             result.append((trk.get_state(), trk.id, state_string, trk.det_type))

#         # wrap up and update the information about the mot trackers
#         self.time_stamp = input_data.time_stamp
#         for trk in self.trackers:
#             trk.sync_time_stamp(self.time_stamp)

#         return result



from copy import deepcopy
from unittest import result
from . import tracklet
from .redundancy import RedundancyModule
from scipy.optimize import linear_sum_assignment
from .frame_data import FrameData
from .update_info_data import UpdateInfoData
from .data_protos import BBox, Validity
from .association import associate_dets_to_tracks
from . import visualization
from . import redundancy
import pdb, os
import numpy as np


class MOTModel:
    def __init__(self, configs):
        self.trackers = list()         # tracker for each single tracklet
        self.frame_count = 0           # record for the frames
        self.count = 0                 # record the obj number to assign ids
        self.time_stamp = None         # the previous time stamp
        self.redundancy = RedundancyModule(configs) # module for no detection cases

        non_key_redundancy_config = deepcopy(configs)
        non_key_redundancy_config['redundancy'] = {
            'mode': 'mm',
            'det_score_threshold': {'giou': 0.1, 'iou': 0.1, 'euler': 0.1, 'm_dis': 0.4, 'center': 0.1},
            'det_dist_threshold': {'giou': -0.5, 'iou': 0.1, 'euler': 4, 'm_dis': 12, 'center': 0.75}
        }
        self.non_key_redundancy = RedundancyModule(non_key_redundancy_config)

        self.configs = configs
        self.match_type = configs['running']['match_type']
        self.score_threshold = configs['running']['score_threshold']
        self.asso = configs['running']['asso']
        self.asso_thres = configs['running']['asso_thres'][self.asso]
        self.motion_model = configs['running']['motion_model']

        self.max_age = configs['running']['max_age_since_update']
        self.min_hits = configs['running']['min_hits_to_birth']

    @property
    def has_velo(self):
        return not (self.motion_model == 'kf' or self.motion_model == 'fbkf' or self.motion_model == 'ma')

    def is_in_blind_spot(self, state, robot_state=None):
        """Check if the given state (box) sits in the robot's blind spot."""

        def _wrap_to_pi(angle_rad: float) -> float:
            return np.arctan2(np.sin(angle_rad), np.cos(angle_rad))

        def _yaw_from_orientation(orientation) -> float:
            if orientation is None:
                return 0.0
            if hasattr(orientation, 'w'):
                qx = getattr(orientation, 'x', 0.0)
                qy = getattr(orientation, 'y', 0.0)
                qz = getattr(orientation, 'z', 0.0)
                qw = getattr(orientation, 'w', 1.0)
                return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            return float(orientation)

        blind_spots = self.configs.get('running', {}).get(
            'blind_spots',
            [(150, 180), (-180, -150)],
        )

        if robot_state is None or robot_state.pose is None:
            angle = np.degrees(np.arctan2(state.y, state.x))
        else:
            robot_pose = robot_state.pose
            robot_yaw = _yaw_from_orientation(robot_pose.orientation)
            dx = state.x - robot_pose.position.x
            dy = state.y - robot_pose.position.y
            if np.isclose(dx, 0.0) and np.isclose(dy, 0.0):
                return False
            relative_angle = _wrap_to_pi(np.arctan2(dy, dx) - robot_yaw)
            angle = np.degrees(relative_angle)

        for (start, end) in blind_spots:
            if start <= angle <= end:
                print ('in blind spots')
                return True

        return False
    
    def frame_mot(self, input_data: FrameData, robot_state):
        """ For each frame input, generate the latest mot results
        Args:
            input_data (FrameData): input data, including detection bboxes and ego information
        Returns:
            tracks on this frame: [(bbox0, id0), (bbox1, id1), ...]
        """
        self.frame_count += 1

        # initialize the time stamp on frame 0
        if self.time_stamp is None:
            self.time_stamp = input_data.time_stamp

        if not input_data.aux_info['is_key_frame']:
            result = self.non_key_frame_mot(input_data)
            return result
    
        if 'kf' in self.motion_model:
            matched, unmatched_dets, unmatched_trks = self.forward_step_trk(input_data, robot_state)
        
        time_lag = input_data.time_stamp - self.time_stamp
        # update the matched tracks
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                for k in range(len(matched)):
                    if matched[k][1] == t:
                        d = matched[k][0]
                        break
                if self.has_velo:
                    aux_info = {
                        'velo': list(input_data.aux_info['velos'][d]), 
                        'is_key_frame': input_data.aux_info['is_key_frame']}
                else:
                    aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=1, bbox=input_data.dets[d], ego=input_data.ego, 
                    frame_index=self.frame_count, pc=input_data.pc, 
                    dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)
            else:
                result_bbox, update_mode, aux_info = self.redundancy.infer(trk, input_data, time_lag)
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=update_mode, bbox=result_bbox, 
                    ego=input_data.ego, frame_index=self.frame_count, 
                    pc=input_data.pc, dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)
        
        # create new tracks for unmatched detections
        for index in unmatched_dets:
            if self.has_velo:
                aux_info = {
                    'velo': list(input_data.aux_info['velos'][index]), 
                    'is_key_frame': input_data.aux_info['is_key_frame']}
            else:
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}

            track = tracklet.Tracklet(self.configs, self.count, input_data.dets[index], input_data.det_types[index], 
                self.frame_count, aux_info=aux_info, time_stamp=input_data.time_stamp)
            self.trackers.append(track)
            self.count += 1
        
        # remove dead tracks
        track_num = len(self.trackers)
        for index, trk in enumerate(reversed(self.trackers)):
            if trk.death(self.frame_count):
                self.trackers.pop(track_num - 1 - index)
                state = trk.get_state()
                angle = np.degrees(np.arctan2(state.y, state.x))
                # print(f"[Death] Track {trk.id} DIED at {angle:.1f} degrees (In Blind Spot: {self.is_in_blind_spot(state)})", flush=True)
        
        # Output only matured/valid tracks. Returning every live internal track
        # publishes tentative births and stale unmatched tracks, which hurts
        # real-time precision and does not reflect the intended life manager.
        result = list()
        for trk in self.trackers:
            if not trk.valid_output(self.frame_count):
                continue
            state_string = trk.state_string(self.frame_count)
            result.append((trk.get_state(), trk.id, state_string, trk.det_type))
        
        # wrap up and update the information about the mot trackers
        self.time_stamp = input_data.time_stamp
        for trk in self.trackers:
            trk.sync_time_stamp(self.time_stamp)

        return result
    
    def forward_step_trk(self, input_data: FrameData, robot_state ):
        dets = input_data.dets
        det_indexes = [i for i, det in enumerate(dets) if det.s >= self.score_threshold]
        dets = [dets[i] for i in det_indexes]

        # prediction and association
        trk_preds = list()
        for trk in self.trackers:
            # Check if track is in blind spot
            in_blind_spot = self.is_in_blind_spot(trk.get_state(), robot_state)
            # if in_blind_spot:
                # print(f"[BlindSpot] Track ID {trk.id} is in blind spot! Pos: ({trk.get_state().x:.2f}, {trk.get_state().y:.2f})", flush=True)
            pred_bbox = trk.predict(
                input_data.time_stamp,
                input_data.aux_info['is_key_frame'],
                in_blind_spot=in_blind_spot,
            )
            if self.asso == 'center':
                pred_bbox = trk.get_state()
            trk_preds.append(pred_bbox)
        
        # for m-distance association
        trk_innovation_matrix = None
        if self.asso == 'm_dis':
            trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers] 

        matched, unmatched_dets, unmatched_trks = associate_dets_to_tracks(dets, trk_preds, 
            self.match_type, self.asso, self.asso_thres, trk_innovation_matrix)
        
        for k in range(len(matched)):
            matched[k][0] = det_indexes[matched[k][0]]
        for k in range(len(unmatched_dets)):
            unmatched_dets[k] = det_indexes[unmatched_dets[k]]
        return matched, unmatched_dets, unmatched_trks
    
    def non_key_forward_step_trk(self, input_data: FrameData):
        """ tracking on non-key frames (for nuScenes)
        """
        dets = input_data.dets
        det_indexes = [i for i, det in enumerate(dets) if det.s >= 0.5]
        dets = [dets[i] for i in det_indexes]

        # prediction and association
        trk_preds = list()
        for trk in self.trackers:
            in_blind_spot = self.is_in_blind_spot(trk.get_state())
            if in_blind_spot:
                print(f"[BlindSpot] Track ID {trk.id} is in blind spot! Pos: ({trk.get_state().x:.2f}, {trk.get_state().y:.2f})", flush=True)
            trk_preds.append(trk.predict(input_data.time_stamp, input_data.aux_info['is_key_frame'], in_blind_spot=False))
        
        # for m-distance association
        trk_innovation_matrix = None
        if self.asso == 'm_dis':
            trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers] 

        matched, unmatched_dets, unmatched_trks = associate_dets_to_tracks(dets, trk_preds, 
            self.match_type, self.asso, self.asso_thres, trk_innovation_matrix)
        
        for k in range(len(matched)):
            matched[k][0] = det_indexes[matched[k][0]]
        for k in range(len(unmatched_dets)):
            unmatched_dets[k] = det_indexes[unmatched_dets[k]]
        return matched, unmatched_dets, unmatched_trks
    
    def non_key_frame_mot(self, input_data: FrameData):
        """ tracking on non-key frames (for nuScenes)
        """
        self.frame_count += 1
        # initialize the time stamp on frame 0
        if self.time_stamp is None:
            self.time_stamp = input_data.time_stamp
        
        if 'kf' in self.motion_model:
            matched, unmatched_dets, unmatched_trks = self.non_key_forward_step_trk(input_data)
        time_lag = input_data.time_stamp - self.time_stamp

        redundancy_bboxes, update_modes = self.non_key_redundancy.bipartite_infer(input_data, self.trackers)
        # update the matched tracks
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                for k in range(len(matched)):
                    if matched[k][1] == t:
                        d = matched[k][0]
                        break
                if self.has_velo:
                    aux_info = {
                        'velo': list(input_data.aux_info['velos'][d]), 
                        'is_key_frame': input_data.aux_info['is_key_frame']}
                else:
                    aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=1, bbox=input_data.dets[d], ego=input_data.ego, 
                    frame_index=self.frame_count, pc=input_data.pc, 
                    dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)
            else:
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=update_modes[t], bbox=redundancy_bboxes[t], 
                    ego=input_data.ego, frame_index=self.frame_count, 
                    pc=input_data.pc, dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)
        
        # Output only matured/valid tracks on non-key frames as well.
        result = list()
        for trk in self.trackers:
            if not trk.valid_output(self.frame_count):
                continue
            state_string = trk.state_string(self.frame_count)
            result.append((trk.get_state(), trk.id, state_string, trk.det_type))

        # wrap up and update the information about the mot trackers
        self.time_stamp = input_data.time_stamp
        for trk in self.trackers:
            trk.sync_time_stamp(self.time_stamp)

        return result
