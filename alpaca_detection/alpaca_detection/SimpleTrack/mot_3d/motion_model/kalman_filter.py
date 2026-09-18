# """ Many parts are borrowed from https://github.com/xinshuoweng/AB3DMOT
# """

# import numpy as np
# from ..data_protos import BBox
# from filterpy.kalman import KalmanFilter


# class KalmanFilterMotionModel:
#     def __init__(self, bbox: BBox, inst_type, time_stamp, covariance='default'):
#         # the time stamp of last observation
#         self.prev_time_stamp = time_stamp
#         self.latest_time_stamp = time_stamp
#         # define constant velocity model
#         self.score = bbox.s
#         self.inst_type = inst_type

#         self.kf = KalmanFilter(dim_x=10, dim_z=7) 
#         self.kf.x[:7] = BBox.bbox2array(bbox)[:7].reshape((7, 1))
#         self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
#                               [0,1,0,0,0,0,0,0,1,0],
#                               [0,0,1,0,0,0,0,0,0,1],
#                               [0,0,0,1,0,0,0,0,0,0],  
#                               [0,0,0,0,1,0,0,0,0,0],
#                               [0,0,0,0,0,1,0,0,0,0],
#                               [0,0,0,0,0,0,1,0,0,0],
#                               [0,0,0,0,0,0,0,1,0,0],
#                               [0,0,0,0,0,0,0,0,1,0],
#                               [0,0,0,0,0,0,0,0,0,1]])     

#         self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
#                               [0,1,0,0,0,0,0,0,0,0],
#                               [0,0,1,0,0,0,0,0,0,0],
#                               [0,0,0,1,0,0,0,0,0,0],
#                               [0,0,0,0,1,0,0,0,0,0],
#                               [0,0,0,0,0,1,0,0,0,0],
#                               [0,0,0,0,0,0,1,0,0,0]])
        
#         self.kf.B = np.zeros((10, 1))                     # dummy control transition matrix

#         # # with angular velocity
#         # self.kf = KalmanFilter(dim_x=11, dim_z=7)       
#         # self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
#         #                       [0,1,0,0,0,0,0,0,1,0,0],
#         #                       [0,0,1,0,0,0,0,0,0,1,0],
#         #                       [0,0,0,1,0,0,0,0,0,0,1],  
#         #                       [0,0,0,0,1,0,0,0,0,0,0],
#         #                       [0,0,0,0,0,1,0,0,0,0,0],
#         #                       [0,0,0,0,0,0,1,0,0,0,0],
#         #                       [0,0,0,0,0,0,0,1,0,0,0],
#         #                       [0,0,0,0,0,0,0,0,1,0,0],
#         #                       [0,0,0,0,0,0,0,0,0,1,0],
#         #                       [0,0,0,0,0,0,0,0,0,0,1]])     

#         # self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
#         #                       [0,1,0,0,0,0,0,0,0,0,0],
#         #                       [0,0,1,0,0,0,0,0,0,0,0],
#         #                       [0,0,0,1,0,0,0,0,0,0,0],
#         #                       [0,0,0,0,1,0,0,0,0,0,0],
#         #                       [0,0,0,0,0,1,0,0,0,0,0],
#         #                       [0,0,0,0,0,0,1,0,0,0,0]])

#         # allow passing a dict for covariance/model tuning
#         self.covariance_type = covariance
#         if isinstance(covariance, dict):
#             cov = covariance
#         else:
#             cov = {}

#         # sensible defaults — user can override via configs['running']['covariance']
#         meas_var = float(cov.get('measurement_var', 0.1))
#         pos_process_var = float(cov.get('pos_process_var', 1e-3))
#         vel_process_var = float(cov.get('vel_process_var', 1e-2))
#         vel_damping = float(cov.get('velocity_damping', 1.0))

#         # measurement uncertainty (7 dims: x,y,z,theta,l,w,h)
#         self.kf.R = np.eye(7) * meas_var

#         # state covariance: give high uncertainty to initial velocities
#         self.kf.P[7:, 7:] *= 1000.  	# velocity cov large initially
#         self.kf.P *= 10.

#         # process noise Q (10x10)
#         Q = np.zeros((10, 10))
#         # small process noise for position/shape states
#         Q[0, 0] = pos_process_var
#         Q[1, 1] = pos_process_var
#         Q[2, 2] = pos_process_var
#         Q[3, 3] = pos_process_var
#         Q[4, 4] = pos_process_var
#         Q[5, 5] = pos_process_var
#         Q[6, 6] = pos_process_var
#         # larger process noise for velocity states so they can adapt, but user can tune
#         Q[7, 7] = vel_process_var
#         Q[8, 8] = vel_process_var
#         Q[9, 9] = vel_process_var
#         self.kf.Q = Q

#         # velocity damping factor in prediction (1.0 means no damping)
#         self.velocity_damping = vel_damping

#         self.history = [bbox]
    
#     def predict(self, time_stamp=None):
#         """ For the motion prediction, use the get_prediction function.
#         """
#         self.kf.predict()
#         if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
#         if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
#         return

#     def update(self, det_bbox: BBox, aux_info=None): 
#         """ 
#         Updates the state vector with observed bbox.
#         """
#         bbox = BBox.bbox2array(det_bbox)[:7]

#         # full pipeline of kf, first predict, then update
#         self.predict()

#         ######################### orientation correction
#         if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
#         if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

#         new_theta = bbox[3]
#         if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
#         if new_theta < -np.pi: new_theta += np.pi * 2
#         bbox[3] = new_theta

#         predicted_theta = self.kf.x[3]
#         if np.abs(new_theta - predicted_theta) > np.pi / 2.0 and np.abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
#             self.kf.x[3] += np.pi       
#             if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
#             if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

#         # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
#         if np.abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
#             if new_theta > 0: self.kf.x[3] += np.pi * 2
#             else: self.kf.x[3] -= np.pi * 2

#         #########################     # flip

#         self.kf.update(bbox)
#         self.prev_time_stamp = self.latest_time_stamp

#         if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the rage
#         if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

#         if det_bbox.s is None:
#             self.score = self.score * 0.01
#         else:
#             self.score = det_bbox.s
        
#         cur_bbox = self.kf.x[:7].reshape(-1).tolist()
#         cur_bbox = BBox.array2bbox(cur_bbox + [self.score])
#         self.history[-1] = cur_bbox
#         return

#     def get_prediction(self, time_stamp=None):       
#         """
#         Advances the state vector and returns the predicted bounding box estimate.
#         """
#         time_lag = time_stamp - self.prev_time_stamp
#         self.latest_time_stamp = time_stamp
#         self.kf.F = np.array([[1,0,0,0,0,0,0,time_lag,0,0],      # state transition matrix
#                               [0,1,0,0,0,0,0,0,time_lag,0],
#                               [0,0,1,0,0,0,0,0,0,time_lag],
#                               [0,0,0,1,0,0,0,0,0,0],  
#                               [0,0,0,0,1,0,0,0,0,0],
#                               [0,0,0,0,0,1,0,0,0,0],
#                               [0,0,0,0,0,0,1,0,0,0],
#                               [0,0,0,0,0,0,0,1,0,0],
#                               [0,0,0,0,0,0,0,0,1,0],
#                               [0,0,0,0,0,0,0,0,0,1]])
#         pred_x = self.kf.get_prediction()[0]
#         # apply optional velocity damping when no measurement updates occur for a while
#         # damping factor of 1.0 = no damping; values <1.0 attenuate velocities each predict
#         if self.velocity_damping is not None and self.velocity_damping > 0.0 and self.velocity_damping < 1.0:
#             # apply per-predict damping; for larger time_lag you may want a stronger decay
#             try:
#                 self.kf.x[7, 0] *= float(self.velocity_damping)
#                 self.kf.x[8, 0] *= float(self.velocity_damping)
#                 self.kf.x[9, 0] *= float(self.velocity_damping)
#                 pred_x = self.kf.x.reshape(-1)
#             except Exception:
#                 pass
#         if pred_x[3] >= np.pi: pred_x[3] -= np.pi * 2
#         if pred_x[3] < -np.pi: pred_x[3] += np.pi * 2
#         pred_bbox = BBox.array2bbox(pred_x[:7].reshape(-1))

#         self.history.append(pred_bbox)
#         return pred_bbox

#     def get_state(self):
#         """
#         Returns the current bounding box estimate.
#         """
#         return self.history[-1]
    
#     def compute_innovation_matrix(self):
#         """ compute the innovation matrix for association with mahalonobis distance
#         """
#         return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R
    
#     def sync_time_stamp(self, time_stamp):
#         self.time_stamp = time_stamp
#         return

""" Many parts are borrowed from https://github.com/xinshuoweng/AB3DMOT
"""

import numpy as np
from ..data_protos import BBox
from filterpy.kalman import KalmanFilter


class KalmanFilterMotionModel:
    def __init__(self, bbox: BBox, inst_type, time_stamp, covariance='default'):
        # the time stamp of last observation
        self.prev_time_stamp = time_stamp
        self.latest_time_stamp = time_stamp
        # define constant velocity model
        self.score = bbox.s
        self.inst_type = inst_type

        self.kf = KalmanFilter(dim_x=10, dim_z=7) 
        self.kf.x[:7] = BBox.bbox2array(bbox)[:7].reshape((7, 1))
        self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
                              [0,1,0,0,0,0,0,0,1,0],
                              [0,0,1,0,0,0,0,0,0,1],
                              [0,0,0,1,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,1]])     

        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
                              [0,1,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0]])
        
        self.kf.B = np.zeros((10, 1))                     # dummy control transition matrix

        # # with angular velocity
        # self.kf = KalmanFilter(dim_x=11, dim_z=7)       
        # self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
        #                       [0,1,0,0,0,0,0,0,1,0,0],
        #                       [0,0,1,0,0,0,0,0,0,1,0],
        #                       [0,0,0,1,0,0,0,0,0,0,1],  
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0],
        #                       [0,0,0,0,0,0,0,1,0,0,0],
        #                       [0,0,0,0,0,0,0,0,1,0,0],
        #                       [0,0,0,0,0,0,0,0,0,1,0],
        #                       [0,0,0,0,0,0,0,0,0,0,1]])     

        # self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
        #                       [0,1,0,0,0,0,0,0,0,0,0],
        #                       [0,0,1,0,0,0,0,0,0,0,0],
        #                       [0,0,0,1,0,0,0,0,0,0,0],
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0]])

        self.covariance_type = covariance
        # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
        self.kf.P[7:, 7:] *= 1000. 	# state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        self.kf.P *= 10.

        # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
        # self.kf.Q[7:, 7:] *= 0.01

        self.history = [bbox]
    
    def predict(self, time_stamp=None):
        """ For the motion prediction, use the get_prediction function.
        """
        self.kf.predict()
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        return

    def update(self, det_bbox: BBox, aux_info=None): 
        """ 
        Updates the state vector with observed bbox.
        """
        bbox = BBox.bbox2array(det_bbox)[:7]

        # full pipeline of kf, first predict, then update
        self.predict()

        ######################### orientation correction
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_theta = bbox[3]
        if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox[3] = new_theta

        predicted_theta = self.kf.x[3]
        if np.abs(new_theta - predicted_theta) > np.pi / 2.0 and np.abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi       
            if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
            if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if np.abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0: self.kf.x[3] += np.pi * 2
            else: self.kf.x[3] -= np.pi * 2

        #########################     # flip

        self.kf.update(bbox)
        self.prev_time_stamp = self.latest_time_stamp

        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the rage
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        if det_bbox.s is None:
            self.score = self.score * 0.01
        else:
            self.score = det_bbox.s
        
        cur_bbox = self.kf.x[:7].reshape(-1).tolist()
        cur_bbox = BBox.array2bbox(cur_bbox + [self.score])
        self.history[-1] = cur_bbox
        return

    def get_prediction(self, time_stamp=None):       
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        time_lag = time_stamp - self.prev_time_stamp
        self.latest_time_stamp = time_stamp
        self.kf.F = np.array([[1,0,0,0,0,0,0,time_lag,0,0],      # state transition matrix
                              [0,1,0,0,0,0,0,0,time_lag,0],
                              [0,0,1,0,0,0,0,0,0,time_lag],
                              [0,0,0,1,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,1]])
        pred_x = self.kf.get_prediction()[0]
        if pred_x[3] >= np.pi: pred_x[3] -= np.pi * 2
        if pred_x[3] < -np.pi: pred_x[3] += np.pi * 2
        pred_bbox = BBox.array2bbox(pred_x[:7].reshape(-1))

        self.history.append(pred_bbox)
        return pred_bbox

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.history[-1]
    
    def compute_innovation_matrix(self):
        """ compute the innovation matrix for association with mahalonobis distance
        """
        return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R
    
    def sync_time_stamp(self, time_stamp):
        self.time_stamp = time_stamp
        return