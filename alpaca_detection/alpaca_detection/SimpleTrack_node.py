from rclpy.node import Node
import rclpy
import ros2_numpy as rnp
import numpy as np
from collections import deque
from alpaca_detection.SimpleTrack.mot_3d.mot import MOTModel
from alpaca_detection.SimpleTrack.mot_3d.frame_data import FrameData
from alpaca_detection.SimpleTrack.mot_3d.data_protos import BBox
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
import math
from tf2_geometry_msgs import do_transform_pose, do_transform_point
from geometry_msgs.msg import PointStamped, PoseStamped, Pose, PoseWithCovarianceStamped
from tf2_ros import TransformListener, Buffer, Duration


class CentroidTrack:
    def __init__(self, internal_id, bbox, stamp):
        self.internal_id = internal_id
        self.id = None
        self.bbox = bbox
        self.last_xy = np.array([bbox.x, bbox.y], dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.last_stamp = float(stamp)
        self.missed = 0
        self.hits = 1
        self.confirmed = False

    def predicted_xy(self, stamp):
        dt = max(0.0, min(0.5, float(stamp) - self.last_stamp))
        return self.last_xy + self.velocity * dt

    def update(self, bbox, stamp, velocity_alpha=0.3):
        stamp = float(stamp)
        xy = np.array([bbox.x, bbox.y], dtype=float)
        dt = max(1e-3, stamp - self.last_stamp)
        if dt < 1.0:
            measured_velocity = (xy - self.last_xy) / dt
            alpha = float(velocity_alpha)
            self.velocity = (1.0 - alpha) * self.velocity + alpha * measured_velocity
        else:
            self.velocity *= 0.25
        self.bbox = bbox
        self.last_xy = xy
        self.last_stamp = stamp
        self.missed = 0
        self.hits += 1

    def mark_missed(self):
        self.missed += 1


class CentroidTracker:
    """Pedestrian tracker using map-frame XY association."""

    def __init__(
        self,
        association_distance=1.4,
        duplicate_distance=0.45,
        max_missed=12,
        velocity_alpha=0.3,
        min_hits_to_birth=3,
        tentative_max_missed=1,
    ):
        self.association_distance = float(association_distance)
        self.duplicate_distance = float(duplicate_distance)
        self.max_missed = int(max_missed)
        self.velocity_alpha = float(velocity_alpha)
        self.min_hits_to_birth = max(1, int(min_hits_to_birth))
        self.tentative_max_missed = max(0, int(tentative_max_missed))
        self.tracks = []
        self.next_internal_id = 0
        self.next_public_id = 0

    def _dedupe(self, dets):
        ordered = sorted(dets, key=lambda b: float(b.s if b.s is not None else 0.0), reverse=True)
        kept = []
        for det in ordered:
            xy = np.array([det.x, det.y], dtype=float)
            if any(np.linalg.norm(xy - np.array([k.x, k.y], dtype=float)) < self.duplicate_distance for k in kept):
                continue
            kept.append(det)
        return kept

    def update(self, dets, stamp):
        dets = self._dedupe(dets)
        stamp = float(stamp)
        matched_track_idx = set()
        matched_det_idx = set()

        if self.tracks and dets:
            cost = np.zeros((len(self.tracks), len(dets)), dtype=float)
            for i, trk in enumerate(self.tracks):
                pred_xy = trk.predicted_xy(stamp)
                for j, det in enumerate(dets):
                    det_xy = np.array([det.x, det.y], dtype=float)
                    cost[i, j] = np.linalg.norm(pred_xy - det_xy)

            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] > self.association_distance:
                    continue
                self.tracks[i].update(dets[j], stamp, self.velocity_alpha)
                if self.tracks[i].hits >= self.min_hits_to_birth:
                    self.tracks[i].confirmed = True
                    if self.tracks[i].id is None:
                        self.tracks[i].id = self.next_public_id
                        self.next_public_id += 1
                matched_track_idx.add(int(i))
                matched_det_idx.add(int(j))

        for i, trk in enumerate(self.tracks):
            if i not in matched_track_idx:
                trk.mark_missed()

        kept_tracks = []
        for trk in self.tracks:
            missed_limit = self.max_missed if trk.confirmed else self.tentative_max_missed
            if trk.missed <= missed_limit:
                kept_tracks.append(trk)
        self.tracks = kept_tracks

        for j, det in enumerate(dets):
            if j in matched_det_idx:
                continue
            track = CentroidTrack(self.next_internal_id, det, stamp)
            if track.hits >= self.min_hits_to_birth:
                track.confirmed = True
                track.id = self.next_public_id
                self.next_public_id += 1
            self.tracks.append(track)
            self.next_internal_id += 1

        return [
            (trk.bbox, trk.id, "alive_1_0", "person")
            for trk in self.tracks
            if trk.confirmed and trk.missed == 0
        ]


class SimpleTrackNode(Node):
    def __init__ (self, configs=None, tracker_hz: float = 20.0):
        super().__init__('SimpleTrack_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pc_numpy = None
        self.declare_parameter('rear_blind_spot_start_deg', 150.0)
        self.declare_parameter('rear_blind_spot_end_deg', 180.0)
        rear_blind_spot_start = float(self.get_parameter('rear_blind_spot_start_deg').value)
        rear_blind_spot_end = float(self.get_parameter('rear_blind_spot_end_deg').value)
        self.world_frame = self.declare_parameter('world_frame', 'map').value
        rear_blind_spot_start = max(0.0, min(180.0, rear_blind_spot_start))
        rear_blind_spot_end = max(rear_blind_spot_start, min(180.0, rear_blind_spot_end))
        blind_spots = [(rear_blind_spot_start, rear_blind_spot_end), (-rear_blind_spot_end, -rear_blind_spot_start)]
        # provide sensible default configs when none provided
        if not configs:
            configs = {
                'running': {
                    'match_type': 'bipartite',
                    'score_threshold': 0.4,
                    # Tuned on the 2026-04-17 forward bags against an honest
                    # end-to-end objective that keeps unmatched tracker outputs
                    # as false positives.
                    'asso': 'center',
                    'asso_thres': {'giou': 1.5, 'iou': 0.9, 'm_dis': 4, 'euler':12, 'center': 0.75},
                    'motion_model': 'kf',
                    'max_age_since_update': 3,
                    'min_hits_to_birth': 1,
                    'covariance': {},
                    'blind_spots': blind_spots,
                },
                'redundancy': {
                    'mode': 'mm',
                    'det_score_threshold': {'giou': 0.1, 'iou': 0.1, 'euler': 0.1, 'm_dis': 0.4, 'center': 0.1},
                    'det_dist_threshold': {'giou': -0.5, 'iou': 0.1, 'euler': 4, 'm_dis': 12, 'center': 0.75},
                }
            }
        else:
            configs.setdefault('running', {})
            configs['running']['blind_spots'] = blind_spots
        self.mot_model = MOTModel(configs)
        self.declare_parameter('centroid_association_distance', 1.4)
        self.declare_parameter('centroid_duplicate_distance', 0.45)
        self.declare_parameter('centroid_max_missed', 12)
        self.declare_parameter('centroid_velocity_alpha', 0.3)
        self.declare_parameter('centroid_min_hits_to_birth', 3)
        self.declare_parameter('centroid_tentative_max_missed', 1)
        self.centroid_tracker = CentroidTracker(
            association_distance=float(self.get_parameter('centroid_association_distance').value),
            duplicate_distance=float(self.get_parameter('centroid_duplicate_distance').value),
            max_missed=int(self.get_parameter('centroid_max_missed').value),
            velocity_alpha=float(self.get_parameter('centroid_velocity_alpha').value),
            min_hits_to_birth=int(self.get_parameter('centroid_min_hits_to_birth').value),
            tentative_max_missed=int(self.get_parameter('centroid_tentative_max_missed').value),
        )
        self.detection_subscriber = self.create_subscription (Detection3DArray, '/detection_results', self.detection_cb, 10)
        self.tracker_marker_publisher = self.create_publisher(MarkerArray, '/SimpleTrack_trackers', 10)
        self.result_publisher = self.create_publisher(Detection3DArray, '/SimpleTrack_results', 10)
        self.robot_pose = None
        self._amcl_subscription = None
        self._has_robot_pose = False

        # keep track of marker ids we've published so we can delete ones that disappear
        self._published_tracker_ids = set()
        # queue incoming detection frames so the tracker step can run at a fixed rate
        self._frame_queue = deque()
        self._last_frame_stamp = None
        self._tracker_engaged = True
        self.declare_parameter('tracker_update_rate', tracker_hz)
        rate_param = float(self.get_parameter('tracker_update_rate').value)
        if rate_param <= 0.0:
            rate_param = 10.0
        self._tracker_period = 1.0 / rate_param
        self._tracker_timer = self.create_timer(self._tracker_period, self._tracker_step)
    
    def detection_cb (self,msg):
        dets = []
        det_time_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        det_types = []

        lookup_time = rclpy.time.Time.from_msg(msg.header.stamp)
        for det in msg.detections: 
            position = det.bbox.center.position
            size = det.bbox.size
            orientation = det.bbox.center.orientation
            score = det.results[0].hypothesis.score
            converted = self.convert_to_map_frame(position.x, position.y, orientation,  lookup_time, msg.header.frame_id)
            if converted is None:
                continue

            position.x, position.y, position.z, orientation= converted

            orientation = 2*math.atan2(orientation.z, orientation.w)

            detection = np.array ([position.x, position.y, position.z , orientation, size.x, size.y , size.z, score] )
            dets.append (detection)
            det_types.append ('person')

        aux_info = {'is_key_frame': True}
        self._frame_queue.append({
            'dets': dets,
            'det_types': det_types,
            'time_stamp': det_time_stamp,
            'aux_info': aux_info
        })

    def _tracker_step(self):
        """Run the tracker at a fixed cadence so it can predict between detections."""
        if not self._tracker_engaged:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if not self._frame_queue:
            return
        if self._last_frame_stamp is not None and self._frame_queue[0]['time_stamp'] > now:
            return
        frame = self._frame_queue.popleft()

        frame_data = FrameData(
            dets=frame['dets'],
            ego=np.eye(4),
            time_stamp=frame['time_stamp'],
            aux_info=frame['aux_info'],
            det_types=frame['det_types']
        )
        tracks = self.centroid_tracker.update(frame_data.dets, frame['time_stamp'])
        self._last_frame_stamp = frame['time_stamp']
        self.draw_tracks(tracks)
        self.publish_results(tracks)

    def _ensure_amcl_subscription(self):
        if self._amcl_subscription is not None:
            if self._amcl_topic_timer is not None:
                self._amcl_topic_timer.cancel()
                self._amcl_topic_timer = None
            return

        for topic_name, topic_types in self.get_topic_names_and_types():
            if topic_name == '/amcl_pose' and 'geometry_msgs/msg/PoseWithCovarianceStamped' in topic_types:
                self._amcl_subscription = self.create_subscription(
                    PoseWithCovarianceStamped,
                    '/amcl_pose',
                    self._amcl_pose_cb,
                    10
                )
                self.get_logger().info('Subscribed to /amcl_pose for robot pose updates.')
                if self._amcl_topic_timer is not None:
                    self._amcl_topic_timer.cancel()
                    self._amcl_topic_timer = None
                break

    def _amcl_pose_cb(self, msg: PoseWithCovarianceStamped):
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        self.robot_pose = pose_stamped
        if not self._has_robot_pose:
            self.get_logger().info('Received robot pose from /amcl_pose.')
            self._has_robot_pose = True

    def draw_tracks(self, tracks):
        tracker_markers = MarkerArray()
        current_ids = set()
        # Add/Update markers for currently tracked objects
        for track in tracks:
            tid = track[1]
            current_ids.add(tid)
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.ns = "trackers"
            marker.id = tid
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 300000000
            marker.scale.x = track[0].l
            marker.scale.y = track[0].w
            marker.scale.z = track[0].h
            marker.color.a = 0.5
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.position.x = track[0].x
            marker.pose.position.y = track[0].y
            marker.pose.position.z = track[0].z
            qz = math.sin(track[0].o / 2)
            qw = math.cos(track[0].o / 2)
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = qz
            marker.pose.orientation.w = qw
            tracker_markers.markers.append(marker)

            # Text label sits slightly above the bbox to display the tracker ID
            text_marker = Marker()
            text_marker.header.frame_id = self.world_frame
            text_marker.ns = "tracker_labels"
            text_marker.id = tid
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.lifetime.sec = 0
            text_marker.lifetime.nanosec = 300000000
            text_marker.scale.z = 0.75
            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.pose.position.x = track[0].x
            text_marker.pose.position.y = track[0].y
            text_marker.pose.position.z = track[0].z + track[0].h * 0.5 + 0.2
            text_marker.text = str(tid)
            tracker_markers.markers.append(text_marker)

        # Any previously published markers that are NOT in current_ids should be removed immediately
        removed_ids = self._published_tracker_ids - current_ids
        for rid in removed_ids:
            del_marker = Marker()
            del_marker.header.frame_id = self.world_frame
            del_marker.ns = "trackers"
            del_marker.id = rid
            del_marker.action = Marker.DELETE
            tracker_markers.markers.append(del_marker)

            del_text_marker = Marker()
            del_text_marker.header.frame_id = self.world_frame
            del_text_marker.ns = "tracker_labels"
            del_text_marker.id = rid
            del_text_marker.action = Marker.DELETE
            tracker_markers.markers.append(del_text_marker)

        # publish and update the published ids set
        self.tracker_marker_publisher.publish(tracker_markers)
        self._published_tracker_ids = current_ids
    
    def publish_results (self, tracks):
        detection_msg = Detection3DArray()
        detection_msg.header.frame_id = self.world_frame
        detection_msg.header.stamp = self.get_clock().now().to_msg()
        for track in tracks:
            bbox = track[0]
            tid = track[1]  
            score = bbox.s if bbox.s is not None else 0.0

            det_msg = Detection3D()
            det_msg.header.frame_id = self.world_frame
            det_msg.header.stamp = self.get_clock().now().to_msg()

            # Fill in bbox
            det_msg.bbox.center.position.x = bbox.x
            det_msg.bbox.center.position.y = bbox.y
            det_msg.bbox.center.position.z = bbox.z
            qz = math.sin(bbox.o / 2)
            qw = math.cos(bbox.o / 2)
            det_msg.bbox.center.orientation.x = 0.0
            det_msg.bbox.center.orientation.y = 0.0
            det_msg.bbox.center.orientation.z = qz
            det_msg.bbox.center.orientation.w = qw
            det_msg.bbox.size.x = bbox.l
            det_msg.bbox.size.y = bbox.w
            det_msg.bbox.size.z = bbox.h

            # Fill in result with track ID and score
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(tid)
            hypothesis.hypothesis.score = float(score)
            det_msg.results.append(hypothesis)

            detection_msg.detections.append(det_msg)
        
        self.result_publisher.publish(detection_msg)
        
    def is_in_blind_spot(self, state):
        """ Check if the given state (box) is in the robot's blind spot.
            state: BBox or object with x, y attributes.
        """
        x, y = state.x, state.y
        # Calculate angle in degrees
        angle = np.degrees(np.arctan2(y, x))
        
        blind_spots = self.mot_model.configs.get('running', {}).get(
            'blind_spots',
            [(150, 180), (-180, -150)],
        )
        
        for (start, end) in blind_spots:
            if start <= angle <= end:
                return True
    
    def convert_to_map_frame (self, x, y, orientation, lookup_time, frame_id):
        point = Pose()
        point.position.x = x
        point.position.y = y 
        point.position.z = float (0.0)
        point.orientation = orientation

        try: 
            transform = self.tf_buffer.lookup_transform(self.world_frame, frame_id, lookup_time, timeout=Duration(seconds=0.1))
            transformed_point = do_transform_pose(point, transform)
            return (
                float(transformed_point.position.x),
                float(transformed_point.position.y),
                0.0,
                transformed_point.orientation
            )
        
        except Exception as e:
            self.get_logger().warn (f'Failed to transform point from {frame_id} to map frame: {e}')
            return None


def main():
    rclpy.init()
    node = SimpleTrackNode(configs={})
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
