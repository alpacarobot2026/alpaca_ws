from rclpy.node import Node
import rclpy
from lidar_det.detector import RPEA
from sensor_msgs.msg import PointCloud2
import ros2_numpy as rnp
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from builtin_interfaces.msg import Duration
import math
import os

class RPEADetectionNode(Node):
    def __init__ (self):
        super().__init__('RPEA_detection_node')
        self.pc_numpy = None
        self.detector = None
        self.declare_parameter('checkpoint_path', '')
        self.ckpt_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value
        if not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(
                f"RPEA checkpoint not found: '{self.ckpt_path}'. Set the 'checkpoint_path' parameter."
            )
        self.point_cloud_subscriber = self.create_subscription (PointCloud2, '/velodyne_points', self.point_cloud_callback, 10)
        self.initialize_detector(self.ckpt_path, use_gpu=True)
        self.marker_publisher = self.create_publisher(MarkerArray, '/RPEA_detections', 10)
        self.detection_publisher = self.create_publisher(Detection3DArray, '/detection_results', 10)
        self.current_cloud_stamp = None

        self.declare_parameter ('score_threshold', 0.9)
        self.score_threshold = self.get_parameter('score_threshold').get_parameter_value().double_value

    def initialize_detector (self, ckpt_path,use_gpu=True):
        self.detector = RPEA(ckpt_path, gpu=use_gpu)
        self.get_logger().info("RPEA detector initialized.")

    def point_cloud_callback (self, msg):
        # ros2_numpy returns an (N,3) array with columns [x,y,z].
        # The detector expects a (3, N) array (rows x,y,z).
        pc_arr = rnp.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=True)
        self.current_cloud_stamp = msg.header.stamp

        if pc_arr is None or pc_arr.size == 0:
            return

        xyz = pc_arr.T.astype(np.float32)

        max_points = 500000
        if xyz.shape[1] > max_points:
            idx = np.random.choice(xyz.shape[1], max_points, replace=False)
            xyz = xyz[:, idx]

        self.pc_numpy = xyz

        self.run_detection()

    def run_detection (self, score_threshold=None):
        if score_threshold is None:
            score_threshold = self.score_threshold

        if self.detector is None:
            self.initialize_detector(self.ckpt_path, use_gpu=True)

        if self.pc_numpy is None:
            self.get_logger().info("No point cloud data received yet.")
            return

        boxes, scores = self.detector(self.pc_numpy)

        # filter by score
        keep = scores >= score_threshold
        boxes_f = boxes[keep]
        scores_f = scores[keep]

        # sort descending by score
        if scores_f.size > 0:
            order = scores_f.argsort()[::-1]
            boxes_f = boxes_f[order]
            scores_f = scores_f[order]

        MarkerArray_msg = MarkerArray()
        for i in range(min(10, boxes_f.shape[0])):
            b = boxes_f[i]
            s = scores_f[i]
            marker = self.draw_detection(b, self.current_cloud_stamp)
            if marker is not None:
                marker.id = i
                MarkerArray_msg.markers.append(marker)

        self.marker_publisher.publish(MarkerArray_msg)
        self.publish_results(boxes_f, scores_f, self.current_cloud_stamp)

    def draw_detection (self, box, stamp=None):
        # box is [x, y, z, length, width, height, yaw]
        marker = None
        if np.linalg.norm ( [box[0],box[1]]) < 10:
            marker = Marker()
            marker.header.frame_id = 'velodyne'
            marker.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
            marker.ns = 'RPEA_detections'
            # Use a single CUBE per detection so viewers (Foxglove, RViz2) render it easily
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Ensure markers disappear shortly after they stop being published
            marker.lifetime = Duration(sec=0, nanosec=500000000)  # 0.5 seconds

            # sizes
            marker.scale.x = float(box[4])  # length
            marker.scale.y = float(box[3])  # width
            marker.scale.z = float(box[5])  # height

            # color (red) with alpha
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            # position
            marker.pose.position.x = float(box[0])
            marker.pose.position.y = float(box[1])
            marker.pose.position.z = float(box[2])

            # orientation from yaw (rotation around Z)
            yaw = float(box[6]) if len(box) > 6 else 0.0
            half_yaw = yaw * 0.5
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = math.sin(half_yaw)
            marker.pose.orientation.w = math.cos(half_yaw)

        return marker

    def publish_results (self, boxes, scores, stamp=None):
        detection_array_msg = Detection3DArray()
        detection_array_msg.header.frame_id = 'velodyne'
        detection_array_msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()

        for i in range(boxes.shape[0]):
            box = boxes[i]
            if np.linalg.norm ( [box[0],box[1]]) <= 10:
                score = scores[i]

                detection_msg = Detection3D()
                detection_msg.header.frame_id = 'velodyne'
                detection_msg.header.stamp = detection_array_msg.header.stamp

                # Fill in the bounding box information
                detection_msg.bbox.center.position.x = float(box[0])
                detection_msg.bbox.center.position.y = float(box[1])
                detection_msg.bbox.center.position.z = float(box[2])

                detection_msg.bbox.size.x = float(box[4])  # length
                detection_msg.bbox.size.y = float(box[3])  # width
                detection_msg.bbox.size.z = float(box[5])  # height

                # Orientation from yaw
                yaw = float(box[6]) if len(box) > 6 else 0.0
                half_yaw = yaw * 0.5
                detection_msg.bbox.center.orientation.x = 0.0
                detection_msg.bbox.center.orientation.y = 0.0
                detection_msg.bbox.center.orientation.z = math.sin(half_yaw)
                detection_msg.bbox.center.orientation.w = math.cos(half_yaw)

                # Add score as a hypothesis
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.score = float(score)
                detection_msg.results.append(hypothesis)
                detection_array_msg.detections.append(detection_msg)

        self.detection_publisher.publish(detection_array_msg)


def main():
    rclpy.init()
    node = RPEADetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()