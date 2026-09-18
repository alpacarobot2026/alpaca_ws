#!/usr/bin/env python3
import math
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data,QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2


class PointCloudFilter(Node):
    def __init__(self):
        super().__init__("pointcloud_filter")

        # Parameters so you can override from launch if needed
        self.declare_parameter("input_topic", "/velodyne_points")
        self.declare_parameter("output_topic", "/filtered_clouds")

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value

        # RELIABLE here made velodyne_transform_node's publisher block on
        # this node's ACK whenever the per-point Python filter loop fell
        # behind, stalling /velodyne_points for other subscribers too.
        # BEST_EFFORT decouples us: a slow filter just drops old scans
        # instead of backpressuring the sensor driver.
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        self.sub = self.create_subscription(
            PointCloud2,
            input_topic,
            self.cloud_cb,
            qos_profile,
        )

        self.pub = self.create_publisher(
            PointCloud2,
            output_topic,
            qos_profile,
        )

        self.get_logger().info(
            f"Filtering point clouds: '{input_topic}' -> '{output_topic}'"
        )

        self._field_names: List[str] = []

    def cloud_cb(self, msg: PointCloud2):
        # Capture field names once to index x/y/z quickly
        if not self._field_names:
            self._field_names = [f.name for f in msg.fields]
            # x/y/z indices (fallback to 0/1/2 if missing for some reason)
            try:
                self._ix = self._field_names.index("x")
                self._iy = self._field_names.index("y")
                self._iz = self._field_names.index("z")
            except ValueError:
                self.get_logger().error("Input cloud missing x/y/z fields.")
                return

        # Read all fields; skip_nans removes NaNs but not ±Inf
        gen = pc2.read_points(
            msg,
            field_names=self._field_names,
            skip_nans=True,
        )

        filtered_points = []
        for p in gen:
            x, y, z = p[self._ix], p[self._iy], p[self._iz]

            # Drop non-finite values (Inf/-Inf) and all-zero points (optional common cleanup)
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            if x == 0.0 and y == 0.0 and z == 0.0:
                continue

            filtered_points.append(p)

        # Build filtered cloud, preserve original fields and header
        out = pc2.create_cloud(
            header=msg.header,
            fields=msg.fields,           # keep original fields structure
            points=filtered_points
        )
        # When we filter, the cloud becomes unorganized (height=1); that's fine for most pipelines
        out.is_dense = True            # points were checked for validity

        self.pub.publish(out)


def main():
    rclpy.init()
    node = PointCloudFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
