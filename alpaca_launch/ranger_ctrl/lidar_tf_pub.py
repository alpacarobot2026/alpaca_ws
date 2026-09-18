
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class LidarTFPublisher(Node):
    def __init__(self):
        super().__init__('lidar_tf_broadcaster')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )

        self.declare_parameter('use_zed_odom', False)
        self.use_zed_odom = self.get_parameter('use_zed_odom').value
        self.br = TransformBroadcaster(self)
        self.timer = self.create_timer(0.02, self.timer_cb)

        self.get_logger().info("Lidar TF Publisher Node has been started.")

    def timer_cb(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'velodyne'
        t.transform.translation.x = 0.307 #0.2
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.535 #0.385 #0.73
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.br.sendTransform(t)

        if not self.use_zed_odom:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'base_link'
            t.child_frame_id = 'zed_camera_link'
            t.transform.translation.x = 0.3
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.9
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            self.br.sendTransform(t)


def main():
    rclpy.init()
    node = LidarTFPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
