import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

def normalize_quat(x, y, z, w):
    q = np.array([x, y, z, w], dtype=float)
    if not np.all(np.isfinite(q)):
        return None
    n = np.linalg.norm(q)
    if n < 1e-12:
        # Zero-norm: return identity (safe fallback)
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n

class OdomToTF(Node):
    def __init__(self):
        super().__init__('odom_to_tf_broadcaster')

        # Parameters
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('odom_frame', '')          # override parent frame if needed
        self.declare_parameter('base_frame', 'base_link') # override child if msg.child_frame_id empty
        self.declare_parameter('throttle_warn_sec', 2.0)  # throttle invalid TF warnings

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )

        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.sub = self.create_subscription(Odometry, odom_topic, self._odom_cb, qos)
        self.br = TransformBroadcaster(self)

        self._last_warn_time = self.get_clock().now()  # for throttling
        self.get_logger().info(f'Starting odom_to_tf: subscribing {odom_topic}')

    def _warn_throttled(self, msg):
        throttle = self.get_parameter('throttle_warn_sec').get_parameter_value().double_value
        now = self.get_clock().now()
        if (now - self._last_warn_time).nanoseconds * 1e-9 >= throttle:
            self.get_logger().warn(msg)
            self._last_warn_time = now

    def _odom_cb(self, msg: Odometry):
        parent = self.get_parameter('odom_frame').get_parameter_value().string_value or (msg.header.frame_id or 'odom')
        child  = msg.child_frame_id or self.get_parameter('base_frame').get_parameter_value().string_value

        if not parent or not child:
            self._warn_throttled('Skipping TF: empty parent/child frame id')
            return
        if parent == child:
            self._warn_throttled(f'Skipping TF: parent==child ({parent})')
            return

        # Normalize quat
        q = msg.pose.pose.orientation
        qn = normalize_quat(q.x, q.y, q.z, q.w)
        if qn is None or not np.all(np.isfinite(qn)):
            self._warn_throttled('Skipping TF: non-finite quaternion from /odom')
            return

        t = TransformStamped()
        t.header.stamp = msg.header.stamp          # use odom timestamp
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation.x = float(qn[0])
        t.transform.rotation.y = float(qn[1])
        t.transform.rotation.z = float(qn[2])
        t.transform.rotation.w = float(qn[3])

        self.br.sendTransform(t)

def main():
    rclpy.init()
    node = OdomToTF()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
