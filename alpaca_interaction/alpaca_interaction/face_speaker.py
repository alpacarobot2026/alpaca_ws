import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Twist


class FaceSpeaker(Node):
    def __init__(self):
        super().__init__('face_speaker')

        self.declare_parameter('image_width', 2208)
        self.declare_parameter('kp', 0.0007)
        self.declare_parameter('max_angular_z', 0.5)
        # Publishes to a twist_mux input, not /cmd_vel directly. twist_mux
        # arbitrates against MPPI (see alpaca_navigation/config/twist_mux.yaml).
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_face')

        self.image_width = float(self.get_parameter('image_width').value)
        self.kp = float(self.get_parameter('kp').value)
        self.max_angular_z = float(self.get_parameter('max_angular_z').value)
        self.image_center_x = self.image_width / 2.0
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)

        # id of last confirmed speaker; held until a new speaker appears
        self.sticky_id: str | None = None
        # latest snapshot of all tracked faces: {id -> center_x}
        self.latest_all_dets: dict[str, float] = {}

        self.bbox_sub = self.create_subscription(
            Detection2DArray,
            '/talknce/active_speaker_bbox',
            self._active_bbox_callback,
            10,
        )
        self.all_dets_sub = self.create_subscription(
            Detection2DArray,
            '/talknce/all_detections',
            self._all_dets_callback,
            10,
        )
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self._enabled: bool = True  # controlled by set_enabled()
        self._last_twist = Twist()       # last tracking command (held across n=0 gaps)
        self._last_tracking_time = None  # time of last useful tracking update

        self.get_logger().info(
            f'FaceSpeaker ready | image_width={self.image_width} '
            f'kp={self.kp} max_angular_z={self.max_angular_z} '
            f'cmd_vel_topic={self.cmd_vel_topic}'
        )

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            self._last_twist = Twist()
            self._last_tracking_time = None
            self.cmd_vel_pub.publish(Twist())
        self.get_logger().info(f'FaceSpeaker {"ENABLED" if enabled else "DISABLED"}')

    def _all_dets_callback(self, msg: Detection2DArray):
        # Always update — needed for sticky lookup when re-enabled after navigation.
        self.latest_all_dets = {
            d.id: d.bbox.center.position.x for d in msg.detections
        }

    def _active_bbox_callback(self, msg: Detection2DArray):
        self.active_bbox_msg = msg
        # Always track sticky_id so it's current when we re-enable.
        if msg.detections and len(msg.detections) == 1:
            self.sticky_id = msg.detections[0].id
        if self._enabled:
            self.face_speaker()

    
    def face_speaker(self):
        n = len(self.active_bbox_msg.detections)

        if n == 1:
            self.sticky_id = self.active_bbox_msg.detections[0].id
            speaker_x = self.active_bbox_msg.detections[0].bbox.center.position.x
        elif n == 0 and self.sticky_id is not None and self.sticky_id in self.latest_all_dets:
            speaker_x = self.latest_all_dets[self.sticky_id]
        else:
            return  # nothing useful to say; MPC zeros hold the robot

        self._last_twist = Twist()
        self._last_twist.angular.z = self._p(speaker_x)
        self.cmd_vel_pub.publish(self._last_twist)

    def _p(self, speaker_x: float) -> float:
        error = speaker_x - self.image_center_x   # + = speaker right of center
        raw = -self.kp * error                     # negative = turn right
        return max(-self.max_angular_z, min(self.max_angular_z, raw))


def main(args=None):
    rclpy.init(args=args)
    node = FaceSpeaker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()