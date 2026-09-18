from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2

import math
import cv2
import cv_bridge

DEFAULT_BAG_PATH = "/home/user/bagfiles/rosbag2_time_syn_experiment"
TF_topic = "/tf"
RGB_topic = "/camera/color/image_raw/compressed"
# Create a typestore and get the string class.
typestore = get_typestore(Stores.LATEST)

block1_pos: list = None
block2_pos: list = None
block_dist = float('inf')

bridge = cv_bridge.CvBridge()

# Create reader instance and open for reading.
with Reader(DEFAULT_BAG_PATH) as reader:
    # Topic and msgtype information is available on .connections list.
    for connection in reader.connections:
        print(connection.topic, connection.msgtype)

    # Iterate over messages.
    for connection, timestamp, rawdata in reader.messages():
        # if connection.topic == '/tf':
        #     msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        #     print(msg.header.frame_id)
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        ros_timestamp = timestamp * 1e-9
        print(f"------------------------------------------\n{connection.topic} {connection.msgtype} at {ros_timestamp}")
        if connection.topic == TF_topic:
            msg: TFMessage = msg
            for tfstamped in msg.transforms:
                tfstamped: TransformStamped = tfstamped
                header = tfstamped.header
                child_frame = tfstamped.child_frame_id
                TFtimestamp = float(header.stamp.sec) + 1e-9 * float(header.stamp.nanosec)
                transform = tfstamped.transform
                print(f"ROS bag timestamp: {ros_timestamp}, TF timestamp: {TFtimestamp}")
                print(f"child_frame: {child_frame}, frame: {header.frame_id}")

                update_pos = False
                if header.frame_id == "map":
                    if child_frame == "block_1":
                        block1_pos = [transform.translation.x, transform.translation.y, transform.translation.z]
                        update_pos = True
                    if child_frame == "block_2":
                        block2_pos = [transform.translation.x, transform.translation.y, transform.translation.z]
                        update_pos = True

                if update_pos and block1_pos is not None and block2_pos is not None:
                    curr_dist = math.dist(block1_pos, block2_pos)
                    update_dist = curr_dist < block_dist
                    block_dist = min(block_dist, curr_dist)
                    if update_dist:
                        collision_ROS_timestamp = ros_timestamp
                        collision_TF_timestamp = TFtimestamp
                        collision_pos = (block1_pos, block2_pos)
                        print(f"!!!new lowest block distance: {block_dist}, block1 at {block1_pos}, block2 at {block2_pos}")
        
        if connection.topic == RGB_topic:
            msg: CompressedImage = msg
            header = msg.header
            rgb_timestamp = float(header.stamp.sec) + 1e-9 * float(header.stamp.nanosec)
            print(f"ROS timestamp {ros_timestamp}, RGB timestamp {rgb_timestamp}")

            bgr_img = bridge.compressed_imgmsg_to_cv2(msg)
            cv2.imshow("RGB Camera", bgr_img)
            cv2.waitKey(1000)



    # # The .messages() method accepts connection filters.
    # connections = [x for x in reader.connections if x.topic == '/imu_raw/Imu']
    # for connection, timestamp, rawdata in reader.messages(connections=connections):
    #     msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
    #     print(msg.header.frame_id)

print(f"****************************************************\nCollision based on TF:\
      \nROS timestamp {collision_ROS_timestamp}, TF timestamp {collision_TF_timestamp}\
      \nPositions: block_1 {collision_pos[0]}, block_2 {collision_pos[1]}\
      \nDistance: {block_dist}")

print(f"As for collision based on Image, you should check the images shown by cv2 and align them with timestamp")