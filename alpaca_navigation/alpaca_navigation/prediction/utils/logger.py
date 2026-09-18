from loguru import logger
import sys

from rclpy.node import Node
# from skeleton_interfaces.msg import MultiHumanSkeleton, HumanSkeleton

logger.remove(0)
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <cyan>{file}:{line}</cyan> - <level>{message}</level>",
)



def get_hst_infer_latency(node: Node, msg):
    
    msg_timestamp = msg.header.stamp
    node_timestamp = node.get_clock().now().to_msg()

    msg_time = msg_timestamp.sec + msg_timestamp.nanosec * 1e-9
    node_time = node_timestamp.sec + node_timestamp.nanosec * 1e-9
    logger.info(f"\nreceive img: {msg_time}, current: {node_time} \
                  \nlatency    : {node_time - msg_time}")