#!/usr/bin/env python3
"""
pcd_publisher.py

Simple ROS2 node that loads a .pcd file and continuously publishes it as
sensor_msgs/PointCloud2. It tries to use Open3D for robust PCD loading and
falls back to a minimal ASCII PCD parser for basic ASCII files.

Usage:
  python3 pcd_publisher.py --pcd /path/to/file.pcd --topic /points --rate 10 --frame-id lidar --loop

Notes:
 - Requires ROS2 Python packages (rclpy, sensor_msgs, sensor_msgs_py).
 - For binary PCD files, install Open3D (pip install open3d) or convert to ASCII.

"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2

try:
    # preferred helper for building PointCloud2
    from sensor_msgs_py import point_cloud2 as pc2
except Exception:
    try:
        # fallback (older ROS2 installs may expose sensor_msgs.point_cloud2)
        from sensor_msgs import point_cloud2 as pc2
    except Exception:
        pc2 = None


def load_pcd_with_open3d(path: str) -> np.ndarray:
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError("open3d not available") from e

    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if pts.ndim == 1:
        pts = pts.reshape((-1, 3))
    return pts.astype(np.float32)


def load_ascii_pcd(path: str) -> np.ndarray:
    """A minimal ASCII PCD parser that extracts X Y Z fields.

    This parser handles common simple ASCII PCD files where the header
    ends with 'END_HEADER' and each following line contains coordinates.
    It ignores non-numeric trailing columns and comments.
    """
    pts = []
    with open(path, 'r', errors='ignore') as f:
        header_ended = False
        for line in f:
            line = line.strip()
            if not header_ended:
                if line.startswith('DATA'):
                    # expect 'DATA ascii' or 'DATA binary' etc.
                    if 'ascii' not in line.lower():
                        raise RuntimeError('Only ASCII PCD supported by fallback parser (install open3d for binary PCDs)')
                if line == 'END_HEADER':
                    header_ended = True
                continue
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
            pts.append((x, y, z))

    if not pts:
        raise RuntimeError('No points parsed from ASCII PCD')
    return np.asarray(pts, dtype=np.float32)


def load_pcd(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'PCD file not found: {path}')

    # Try open3d first (handles many formats incl. binary)
    try:
        return load_pcd_with_open3d(path)
    except Exception:
        # fallback to ASCII parser
        return load_ascii_pcd(path)


class PCDPublisher(Node):
    def __init__(self, pcd_path: str, topic: str, frame_id: str, rate_hz: float, loop: bool):
        super().__init__('pcd_publisher')
        self.pcd_path = pcd_path
        self.topic = topic
        self.frame_id = frame_id
        self.rate = rate_hz
        self.loop = loop

        self.pub = self.create_publisher(PointCloud2, self.topic, 10)

        self.get_logger().info(f'Loading PCD from: {self.pcd_path}')
        self.points = load_pcd(self.pcd_path)
        self.get_logger().info(f'Loaded {self.points.shape[0]} points')

        if pc2 is None:
            self.get_logger().error('No point_cloud2 helper available (sensor_msgs_py or sensor_msgs). Cannot publish.')
            raise RuntimeError('point_cloud2 helper missing')

        # prepare message template
        self.header = Header()
        self.header.frame_id = self.frame_id

        # Start periodic publishing
        period = 1.0 / max(0.001, float(self.rate))
        self.timer = self.create_timer(period, self.timer_cb)

    def timer_cb(self):
        # Build PointCloud2 message and publish
        now = self.get_clock().now().to_msg()
        self.header.stamp = now

        # pc2.create_cloud_xyz32 takes header and iterable of (x,y,z)
        try:
            msg = pc2.create_cloud_xyz32(self.header, (tuple(p) for p in self.points))
        except TypeError:
            # Some older helpers expect a list, convert
            msg = pc2.create_cloud_xyz32(self.header, [tuple(p) for p in self.points])

        self.pub.publish(msg)
        # Optionally loop/pause behavior
        if not self.loop:
            # If not looping, publish once and shutdown after a short delay to let message go out
            self.get_logger().info('Published once (loop disabled), shutting down')
            # Allow a small sleep then shutdown
            time.sleep(0.1)
            rclpy.shutdown()


def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Publish a .pcd file continuously as sensor_msgs/PointCloud2')
    parser.add_argument('--pcd', required=True, help='Path to .pcd file')
    parser.add_argument('--topic', default='/points', help='ROS2 topic to publish (default: /points)')
    parser.add_argument('--frame-id', default='lidar', help='Frame id to use in header')
    parser.add_argument('--rate', type=float, default=10.0, help='Publish rate in Hz')
    parser.add_argument('--loop', action='store_true', help='Continuously loop publishing the pointcloud (default: false)')

    args = parser.parse_args(argv)

    rclpy.init()
    try:
        node = PCDPublisher(args.pcd, args.topic, args.frame_id, args.rate, args.loop)
        rclpy.spin(node)
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        return 1
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
