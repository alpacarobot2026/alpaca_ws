from setuptools import find_packages, setup

package_name = 'alpaca_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',['launch/alpaca_bringup.launch.py']),
        ('share/' + package_name + '/launch',['launch/odom_2_tf_launch.py']),
        ('share/' + package_name + '/launch',['launch/lidar_tf_pub.launch.py']),
        ('share/' + package_name + '/launch',['launch/rtabmap_lidar.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anonymous',
    maintainer_email='anonymous@example.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ 'odom_to_tf = ranger_ctrl.odom_2_tf:main',
                            'lidar_tf_pub = ranger_ctrl.lidar_tf_pub:main',
                            'pointcloud_filter = ranger_ctrl.filter_pointcloud:main',
        ],
    },
)
