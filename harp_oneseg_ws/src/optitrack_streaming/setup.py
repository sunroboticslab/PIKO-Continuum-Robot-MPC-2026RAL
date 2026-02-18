from setuptools import find_packages, setup

package_name = 'optitrack_streaming'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='UDP receiver for OptiTrack data',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            # matches src/optitrack_streaming/optitrack_streaming/udp_optitrack_publisher.py
            'udp_optitrack_publisher = optitrack_streaming.udp_optitrack_publisher:main',
        ],
    },
)
