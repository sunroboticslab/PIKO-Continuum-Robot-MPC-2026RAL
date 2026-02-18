from setuptools import find_packages, setup

package_name = 'harp_oneseg_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament index registration
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # install package manifest
        ('share/' + package_name, ['package.xml']),
        # install launch files
        ('share/' + package_name + '/launch', [
            'launch/harp_launch.py',
            'launch/mpc_launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Launch file package for HARP soft robot',
    license='Apache-2.0',
)
