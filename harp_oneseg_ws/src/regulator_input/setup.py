from setuptools import find_packages, setup

package_name = 'regulator_input'

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
    description='Publishes regulator set-points at 10 Hz',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            # matches src/regulator_input/regulator_input/regulator_control.py
            'regulator_control = regulator_input.regulator_control:main',
        ],
    },
)
