from setuptools import find_packages, setup

package_name = 'data_collection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # register package with ament
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        # install your package.xml
        ('share/' + package_name, ['package.xml']),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Data‐collection, error‐evaluation and MPC nodes for harp_oneseg',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_reciever       = data_collection.data_reciever:main',
            'error_evaluation    = data_collection.error_evaluation:main',
            'mpc_regulator_control = data_collection.mpc_regulator_control:main',
        ],
    },
)
