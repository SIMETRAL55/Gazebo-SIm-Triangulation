from setuptools import setup
import os
from glob import glob

package_name = 'thermal_3d_localization'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'thermal_detector_node = thermal_3d_localization.thermal_detector:main',
            'triangulation_node = thermal_3d_localization.triangulation_node:main',
            'visualization_node = thermal_3d_localization.visualization:main',
        ],
    },
    data_files=[
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*')
        ),
        
    ],
)
