import os
from setuptools import setup, find_packages


packages = find_packages()
# Ensure that we don't pollute the global namespace.
for p in packages:
    assert p == 'pybullet_multigoal_gym' or p.startswith('pybullet_multigoal_gym.')

setup(name='pybullet-multigoal-gym',
      version='1.0.0',
      description='A migration of the OpenAI Gym multi-goal robotic environment based on pybullet',
      url='#',
      author='XintongYang',
      author_email='YangX66@cardiff.ac.uk',
      packages=packages,
      package_dir={'pybullet_multigoal_gym': 'pybullet_multigoal_gym'},
      package_data={'pybullet_multigoal_gym': [
          'assets/objects/*.urdf',
          'assets/objects/assembling_shape/*.urdf',
          'assets/objects/insertion/*.urdf',
          'assets/robots/*.urdf',
          'assets/robots/kuka/*.urdf',
          'assets/robots/kuka/meshes/iiwa14/collision/*.stl',
          'assets/robots/kuka/meshes/iiwa14/visual/*.stl',
          'assets/robots/kuka/meshes/robotiq_2f_85/collision/*.stl',
          'assets/robots/kuka/meshes/robotiq_2f_85/visual/*.dae',
      ]},
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent",
      ])
