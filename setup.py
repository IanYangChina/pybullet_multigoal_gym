from setuptools import setup, find_packages


packages = find_packages()
# Ensure that we don't pollute the global namespace.
for p in packages:
    assert p == 'pybullet_multigoal_gym' or p.startswith('pybullet_multigoal_gym.')

setup(name='pybullet-multigoal-gym',
      version='0.0.1',
      description='A migration of the OpenAI Gym multi-goal robotic environment based on pybullet',
      url='#',
      author='XintongYang',
      author_email='YangX66@cardiff.ac.uk',
      packages=packages,
      package_dir={'pybullet_multigoal_gym': 'pybullet_multigoal_gym'},
      include_package_data=True,
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent",
      ],)
