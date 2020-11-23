from gym.envs.registration import register


# multigoal envs
register(
    id='MGKuka2ObjPyBulletEnv-v0',
    entry_point='pybullet_multigoal_gym.envs.kuka.2_obj:Kuka2ObjEnv',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

def get_list():
	envs = ['- ' + spec.id for spec in gym.pgym.envs.registry.all() if spec.id.find('Bullet') >= 0 or spec.id.find('MuJoCo') >= 0]
	return envs
