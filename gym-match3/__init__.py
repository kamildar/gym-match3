from gym.envs.registration import register

register(
    id='match3-v0',
    entry_point='gym_match3.envs:Match3Env',
)
