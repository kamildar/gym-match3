# gym-match3

`gym-match3` is an environment for reinforcement learning purposes.
It replicates Match-3 board and allows to create your own levels.
    

## Getting started
### Installing
```bash
git clone https://github.com/kamildar/gym-match3.git
cd gym-match3
pip install -e .
```


### Running the tests
For testing the environment after installation run
following command in `tests` directory of a project:

```bash
python -m unittest test_game
```

### Usage
Here’s a bare minimum example of getting something running.

```python
from gym_match3.envs import Match3Env

env = Match3Env()
obs, reward, done, info = env.step(0) 
```

For more information on `gym` interface visit [gym documentation](https://gym.openai.com/docs/)


### Levels
Current number of levels is 30, but environment supports custom levels.
Example:
```python
from gym_match3.envs import Match3Env
from gym_match3.envs.levels import LEVELS #  default levels
from gym_match3.envs.levels import Match3Levels, Level


custom_level = Level(h=9, w=9, n_shape=6, board=[
    [-1, -1, -1, -1,  0, -1, -1, -1, -1],
    [-1, -1, -1,  0,  0,  0, -1, -1, -1],
    [-1, -1,  0,  0,  0,  0,  0, -1, -1],
    [-1,  0,  0,  0,  0,  0,  0,  0, -1],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [-1,  0,  0,  0,  0,  0,  0,  0, -1],
    [-1, -1,  0,  0,  0,  0,  0, -1, -1],
    [-1, -1, -1,  0,  0,  0, -1, -1, -1],
    [-1, -1, -1, -1,  0, -1, -1, -1, -1],
])

# create an instance with extended levels
custom_m3_levels = Match3Levels(levels=LEVELS + [custom_level]) 
env = Match3Env(levels=custom_m3_levels) 
```
