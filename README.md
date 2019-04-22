# gym-match3

`gym-match3` is an environment for reinforcement learning puroses (`link to smth`).
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
tbc
