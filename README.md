# Super Mario Bros. DQN
A reinforcement learning project designed to learn and complete the original
Super Mario Bros. for the Nintendo Entertainment System using a Deep Q-Learning
model.

## Setting up the repository

### Creating a virtual environment
After cloning the repository, it is highly recommended to install a virtual
environment (such as `virtualenv`) or Anaconda to isolate the dependencies of
this project with other system dependencies.

To install `virtualenv`, simply run

```
pip install virtualenv
```

Once installed, a new virtual environment can be created by running

```
virtualenv env
```

This will create a virtual environment in the `env` directory in the current
working directory. To change the location and/or name of the environment
directory, change `env` to the desired path in the command above.

To enter the virtual environment, run

```
source env/bin/activate
```

You should see `(env)` at the beginning of the terminal prompt, indicating the
environment is active. Again, replace `env` with your desired directory name.

To get out of the environment, simply run

```
deactivate
```

### Installing Dependencies
While the virtual environment is active, install the required dependencies by
running

```
pip -r requirements.txt
```

This will install all of the dependencies at specific versions to ensure they
are compatible with one another.

## Training a model

To train a model, use the `train.py` script and specify any parameters that need
to be changed, such as the environment or epsilon decay factors. A list of the
default values for every parameters can be found by running

```
python train.py --help
```

If you desire to run with the default settings, execute the script directly with

```
python train.py
```

The script will train the default environment over a set number of episodes and
display the training progress after the conclusion of every episode. The updates
indicate the episode number, the reward for the current episode, the best reward
the model has achieved so far, a rolling average of the previous 100 episode
rewards, and the current value for epsilon.

Any time the model reaches a new best rolling average or a new high score, the
current model weights are saved as a `.dat` file with the environment's name
(such as `SuperMarioBros-1-1-v0.dat`). This saved model will overwrite any
existing model weight files for the same environment.

Once the new model is saved, the model will be tested against the requested
level to determine the overall performance. The test run is saved in the
`recording` directory and contains a MP4 file of the run to analyze the current
performance of the model.

**Note**: Currently, the testing module throws errors stating the environment is
already closed. Despite the errors being thrown, all of the functionality will
continue as expected and these can be safely ignored.

## Progress
The following table shows the current progress of the model on various levels
and the settings used to achieve the indicated performance:

| Level           | Version | Status  | Actions | GIF         |
|:----------------|:--------|:--------|:--------|:------------|
| **World 1-1**   | v0      | Optimal | Complex | ![][1-1]    |
| **World 1-2**   | v0      | Finish  | Simple  | ![][1-2]    |

[1-1]: media/smb-1-1-complete.gif
[1-2]: media/smb-1-2-complete.gif
