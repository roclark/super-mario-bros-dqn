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

### Action spaces
This repository allows users to specify a custom set of actions that Mario can
use with various degrees of complexity. Choosing a simpler action space makes it
quicker and easier for Mario to learn, but prevents him from trying more complex
movements which can include entering pipes and making advanced jumps which might
be required to solve some levels. If Mario appears to struggle with a particular
level, try simplifying the action space to see if he makes further progress.

Currently, the following options are supported:

#### Right only
Mario can effectively only go right. This simplifies the training process, but
prevents Mario from trying more complex actions. The following buttons are
supported:
  * Nothing
  * Right
  * Right + A
  * Right + B
  * Right + A + B

#### Simple movement
In addition to moving right and running/jumping, Mario can now walk left and
jump in place. The following buttons are supported:
  * Nothing
  * Right
  * Right + A
  * Right + B
  * Right + A + B
  * A
  * Left

#### Complex movement
This action allows Mario to try nearly any of his possible actions from the
game. This option should be chosen by default for the most realistic exploration
of a level, but can increase the time and complexity of learning a level. This
is the only provided action space that allows Mario to enter vertically-oriented
pipes. The following buttons are supported:
  * Nothing
  * Right
  * Right + A
  * Right + B
  * Right + A + B
  * A
  * Left
  * Left + A
  * Left + B
  * Left + A + B
  * Down
  * Up

## Progress
The following table shows the current progress of the model on various levels
and the settings used to achieve the indicated performance:

| Level           | Version | Status   | Actions | GIF         |
|:----------------|:--------|:---------|:--------|:------------|
| **World 1-1**   | v0      | Optimal  | Complex | ![][1-1]    |
| **World 1-2**   | v0      | Optimal  | Simple  | ![][1-2]    |
| **World 1-3**   | v0      | Optimal  | Simple  | ![][1-3]    |
| **World 1-4**   | v0      | Optimal  | Simple  | ![][1-4]    |
| **World 2-1**   | N/A     | Untested | N/A     | N/A         |
| **World 2-2**   | N/A     | Untested | N/A     | N/A         |
| **World 2-3**   | v0      | Optimal  | Simple  | ![][2-3]    |
| **World 2-4**   | v0      | Optimal  | Simple  | ![][2-4]    |
| **World 3-1**   | N/A     | Untested | N/A     | N/A         |
| **World 3-2**   | v0      | Optimal  | Simple  | ![][3-2]    |
| **World 3-3**   | N/A     | Untested | N/A     | N/A         |
| **World 3-4**   | v0      | Optimal  | Simple  | ![][3-4]    |
| **World 4-1**   | N/A     | Untested | N/A     | N/A         |
| **World 4-2**   | N/A     | Untested | N/A     | N/A         |
| **World 4-3**   | v0      | Optimal  | Simple  | ![][4-3]    |
| **World 4-4**   | N/A     | Untested | N/A     | N/A         |
| **World 5-1**   | N/A     | Untested | N/A     | N/A         |
| **World 5-2**   | N/A     | Untested | N/A     | N/A         |
| **World 5-3**   | N/A     | Untested | N/A     | N/A         |
| **World 5-4**   | v0      | Optimal  | Simple  | ![][5-4]    |
| **World 6-1**   | v0      | Optimal  | Simple  | ![][6-1]    |
| **World 6-2**   | N/A     | Untested | N/A     | N/A         |
| **World 6-3**   | N/A     | Untested | N/A     | N/A         |
| **World 6-4**   | v0      | Optimal  | Simple  | ![][6-4]    |
| **World 7-1**   | N/A     | Untested | N/A     | N/A         |
| **World 7-2**   | N/A     | Untested | N/A     | N/A         |
| **World 7-3**   | v0      | Optimal  | Simple  | ![][7-3]    |

[1-1]: media/smb-1-1-complete.gif
[1-2]: media/smb-1-2-complete.gif
[1-3]: media/smb-1-3-complete.gif
[1-4]: media/smb-1-4-complete.gif
[2-3]: media/smb-2-3-complete.gif
[2-4]: media/smb-2-4-complete.gif
[3-2]: media/smb-3-2-complete.gif
[3-4]: media/smb-3-4-complete.gif
[4-3]: media/smb-4-3-complete.gif
[5-4]: media/smb-5-4-complete.gif
[6-1]: media/smb-6-1-complete.gif
[6-4]: media/smb-6-4-complete.gif
[7-3]: media/smb-7-3-complete.gif

### Legend
The following is a legend of values to decipher the table above.

#### Level
The level as displayed in the actual game. **World 1-1** referes to World 1,
level 1 of Super Mario Bros. (ie. the first level).

#### Version
The version of the environment that was tested. See the Environments section of
[gym-super-mario-bros' README](https://github.com/Kautenja/gym-super-mario-bros/blob/master/README.md#environments)
for examples of the various environment versions.

#### Status
The current status of training for the indicated level. The status can take on
the following values:
  * **Untested**: No attempts or progress has been made on training for the
given level yet.
  * **Training**: Training has begun for the indicated level, but Mario has not
yet completed the level. If a model is provided, it will correspond to the most
recent training pass achieved, and not necessarily the best run so far.
  * **Satisfactory**: Mario can successfully complete the level, but is
currently unable to do so in an optimal manner for any reason, including
standing in place, losing health, not making forward progress, or others.
  * **Optimal**: Mario has trained enough that he can beat the level at
near-optimal performance. This does not necessarily mean the run is perfect, but
he can complete the level with only a couple minor interruptions at most. In
this state, further progress will likely not be made.

#### Actions
The action-space that Mario has been trained to use. See "Action spaces" above
for more details on the various action spaces.

#### GIF
An animated GIF of the run that corresponds to the saved model provided in the
repository.
