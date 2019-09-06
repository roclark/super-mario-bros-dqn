from argparse import Action, ArgumentParser
from gym_super_mario_bros.actions import (COMPLEX_MOVEMENT,
                                          RIGHT_ONLY,
                                          SIMPLE_MOVEMENT)

from core.constants import (ACTION_SPACE,
                            BATCH_SIZE,
                            BETA_FRAMES,
                            BETA_START,
                            ENVIRONMENT,
                            EPSILON_START,
                            EPSILON_FINAL,
                            EPSILON_DECAY,
                            GAMMA,
                            INITIAL_LEARNING,
                            LEARNING_RATE,
                            MEMORY_CAPACITY,
                            NUM_EPISODES,
                            TARGET_UPDATE_FREQUENCY)
from core.helpers import Range


ACTION_SPACE_CHOICES = {
    'right-only': RIGHT_ONLY,
    'simple': SIMPLE_MOVEMENT,
    'complex': COMPLEX_MOVEMENT
}


class ActionSpace(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.choices.get(values, self.default))


def parse_args():
    parser = ArgumentParser(description='')
    parser.add_argument('--action-space', action=ActionSpace,
                        choices=ACTION_SPACE_CHOICES, help='Specify the '
                        'action space to use as given by gym-super-mario-bros.'
                        ' Refer to the README for more details on the various '
                        'choices. Default: %s' % ACTION_SPACE,
                        default=ACTION_SPACE_CHOICES[ACTION_SPACE])
    parser.add_argument('--batch-size', type=int, help='Specify the batch '
                        'size to use when updating the replay buffer. '
                        'Default: %s' % BATCH_SIZE, default=BATCH_SIZE)
    parser.add_argument('--beta-frames', type=int, help='The number of frames '
                        'to update the beta value before reaching the maximum '
                        'of 1.0. Default: %s' % BETA_FRAMES,
                        default=BETA_FRAMES)
    parser.add_argument('--beta-start', type=float, help='The initial value '
                        'of beta to be used in the prioritized replay. '
                        'Default: %s' % BETA_START, default=BETA_START)
    parser.add_argument('--buffer-capacity', type=int, help='The capacity to '
                        'use in the experience replay buffer. Default: %s'
                        % MEMORY_CAPACITY, default=MEMORY_CAPACITY)
    parser.add_argument('--environment', type=str, help='The OpenAI gym '
                        'environment to use. Default: %s' % ENVIRONMENT,
                        default=ENVIRONMENT)
    parser.add_argument('--epsilon-start', type=float, help='The initial '
                        'value for epsilon to be used in the epsilon-greedy '
                        'algorithm. Default: %s' % EPSILON_START,
                        choices=[Range(0.0, 1.0)], default=EPSILON_START,
                        metavar='EPSILON_START')
    parser.add_argument('--epsilon-final', type=float, help='The final value '
                        'for epislon to be used in the epsilon-greedy '
                        'algorithm. Default: %s' % EPSILON_FINAL,
                        choices=[Range(0.0, 1.0)], default=EPSILON_FINAL,
                        metavar='EPSILON_FINAL')
    parser.add_argument('--epsilon-decay', type=int, help='The decay factor '
                        'for epsilon in the epsilon-greedy algorithm. '
                        'Default: %s' % EPSILON_DECAY, default=EPSILON_DECAY)
    parser.add_argument('--force-cpu', action='store_true', help='By default, '
                        'the program will run on the first supported GPU '
                        'identified by the system, if applicable. If a '
                        'supported GPU is installed, but all computations are '
                        'desired to run on the CPU only, specify this '
                        'parameter to ignore the GPUs. All actions will run '
                        'on the CPU if no supported GPUs are found. Default: '
                        'False')
    parser.add_argument('--gamma', type=float, help='Specify the discount '
                        'factor, gamma, to use in the Q-table formula. '
                        'Default: %s' % GAMMA, choices=[Range(0.0, 1.0)],
                        default=GAMMA, metavar='GAMMA')
    parser.add_argument('--initial-learning', type=int, help='The number of '
                        'iterations to explore prior to updating the model '
                        'and begin the learning process. Default: %s'
                        % INITIAL_LEARNING, default=INITIAL_LEARNING)
    parser.add_argument('--learning-rate', type=int, help='The learning rate '
                        'to use for the optimizer. Default: %s'
                        % LEARNING_RATE, default=LEARNING_RATE)
    parser.add_argument('--num-episodes', type=int, help='The number of '
                        'episodes to run in the given environment. Default: '
                        '%s' % NUM_EPISODES, default=NUM_EPISODES)
    parser.add_argument('--render', action='store_true', help='Specify to '
                        'render a visualization in another window of the '
                        'learning process. Note that a Desktop Environment is '
                        'required for visualization. Rendering scenes will '
                        'lower the learning speed. Default: False')
    parser.add_argument('--target-update-frequency', type=int, help='Specify '
                        'the number of iterations to run prior to updating '
                        'target network with the primary network\'s weights. '
                        'Default: %s' % TARGET_UPDATE_FREQUENCY,
                        default=TARGET_UPDATE_FREQUENCY)
    parser.add_argument('--transfer', action='store_true', help='Transfer '
                        'model weights from a previously-trained model to new '
                        'models for faster learning and improved accuracy. '
                        'Default: False')
    return parser.parse_args()
