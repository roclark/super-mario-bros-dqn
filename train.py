from core.argparser import parse_args
from core.constants import PRETRAINED_MODELS
from core.helpers import (compute_td_loss,
                          initialize_models,
                          set_device,
                          update_beta,
                          update_epsilon)
from core.replay_buffer import PrioritizedBuffer
from core.train_information import TrainInformation
from core.wrappers import wrap_environment
from os.path import join
from shutil import copyfile, move
from test import test

from torch import save
from torch.optim import Adam


def update_graph(model, target_model, optimizer, replay_buffer, args, device,
                 info, beta):
    if len(replay_buffer) > args.initial_learning:
        if not info.index % args.target_update_frequency:
            target_model.load_state_dict(model.state_dict())
        optimizer.zero_grad()
        compute_td_loss(model, target_model, replay_buffer, args.gamma, device,
                        args.batch_size, beta)
        optimizer.step()


def test_new_model(model, environment, info, action_space):
    save(model.state_dict(), join(PRETRAINED_MODELS, '%s.dat' % environment))
    print('Testing model...')
    flag = test(environment, action_space, info.new_best_counter)
    if flag:
        copyfile(join(PRETRAINED_MODELS, '%s.dat' % environment),
                 'recording/run%s/%s.dat' % (info.new_best_counter,
                                             environment))


def complete_episode(model, environment, info, episode_reward, episode,
                     epsilon, stats, action_space):
    new_best = info.update_rewards(episode_reward)
    if new_best:
        print('New best average reward of %s! Saving model'
              % round(info.best_average, 3))
        test_new_model(model, environment, info, action_space)
    elif stats['flag_get']:
        info.update_best_counter()
        test_new_model(model, environment, info, action_space)
    print('Episode %s - Reward: %s, Best: %s, Average: %s '
          'Epsilon: %s' % (episode,
                           round(episode_reward, 3),
                           round(info.best_reward, 3),
                           round(info.average, 3),
                           round(epsilon, 4)))


def run_episode(env, model, target_model, optimizer, replay_buffer, args,
                device, info, episode):
    episode_reward = 0.0
    state = env.reset()

    while True:
        epsilon = update_epsilon(info.index, args)
        if len(replay_buffer) > args.batch_size:
            beta = update_beta(info.index, args)
        else:
            beta = args.beta_start
        action = model.act(state, epsilon, device)
        if args.render:
            env.render()
        next_state, reward, done, stats = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        info.update_index()
        update_graph(model, target_model, optimizer, replay_buffer, args,
                     device, info, beta)
        if done:
            complete_episode(model, args.environment, info, episode_reward,
                             episode, epsilon, stats, args.action_space)
            break


def train(env, model, target_model, optimizer, replay_buffer, args, device):
    info = TrainInformation()

    for episode in range(args.num_episodes):
        run_episode(env, model, target_model, optimizer, replay_buffer, args,
                    device, info, episode)


def main():
    args = parse_args()
    env = wrap_environment(args.environment, args.action_space)
    device = set_device(args.force_cpu)
    model, target_model = initialize_models(args.environment, env, device,
                                            args.transfer)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    replay_buffer = PrioritizedBuffer(args.buffer_capacity)
    train(env, model, target_model, optimizer, replay_buffer, args, device)
    env.close()


if __name__ == '__main__':
    main()
