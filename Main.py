import numpy as np
import random
import cv2
import tensorflow as tf
import NeuralNets as NN
import PPO
import Datapreprocessing as dpp
import Common_constants as CC
import Auxiliars as AUX

tf.keras.backend.set_floatx('float32')

env = CC.env  # 這裡假設你已經在 Common_constants.py 定義好 env
obs_shape = CC.obs_shape
num_actions = CC.num_actions
env_name = CC.env_name
start_t = CC.start_t
save_path = CC.save_path
max_steps = CC.max_steps
base_learning_rate = CC.base_learning_rate
log_dir = CC.log_dir
SMALL_NUM = CC.SMALL_NUM
load_model = CC.load_model
optim_epochs = CC.optim_epochs
batch_size = CC.batch_size
horizon = CC.horizon

def train(load=True):
    writer_sum = tf.summary.create_file_writer(log_dir)
    t = start_t
    last_save = 0

    value_network = NN.value_nn()
    policy_network = NN.policy_nn()

    if load:
        AUX.loader([value_network, policy_network], save_path)

    obs_horizon = []
    act_horizon = []
    policy_horizon = []
    adv_est_horizon = []
    val_est_horizon = []

    state = env.reset()
    episode_rewards = []
    episode_x = []

    while t <= max_steps:
        learning_rate = base_learning_rate
        adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-5)
        model_grads = PPO.gradients(adam)

        # 收集 horizon 步資料
        for _ in range(horizon):
            state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
            policy_t = policy_network(state_tensor).numpy()[0]
            action = np.random.choice(num_actions, p=policy_t)
            next_state, reward, done, info = env.step(action)
            value = value_network(state_tensor).numpy()[0][0]

            obs_horizon.append(state)
            act_horizon.append(action)
            policy_horizon.append(policy_t)
            val_est_horizon.append(value)
            adv_est_horizon.append(reward)  # 這裡僅為範例，實際應計算GAE

            episode_rewards.append(reward)
            episode_x.append(info.get('x', 0) if isinstance(info, dict) else 0)

            state = next_state
            t += 1
            if done:
                state = env.reset()

        # Advantage normalization
        adv_est_horizon = np.array(adv_est_horizon)
        adv_est_horizon = (adv_est_horizon - np.mean(adv_est_horizon)) / (np.std(adv_est_horizon) + SMALL_NUM)

        num_samples = len(obs_horizon)
        indices = list(range(num_samples))

        for e in range(optim_epochs):
            random.shuffle(indices)
            ii = 0
            while ii < num_samples:
                obs_batch = []
                act_batch = []
                policy_batch = []
                adv_batch = []
                value_sample_batch = []

                for b in range(batch_size):
                    if ii >= num_samples:
                        break
                    index = indices[ii]
                    obs_batch.append(np.squeeze(obs_horizon[index], axis=0))
                    act_batch.append(act_horizon[index])
                    policy_batch.append(policy_horizon[index])
                    adv_batch.append(adv_est_horizon[index])
                    value_sample_batch.append(val_est_horizon[index])
                    ii += 1

                obs_batch = tf.convert_to_tensor(np.asarray(obs_batch), dtype=tf.float32)
                act_batch = tf.convert_to_tensor(np.asarray(act_batch), dtype=tf.uint8)
                policy_batch = tf.convert_to_tensor(np.asarray(policy_batch), dtype=tf.float32)
                adv_batch = tf.convert_to_tensor(np.asarray(adv_batch), dtype=tf.float32)
                value_sample_batch = tf.convert_to_tensor(np.asarray(value_sample_batch), dtype=tf.float32)

                entropy_loss, clip_loss, value_loss, total_loss = model_grads(
                    1.0,  # alpha_anneal(t) 可視需要調整
                    policy_network, value_network, obs_batch,
                    act_batch, adv_batch, policy_batch, value_sample_batch
                )

        if t - last_save > 1000:
            AUX.saver([value_network, policy_network], save_path)
            last_save = t

        if len(episode_rewards) >= 10:
            print("T: %d" % (t,))
            print("AVG Reward: %f" % (np.mean(episode_rewards),))
            print("MIN Reward: %f" % (np.amin(episode_rewards),))
            print("MAX Reward: %f" % (np.amax(episode_rewards),))
            print("AVG X: %f" % (np.mean(episode_x),))
            print("MIN X: %f" % (np.min(episode_x),))
            print("MAX X: %f" % (np.max(episode_x),))
            AUX.sum_writer(writer_sum, np.mean(episode_rewards), t, 'Avg_Reward')
            AUX.sum_writer(writer_sum, np.mean(episode_x), t, 'Avg_X')
            AUX.sum_writer(writer_sum, np.max(episode_x), t, 'Max_X')
            episode_rewards = []
            episode_x = []

def test(episodes):
    assert load_model == True

    policy_network = NN.policy_nn()
    AUX.loader_test(policy_network, save_path)
    scores = []

    for e in range(episodes):
        state = env.reset()
        state = np.expand_dims(state, axis=0)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        score = 0
        video_frames = []
        done = False

        while not done:
            video_frames.append(cv2.cvtColor(env.render(mode='rgb_array'), cv2.COLOR_RGB2BGR))
            policy_t = policy_network(state).numpy()[0]
            action_t = np.argmax(policy_t)
            next_state, reward, done, info = env.step(action_t)
            state = np.expand_dims(next_state, axis=0)
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            score += reward

        video_name = 'test_' + str(e) + '.mp4'
        _, height, width, _ = np.shape(video_frames)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(video_name, fourcc, 5, (width, height))
        for image in video_frames:
            video.write(image)

        cv2.destroyAllWindows()
        video.release()
        print('Test #%s , Score: %0.1f' % (e, score))
        scores.append(score)

    print('Average reward: %0.2f of %s episodes' % (np.mean(scores), episodes))