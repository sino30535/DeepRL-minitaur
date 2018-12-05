import numpy as np
import tensorflow as tf
import tflearn

from pybullet_envs.bullet import minitaur_gym_env
from pybullet_envs.bullet import minitaur_env_randomizer
from replay_buffer import ReplayBuffer

MAX_EPISODES = 1500
MAX_EP_STEPS = 1000
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.001      # soft replacement
MEMORY_CAPACITY = 1000000
BATCH_SIZE = 100


class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh which range is between (-1, 1)
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 256)
        net = tflearn.activations.relu(net)
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        return inputs, out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)
             + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 256)
        net = tflearn.activations.relu(net)

        t1 = tflearn.fully_connected(net, 256)
        t2 = tflearn.fully_connected(action, 256)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        out = tflearn.fully_connected(net, 1)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


def train(sess, env, actor, critic, actor_noise, load_weights=False):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()
    if load_weights:
        saver = tf.train.Saver()
        saver.restore(sess, "./model/model.ckpt")
    else:
        sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./summary', sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(MEMORY_CAPACITY)
    decay = 0.999
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        ep_ave_max_q = 0
        for j in range(MAX_EP_STEPS):

            # Added exploration noise
            action = np.squeeze(actor.predict(np.expand_dims(s, axis=0))) + actor_noise()*decay
            action = np.clip(action, -1, 1)
            a = np.array([action[:2], action[:2], action[2:], action[2:]]).flatten()
            s2, r, done, _ = env.step(a)
            replay_buffer.add(s, action, r, s2)
            ep_reward += r
            s = s2

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > BATCH_SIZE:
                s_batch, action_batch, r_batch, s2_batch = replay_buffer.sample_batch(BATCH_SIZE)
                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(BATCH_SIZE):
                    y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, action_batch, np.reshape(y_i, (BATCH_SIZE, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

        decay *= 0.999

        summary_str = sess.run(summary_ops, feed_dict={
            summary_vars[0]: ep_reward,
            summary_vars[1]: ep_ave_max_q / float(j)
        })

        writer.add_summary(summary_str, i)
        writer.flush()

        print('| Reward: {:.7f} | Episode: {:d} | Qmax: {:.4f}'.format(float(ep_reward),
              i, (ep_ave_max_q / float(j))))

    saver = tf.train.Saver()
    saver.save(sess, "./model/model.ckpt")


def DDPG_Example():
    randomizer = (minitaur_env_randomizer.MinitaurEnvRandomizer())
    env = minitaur_gym_env.MinitaurBulletEnv(
        render=True,
        motor_velocity_limit=np.inf,
        pd_control_enabled=True,
        hard_reset=False,
        env_randomizer=randomizer,
        shake_weight=0.0,
        drift_weight=0.0,
        energy_weight=0.0,
        on_rack=False)

    state_dim = 28
    action_dim = 4
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
    with tf.Session() as sess:
        actor = ActorNetwork(sess, state_dim, action_dim,
                             learning_rate=float(LR_A), tau=TAU,
                             batch_size=BATCH_SIZE)

        critic = CriticNetwork(sess, state_dim, action_dim,
                               learning_rate=float(LR_C), tau=TAU,
                               gamma=float(GAMMA),
                               num_actor_vars=actor.get_num_trainable_vars())

        train(sess, env, actor, critic, actor_noise, load_weights=True)


if __name__ == "__main__":
    DDPG_Example()
