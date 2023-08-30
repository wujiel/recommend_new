import keras
import tensorflow
import numpy

# actor网络，输入state，返回action
class ActorNetwork_DRRN(tensorflow.keras.Model):
    def __init__(self, state_dim,action_dim):
        super(ActorNetwork_DRRN, self).__init__()
        self.inputs = tensorflow.keras.layers.InputLayer(name='input_layer', input_shape=(state_dim))
        self.fc1 = tensorflow.keras.Sequential([
            tensorflow.keras.layers.Dense(state_dim/2, activation='linear'),
            tensorflow.keras.layers.Dense(state_dim/4, activation='elu'),
            tensorflow.keras.layers.Dense(state_dim/8, activation='linear')
        ])
        self.out = tensorflow.keras.layers.Dense(action_dim, activation='tanh')

    def call(self, x):
        x = self.inputs(x)
        x = self.fc1(x)
        return self.out(x)


class Actor_DRRN(object):
    def __init__(self, state_dim, action_dim, learning_rate, target_network_update_rate):
        # 直接输出action
        self.network = ActorNetwork_DRRN(state_dim=state_dim,action_dim=action_dim)
        self.target_network = ActorNetwork_DRRN(state_dim=state_dim,action_dim=action_dim)
        # build并summary
        self.network(numpy.zeros((1, state_dim)))
        self.target_network(numpy.zeros((1, state_dim)))
        self.network.summary()
        self.target_network.summary()
        # 优化器
        self.optimizer = tensorflow.keras.optimizers.Adam(learning_rate)
        # soft target network update hyperparameter
        self.target_network_update_rate = target_network_update_rate
    def act(self, state):
        action = self.network.call(state)
        return action

    def target_act(self, state):
        action = self.target_network.call(state)
        return action

    def update_target_network(self):
        # target网络更新
        c_theta, t_theta = self.network.get_weights(), self.target_network.get_weights()
        for i in range(len(c_theta)):
            t_theta[i] = self.target_network_update_rate * c_theta[i] + (1 - self.target_network_update_rate) * t_theta[i]
        self.target_network.set_weights(t_theta)

    def train(self, states, dq_das):
        with tensorflow.GradientTape(persistent=True,watch_accessed_variables=True) as g:
            actions = self.network(states)
            dj_dtheta = g.gradient(actions, self.network.trainable_weights, -dq_das)
            grads_m = zip(dj_dtheta, self.network.trainable_weights)
            self.optimizer.apply_gradients(grads_m)

    def save_weights(self,path):
        self.network.save_weights(path)

    def load_weights(self, path):
        self.network.load_weights(path)

