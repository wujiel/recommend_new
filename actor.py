import keras
import tensorflow
import numpy

# actor网络，输入state，返回action
class ActorNetwork(tensorflow.keras.Model):
    def __init__(self, state_dim,action_dim, hidden_dim):
        super(ActorNetwork, self).__init__()
        self.inputs = tensorflow.keras.layers.InputLayer(name='input_layer', input_shape=(state_dim*10,))
        self.fc1 = tensorflow.keras.Sequential([
            tensorflow.keras.layers.Dense(hidden_dim, activation='relu'),
            tensorflow.keras.layers.Dense(hidden_dim, activation='relu'),
            tensorflow.keras.layers.Dense(hidden_dim, activation='relu')
        ])
        self.out = tensorflow.keras.layers.Dense(action_dim, activation='tanh')

    def call(self, x):
        x = self.inputs(x)
        x = self.fc1(x)
        return self.out(x)

# 显式rnn网络
# class DdpgExplicitRnnActorNetwork(tensorflow.keras.Model):
#     def __init__(self, state_dim, action_dim, hidden_dim):
#         super(DdpgExplicitRnnActorNetwork, self).__init__()
#         self.inputs = tensorflow.keras.layers.InputLayer(input_shape=(state_dim*2))
#
#         # 处理之前输出的层（相当于rnn的记忆单元），处理x[0](由之前的输出组成)
#         self.layer_before = tensorflow.keras.layers.Dense(action_dim, activation='relu')
#         # 处理最近状态的层，处理x[1](由最近的状态组成)
#         self.layer_now = tensorflow.keras.layers.Dense(action_dim, activation='relu')
#
#
#         self.normal1 = tensorflow.keras.layers.BatchNormalization\
#             (axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
#              gamma_initializer='ones', moving_mean_initializer='zeros',
#              moving_variance_initializer='ones', beta_regularizer=None,
#              gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
#
#         self.normal2 = tensorflow.keras.layers.BatchNormalization \
#             (axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
#              gamma_initializer='ones', moving_mean_initializer='zeros',
#              moving_variance_initializer='ones', beta_regularizer=None,
#              gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
#         self.normal3 = tensorflow.keras.layers.BatchNormalization \
#             (axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
#              gamma_initializer='ones', moving_mean_initializer='zeros',
#              moving_variance_initializer='ones', beta_regularizer=None,
#              gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
#         self.normal4 = tensorflow.keras.layers.BatchNormalization \
#             (axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
#              gamma_initializer='ones', moving_mean_initializer='zeros',
#              moving_variance_initializer='ones', beta_regularizer=None,
#              gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
#         self.fc1 = tensorflow.keras.layers.Dense(action_dim, activation='relu')
#         self.concat = tensorflow.keras.layers.Concatenate()
#         self.fc2 = tensorflow.keras.layers.Dense(hidden_dim, activation='relu')
#         self.fc3 = tensorflow.keras.layers.Dense(hidden_dim, activation='relu')
#         self.out = tensorflow.keras.layers.Dense(action_dim, activation='tanh')
#
#     def call(self, x):
#         before = x[:,0:300]
#         now = x[:,300:]
#         s_before = self.layer_before(before)
#         # s_before = self.normal1(s_before)
#         s_now = self.layer_now(now)
#         # s_now = self.normal2(s_now)
#         s = self.concat([s_before,s_now])
#         s = self.fc1(s)
#         # s = self.normal3(s)
#         s = self.fc2(s)
#         # s = self.normal4(s)
#         s = self.fc3(s)
#         return self.out(s)

class DdpgExplicitRnnActorNetwork(tensorflow.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DdpgExplicitRnnActorNetwork, self).__init__()
        self.inputs = tensorflow.keras.layers.InputLayer(input_shape=(1700))

        # 处理movie_eb并生成二级信息
        self.layer_movie_eb_deal = tensorflow.keras.layers.Dense(action_dim, activation='tanh')
        # 处理second并生成三级信息
        self.layer_second_deal = tensorflow.keras.layers.Dense(action_dim, activation='tanh')
        # 处理三级信息
        self.layer_third_deal = tensorflow.keras.layers.Dense(action_dim, activation='tanh')

        self.fc1 = tensorflow.keras.layers.Dense(action_dim, activation='relu')
        self.concat1 = tensorflow.keras.layers.Concatenate()
        self.concat2 = tensorflow.keras.layers.Concatenate()
        self.concat3 = tensorflow.keras.layers.Concatenate()
        self.concat4 = tensorflow.keras.layers.Concatenate()
        self.fc2 = tensorflow.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc3 = tensorflow.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc4 = tensorflow.keras.layers.Dense(hidden_dim, activation='relu')
        self.out = tensorflow.keras.layers.Dense(action_dim, activation='tanh')
    def call(self, x):
        movie_eb = x[:,0:500]
        second_info = x[:,500:1000]
        third_info = x[:,1000:1500]
        action_before_user_eb = x[:,1500:1700]

        second_info_new = self.layer_movie_eb_deal(movie_eb)
        third_info_new = self.layer_second_deal(second_info)
        # 四级信息不返回
        fourth_info_new = self.layer_third_deal(third_info)

        s1 = self.concat1([second_info_new,third_info_new])
        s = self.fc1(s1)
        # 让四级信息靠下一些
        s2 = self.concat2([s,fourth_info_new])
        s = self.fc2(s2)
        s3 = self.concat2([s,action_before_user_eb])
        s = self.fc3(s3)
        s = self.fc4(s)
        return self.out(s),second_info_new,third_info_new



class DdpgActor(object):
    def __init__(self, state_dim, hidden_dim,action_dim, learning_rate, target_network_update_rate):
        # 直接输出action
        self.network = DdpgExplicitRnnActorNetwork(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim)
        self.target_network = DdpgExplicitRnnActorNetwork(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim)
        #  Build networks并summary（）
        # build并summary
        self.network(numpy.zeros((1, 1700)))
        self.target_network(numpy.zeros((1, 1700)))

        # self.network(numpy.zeros((1, state_dim)))
        # self.target_network(numpy.zeros((1, state_dim)))
        print("ddpg网络已创建")
        self.network.summary()
        print("ddpg目标网络已创建")
        self.target_network.summary()
        # 优化器
        self.optimizer = tensorflow.keras.optimizers.Adam(learning_rate)
        # soft target network update hyperparameter
        self.target_network_update_rate = target_network_update_rate

    def act(self, state):
        action,second_info,third_info = self.network.call(state)
        return action,second_info,third_info

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
            actions,_,_ = self.network(states)
            dj_dtheta = g.gradient(actions, self.network.trainable_weights, -dq_das)
            grads_m = zip(dj_dtheta, self.network.trainable_weights)
            self.optimizer.apply_gradients(grads_m)

    def save_weights(self,path):
        self.network.save_weights(path)

    def load_weights(self, path):
        self.network.load_weights(path)


class DdpgAveActor(object):
    def __init__(self, state_dim, hidden_dim,action_dim, learning_rate, target_network_update_rate):
        # 直接输出action
        self.network = ActorNetwork(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim)
        self.target_network = ActorNetwork(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim)
        #  Build networks并summary（）
        # build并summary
        self.network(numpy.zeros((1, state_dim*10)))
        self.target_network(numpy.zeros((1, state_dim*10)))

        # self.network(numpy.zeros((1, state_dim)))
        # self.target_network(numpy.zeros((1, state_dim)))
        print("ddpg网络已创建")
        self.network.summary()
        print("ddpg目标网络已创建")
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

