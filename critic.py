import keras.layers
import tensorflow as tf
import numpy as np
class CriticNetwork_DRRN(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork_DRRN, self).__init__()
        self.inputs = tf.keras.layers.InputLayer(input_shape=(action_dim, state_dim))
        self.fc1 = tf.keras.layers.Dense(state_dim/2, activation='linear')
        self.concat = tf.keras.layers.Concatenate()
        self.fc2 = tf.keras.layers.Dense(state_dim/4, activation='elu')
        self.fc3 = tf.keras.layers.Dense(state_dim/8, activation='elu')
        self.out = tf.keras.layers.Dense(1, activation='linear')

    def call(self, x):
        # 第一层处理state
        s = self.fc1(x[1])
        # 连接上动作和状态
        s = self.concat([x[0],  s])
        s = self.fc2(s)
        s = self.fc3(s)
        return self.out(s)


class Critic_DRRN(object):

    def __init__(self, learning_rate, state_dim,action_dim, target_network_update_rate):
        # 状态向量的维度
        self.state_dim = state_dim
        # 动作向量的维度
        self.action_dim = action_dim
        # 隐藏层的维度
        # critic network / target network
        self.network = CriticNetwork_DRRN(state_dim=state_dim,action_dim=action_dim)
        self.target_network = CriticNetwork_DRRN(state_dim=state_dim,action_dim=action_dim)
        # build并summary
        self.network([np.zeros((1, self.action_dim)),np.zeros((1, state_dim))])
        self.target_network([np.zeros((1, self.action_dim)),np.zeros((1, state_dim))])
        self.network.summary()
        self.target_network.summary()
        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # 损失函数
        self.loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        # 网络编译
        self.network.compile(self.optimizer, self.loss)
        #  soft target network update hyperparameter
        self.target_network_update_rate = target_network_update_rate
    def estimate_state_action(self,state):
        self.network.call(state)
    # 给状态——动作对打分（double Q）
    def target_estimate_state_action(self,state):
        self.target_network.call(state)
    # 目标网络更新
    def update_target_network(self):
        c_omega = self.network.get_weights()
        t_omega = self.target_network.get_weights()
        for i in range(len(c_omega)):
            t_omega[i] = self.target_network_update_rate * c_omega[i] + (1 - self.target_network_update_rate) * t_omega[i]
        self.target_network.set_weights(t_omega)

    #  q对a求导（a之后对actor的参数求导让actor朝着最大化q的方向优化）
    def dq_da(self, inputs):
        actions = inputs[0]
        states = inputs[1]
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as g:
            # 转为tensor才能求梯度
            actions = tf.convert_to_tensor(actions)
            g.watch(actions)
            qualities = self.network([actions, states])
        q_grads = g.gradient(qualities, actions)
        return q_grads
    # 训练一轮（拟合q）

    def train_only(self, inputs, td_targets):
        # 单纯的把target作为label去拟合
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as g:
            outputs = self.network(inputs)
            loss = self.loss(td_targets, outputs)
            weighted_loss = tf.reduce_mean(loss)
        dl_domega = g.gradient(weighted_loss, self.network.trainable_weights)
        grads = zip(dl_domega, self.network.trainable_weights)
        self.optimizer.apply_gradients(grads)
        return weighted_loss

    def train(self, inputs, td_targets, weight_batch):
        # 单纯的把target作为label去拟合
        weight_batch = tf.convert_to_tensor(weight_batch, dtype=tf.float32)
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as g:
            outputs = self.network(inputs)
            loss = self.loss(td_targets, outputs)
            weighted_loss = tf.reduce_mean(loss * weight_batch)
        dl_domega = g.gradient(weighted_loss, self.network.trainable_weights)
        grads = zip(dl_domega, self.network.trainable_weights)
        self.optimizer.apply_gradients(grads)
        return weighted_loss
    # 保存权重
    def save_weights(self, path):
        self.network.save_weights(path)

    # 加载权重
    def load_weights(self, path):
        self.network.load_weights(path)






