import tensorflow as tf
import numpy
from state_representation import DRRAveStateRepresentation
from EmbeddingNetwork import MultiLayerNetwork
import tensorflow
"""
状态表示模块应该掌握所有用户的历史交互信息
网络训练应尽量打乱数据，

"""
class DdpgCriticNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DdpgCriticNetwork, self).__init__()
        self.inputs = tf.keras.layers.InputLayer(input_shape=(action_dim, state_dim))
        self.fc1 = tf.keras.layers.Dense(action_dim, activation='relu')
        self.concat = tf.keras.layers.Concatenate()
        self.fc2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc3 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.out = tf.keras.layers.Dense(1, activation='linear')

    def call(self, x):
        # 第一层处理state
        s = self.fc1(x[1])
        # 连接上均值和方差
        s = self.concat([x[0],  s])
        s = self.fc2(s)
        s = self.fc3(s)
        return self.out(s)


class DdpgCritic(object):

    def __init__(self, hidden_dim, learning_rate, state_dim, action_dim, target_network_update_rate):
        # 目前全部用户的信息
        self.all_history_dict = {}

        # embedding网络
        self.embedding_network = MultiLayerNetwork(len_users=6040, len_movies=3900, embedding_dim=100)
        self.embedding_network([numpy.zeros((1)), numpy.zeros((1))])
        print("已创建embedding网络")
        self.embedding_network.load_weights(r'embedding_weights/multilayer/multilayer_network_weights2400000.h5')
        print("已加载权重")
        self.embedding_network.summary()



        # 状态向量的维度
        self.state_dim = state_dim
        # 动作向量的维度
        self.action_dim = action_dim
        # 隐藏层的维度
        self.hidden_dim = hidden_dim
        # critic network / target network
        self.network = DdpgCriticNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.target_network = DdpgCriticNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        # build并summary
        self.network([numpy.zeros((1, self.action_dim)), numpy.zeros((1, state_dim))])
        self.target_network([numpy.zeros((1, self.action_dim)), numpy.zeros((1, state_dim))])
        print("ddpg_critic_network已创建")
        print()
        self.network.summary()
        print("ddpg_target_critic_network已创建")
        self.target_network.summary()
        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # 损失函数
        self.loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        # 网络编译
        self.network.compile(self.optimizer, self.loss)
        #  soft target network update hyperparameter
        self.target_network_update_rate = target_network_update_rate

    # 给状态——动作对打分
    def estimate_state_action(self, state):
        self.network.call(state)

    # 给状态——动作对打分（double Q）
    def target_estimate_state_action(self, state):
        self.target_network.call(state)

    # 目标网络更新
    def update_target_network(self):
        c_omega = self.network.get_weights()
        t_omega = self.target_network.get_weights()
        for i in range(len(c_omega)):
            t_omega[i] = self.target_network_update_rate * c_omega[i] + (1 - self.target_network_update_rate) * t_omega[
                i]
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
        # print("shit japanese")
        return q_grads

    # 训练一轮（拟合q）
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

class RepresentationModel:
    def __init__(self,short_term_state_size):
        # 短期模式包含最近short_term_state_size条交互
        self.short_term_state_size = short_term_state_size
        # 包括item_id列表和评分列表
        self.history = None
        # 用户id
        self.user_id = None
        # 目前全部用户的信息
        self.all_history_dict = {}


        # embedding网络
        self.embedding_network = MultiLayerNetwork(len_users=6040, len_movies=3900, embedding_dim=100)
        self.embedding_network([numpy.zeros((1)), numpy.zeros((1))])
        print("已创建embedding网络")
        self.embedding_network.load_weights(r'embedding_weights/multilayer/multilayer_network_weights2400000.h5')
        print("已加载权重")
        self.embedding_network.summary()
        self.state_represent_network = DRRAveStateRepresentation(embedding_dim=100)
        self.state_represent_network([numpy.zeros((1, 100,)), numpy.zeros((1, self.short_term_state_size, 100))])
        print("已创建state_represent_network")
        self.state_represent_network.summary()

        # optimizer优化器
        self.optimizer = tensorflow.keras.optimizers.Adam()
        # loss损失函数
        self.loss_function = tensorflow.keras.losses.CategoricalCrossentropy()
        #

    def calculate_state(self,short_term_item_ids,user_id):
        # embedding向量计算
        user_embedding_tensor = self.embedding_network.get_layer('user_embedding')(numpy.array(user_id))
        short_term_item_ids_embedding_tnnsor = self.embedding_network.get_layer('movie_embedding')(numpy.array(short_term_item_ids))
        ## 组合成状态向量
        state = self.state_represent_network([numpy.expand_dims(user_embedding_tensor, axis=0), numpy.expand_dims(short_term_item_ids_embedding_tnnsor, axis=0)])
        return state

    def get_state(self):
        # 先更新embedding网络
        # self.train()
        short_term_item_ids = self.history['item_ids_list'][-self.short_term_state_size:]
        state = self.calculate_state(short_term_item_ids,self.user_id)
        return state

    # 状态表示也是需要更新的
    def train(self):
        user_batch, movie_batch, label_batch = self.generate_rate_batch()
        for i in range(10):
            precision = self.train_step([user_batch, movie_batch], label_batch)
            if precision > 0.9:
                break


            # print("precision_llllaaaaa:",precision)
        # print("precision:",precision)


    # 更新一步embedding网络
    def train_step(self,inputs, labels):
        with tensorflow.GradientTape() as tape:
            predictions = self.embedding_network(inputs, training=True)
            labels_to_train = self.label_translation(labels)
            # labels = keras.backend.eval(labels)
            # 这个损失函数不需要转成tensor也能算
            loss = self.loss_function(labels_to_train, predictions)
            predictions = self.predictions_translation(predictions)
            precision = self.calculate_accuracy(labels, predictions)

        gradients = tape.gradient(loss, self.embedding_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.embedding_network.trainable_variables))
        return precision

    '''最好优先学新的历史记录'''
    def generate_rate_batch(self):
        user_id_history = self.all_history_dict[self.user_id]
        user_batch = numpy.random.randint(self.user_id,self.user_id+1,size=(self.short_term_state_size))
        movie_batch = numpy.array(user_id_history['item_ids_list'][-self.short_term_state_size:])
        label_batch = numpy.array(user_id_history['rates_list'][-self.short_term_state_size:])
        return user_batch,movie_batch,label_batch



    # 多分类，将预测值转为一维度格式
    def predictions_translation(self,predictions):
        result = numpy.zeros((len(predictions)))
        for i, label in enumerate(predictions):
            rate = numpy.argmax(label) + 1
            result[i] = rate

        return result

    # 多分类，将标签转为相应格式
    def label_translation(self, labels):
        result = numpy.zeros((len(labels), 5))
        for i, label in enumerate(labels):
            result[i][int(label - 1)] = 1
        return result

    def calculate_accuracy(self, labels, predictions):
        correct = 0
        for i in range(len(labels)):
            if labels[i] == predictions[i]:
                correct += 1
        return correct / len(labels)
