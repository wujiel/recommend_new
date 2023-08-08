import copy
import numpy.random
import tensorflow
import numpy
import matplotlib.pyplot as plt
from state_representation import DRRAveStateRepresentation

'''
训练方式：选出历史记录中的好评记录
抛开embedding，本质上是一个在线模型
'''



def getMat(user_history,user_num,item_num):
    # 构造U-I评分矩阵
    UI_matrix = numpy.zeros((user_num,item_num))
    UI_matrix_01 = numpy.zeros((user_num,item_num))
    # 遍历历史数据，令uimat[u][i] = r
    for user in user_history:
        for tuple_rate in user_history[user]:
            UI_matrix[user][tuple_rate[0]] = tuple_rate[1]
            if tuple_rate[1]>3:
                UI_matrix_01[user][tuple_rate[0]] = 1
            else:
                UI_matrix_01[user][tuple_rate[0]] = 0


    return UI_matrix,UI_matrix_01

from RepresentationModel import RepresentationModel
class RecommendSystem:

    def __init__(self, env,ddpg_actor,ddpg_critic, ddpg_buffer,ddpg_ave_actor,ddpg_ave_critic, ddpg_ave_buffer,embedding_network,is_test = False):
        #模拟环境00
        self.env = env



        # 对比实验DDRave
        self.state_represent_network = DRRAveStateRepresentation(embedding_dim=100)
        self.state_represent_network([numpy.zeros((1, 1000,)), numpy.zeros((1, 10, 1000))])
        print("已创建state_represent_network")
        self.state_represent_network.summary()

        self.ddpg_ave_actor = ddpg_ave_actor
        self.ddpg_ave_critic = ddpg_ave_critic
        # ddpg_ave_buffer(经历重放的容器)
        self.ddpg_ave_buffer = ddpg_ave_buffer
        # embedding网络还是应该由推荐系统本身来维护
        # self.embedding_network = MultiLayerNetwork(len_users=6040, len_movies=3900, embedding_dim=100)
        # self.embedding_network([numpy.zeros((1)), numpy.zeros((1))])
        # self.embedding_network_cold = MultiLayerNetwork(len_users=6040, len_movies=3900, embedding_dim=100)
        # self.embedding_network_cold([numpy.zeros((1)), numpy.zeros((1))])
        # print("已创建embedding网络")
        # self.embedding_network.load_weights(r'embedding_weights/multilayer/multilayer_network_weights2400000.h5')
        # print("已加载权重")
        # self.embedding_network.summary()

        # embedding网络还是应该由推荐系统本身来维护

        print("已创建embedding网络")

        self.embedding_network = embedding_network
        # model.load_weights(r'embedding_weights/muti/multi_network_weights99000.h5')



        # ddpg_actor
        self.ddpg_actor = ddpg_actor
        # ddpg_critic
        self.ddpg_critic = ddpg_critic
        # ddpg_buffer(经历重放的容器)
        self.ddpg_buffer = ddpg_buffer

        # ddpg采用ε-贪婪探索
        self.epsilon = 1.
        self.epsilon_decay = (self.epsilon - 0.1) / 500000

        # ddpg采用ε-贪婪探索
        self.epsilon_ave = 1.
        self.epsilon_ave_decay = (self.epsilon_ave - 0.1) / 500000

        # 随机探索时的方差
        self.std = 1.5

        self.is_test = is_test
        #训练监测模块
        # 从buffer里取出的批量大小
        self.batch_size = 32
        # 未来期望奖励的衰减γ值
        self.discount_factor = 0.9
        # buffer容器的
        self.epsilon_for_priority = 1e-6

        # 推荐空间，1—D array
        # Ddpg方式训练时的推荐空间
        self.ddpg_recommend_space = copy.deepcopy(self.env.recommend_space_train)
        # 随机方式训练时的推荐空间
        self.random_recommend_space = copy.deepcopy(self.env.recommend_space_train)

        # 对推荐系统来说，它交互过的全部历史记录，初始为空
        # 历史记录,字典类型，三个列表，分别是item—id和评分和action，用来进行推荐
        # Ddpg方式训练时的历史记录
        self.ddpg_history_dict_all = {}
        # Ddpg_ave方式训练时的历史记录
        self.ddpg_ave_history_dict_all = {}
        # 随机方式训练时的历史记录
        self.random_history_dict_all = {}

        # 训练时各个指标记录，用来可视化指标
        self.train_history_dict = {}
        self.train_history_dict["steps"] = []

        self.train_history_dict["precision"] = {}
        self.train_history_dict["precision_mean"] = {}
        self.train_history_dict["precision"]["ddpg_positive_precision"] = []
        self.train_history_dict["precision"]["ddpg_rewards_precision"] = []
        self.train_history_dict["precision"]["ddpg_ave_positive_precision"] = []
        self.train_history_dict["precision"]["ddpg_ave_rewards_precision"] = []
        self.train_history_dict["precision"]["random_positive_precision"] = []
        self.train_history_dict["precision"]["random_rewards_precision"] = []

        self.train_history_dict["precision_mean"]["our_positive_precision_mean"] = []
        self.train_history_dict["precision_mean"]["ave_positive_precision_mean"] = []
        self.train_history_dict["precision_mean"]["random_positive_precision_mean"] = []

        # self.train_history_dict["precision_mean"]["our_rewards_precision_mean"] = []
        # self.train_history_dict["precision_mean"]["ave_rewards_precision_mean"] = []
        # self.train_history_dict["precision_mean"]["random_rewards_precision_mean"] = []


    # 根据用户id，全部用户历史记录，推荐空间，top-k。做出推荐，核心方法。返回推荐项目列表，直接传给环境获取真实评价
    '''
    之前的状态应该保留
    前几次的action说不定可以
    action和movie—eb的混合作为状态
    action带有前面的交互信息
    eb带有电影本身的信息
    '''


    # 根据最近几条（具体几条可以调）item—eb和之前几条（一样）action推荐，暂时各3条
    # 在没有历史记录的情况下应先使用随机推荐，模拟试验中直接从环境获取

    # 不
    def recommend_item_ddpg_experiment(self, user_id, state_before,top_k=False):
        # 最近3条记录的embedding
        self.env.all_history_dict[user_id]["movie_id"]
        complex_items_eb = self.embedding_network.get_layer('movie_embedding')(self.ddpg_history_dict_all[user_id][-3:])



        ddpg_state = self.ddpg_critic.get_state(user_id,self.ddpg_history_dict_all,self.embedding_network)
        ddpg_action = self.ddpg_actor.act(state=ddpg_state)
        ## ε-greedy exploration
        if self.epsilon > numpy.random.uniform() and not self.is_test:
            self.epsilon -= self.epsilon_decay
            ddpg_action += numpy.random.normal(0, self.std, size=ddpg_action.shape)
        # 待推荐项目的空间，可以考虑训练和应用策略分开
        recommend_space_ebs = self.embedding_network.get_layer('movie_embedding')(self.ddpg_recommend_space)
        action_t = tensorflow.transpose(ddpg_action, perm=(1, 0))
        if top_k:
            # arg函数返回的是索引
            item_index = numpy.argsort(
                tensorflow.transpose(tensorflow.keras.backend.dot(recommend_space_ebs, action_t), perm=(1, 0)))[0][
                         -top_k:]
            return self.ddpg_recommend_space[item_index], ddpg_action,ddpg_state
        else:
            rank = tensorflow.keras.backend.dot(recommend_space_ebs, action_t)
            item_idx = numpy.argmax(rank)
            ddpg_item_id = self.ddpg_recommend_space[item_idx]
            return ddpg_item_id, ddpg_action,ddpg_state

    # 获取用户的最新状态
    def get_state(self,user_id):
        # 最近3条记录的embedding
        # 填充向量
        # 一级状态信息即是电影本身的eb向量
        complex_movies_eb = numpy.ones((1, 500))
        # 二级状态信息
        second_state_info = numpy.ones((1, 500))
        # 三级状态信息
        third_state_info = numpy.ones((1, 500))

        action = self.ddpg_history_dict_all[user_id]["actions"][-1:]
        if len(action)>0:
            action = action[0]
        else:
            action = numpy.ones((1, 100))
        movie_record_list = copy.deepcopy(self.ddpg_history_dict_all[user_id]["movie_id"][-5:])
        second_state_info_list = copy.deepcopy(self.ddpg_history_dict_all[user_id]["second"][-5:-26:-5])
        third_state_info_list = copy.deepcopy(self.ddpg_history_dict_all[user_id]["third"][-25:-126:-25])
        second_state_info_list.reverse()
        third_state_info_list.reverse()


        for movie_record in movie_record_list:
            movie_eb = self.embedding_network.get_layer('movie_embedding')(numpy.array([movie_record]))
            complex_movies_eb = numpy.concatenate([complex_movies_eb,movie_eb], axis=1)
        for second_state in second_state_info_list:
            second_state_info = numpy.concatenate([second_state_info,second_state], axis=1)
        for third_state in third_state_info_list:
            third_state_info = numpy.concatenate([third_state_info,third_state], axis=1)

        # 目前的历史长度

        complex_movies_eb = complex_movies_eb[:,-500:]
        second_state_info = second_state_info[:,-500:]
        third_state_info = third_state_info[:,-500:]

        user_embedding_tensor = self.embedding_network.get_layer('user_embedding')(numpy.array(user_id))
        user_embedding_tensor = numpy.expand_dims(user_embedding_tensor, axis=0)



        ddpg_state = numpy.concatenate([complex_movies_eb,second_state_info], axis=1)
        ddpg_state = numpy.concatenate([ddpg_state,third_state_info], axis=1)
        ddpg_state = numpy.concatenate([ddpg_state,action], axis=1)
        ddpg_state = numpy.concatenate([ddpg_state,user_embedding_tensor], axis=1)
        return ddpg_state

    def get_state_ave(self,user_id):
        # 填充向量
        complex_movies_eb = numpy.ones((100, 100))
        movie_record_list = copy.deepcopy(self.ddpg_ave_history_dict_all[user_id]["movie_id"][-60:])
        movie_eb = self.embedding_network.get_layer('movie_embedding')(numpy.array(movie_record_list))
        complex_movies_eb = numpy.concatenate([complex_movies_eb,movie_eb], axis=0)
        complex_movies_eb = complex_movies_eb[-100:,:]

        complex_movies_eb = complex_movies_eb.reshape(10,1000)
        # embedding向量计算
        user_embedding_tensor = self.embedding_network.get_layer('user_embedding')(numpy.array(user_id))
        user_embedding_tensor_1000 = user_embedding_tensor

        for i in range(9):
            user_embedding_tensor_1000 = numpy.concatenate([user_embedding_tensor_1000, user_embedding_tensor], axis=0)


        ## 组合成状态向量
        ddpg_ave_state = self.state_represent_network([numpy.expand_dims(user_embedding_tensor_1000, axis=0),
                                                      numpy.expand_dims(complex_movies_eb, axis=0)])

        return ddpg_ave_state



    def recommend_item_ddpg(self, user_id,top_k=False):
        ddpg_state = self.get_state(user_id)

        ddpg_action,second_info,third_info = self.ddpg_actor.act(state=ddpg_state)
        ## ε-greedy exploration
        if self.epsilon > numpy.random.uniform() and not self.is_test:
            self.epsilon -= self.epsilon_decay
            ddpg_action += numpy.random.normal(0, self.std, size=ddpg_action.shape)
        # 待推荐项目的空间，可以考虑训练和应用策略分开
        recommend_space_ebs = self.embedding_network.get_layer('movie_embedding')(self.ddpg_recommend_space)
        action_t = tensorflow.transpose(ddpg_action, perm=(1, 0))
        if top_k:
            # arg函数返回的是索引
            item_index = numpy.argsort(
                tensorflow.transpose(tensorflow.keras.backend.dot(recommend_space_ebs, action_t), perm=(1, 0)))[0][
                         -top_k:]
            return self.ddpg_recommend_space[item_index], ddpg_action,ddpg_state,second_info,third_info
        else:
            rank = tensorflow.keras.backend.dot(recommend_space_ebs, action_t)
            item_idx = numpy.argmax(rank)
            ddpg_item_id = self.ddpg_recommend_space[item_idx]
            return ddpg_item_id, ddpg_action,ddpg_state,second_info,third_info

    def recommend_item_ddpg_ave(self, user_id,top_k=False):
        ddpg_state = self.get_state_ave(user_id)

        ddpg_action = self.ddpg_ave_actor.act(state=ddpg_state)
        ## ε-greedy exploration
        if self.epsilon_ave > numpy.random.uniform() and not self.is_test:
            self.epsilon_ave -= self.epsilon_ave_decay
            ddpg_action += numpy.random.normal(0, self.std, size=ddpg_action.shape)
        # 待推荐项目的空间，可以考虑训练和应用策略分开
        recommend_space_ebs = self.embedding_network.get_layer('movie_embedding')(self.ddpg_ave_recommend_space)
        action_t = tensorflow.transpose(ddpg_action, perm=(1, 0))
        if top_k:
            # arg函数返回的是索引
            item_index = numpy.argsort(
                tensorflow.transpose(tensorflow.keras.backend.dot(recommend_space_ebs, action_t), perm=(1, 0)))[0][
                         -top_k:]
            return self.ddpg_ave_recommend_space[item_index], ddpg_action,ddpg_state
        else:
            rank = tensorflow.keras.backend.dot(recommend_space_ebs, action_t)
            item_idx = numpy.argmax(rank)
            ddpg_item_id = self.ddpg_ave_recommend_space[item_idx]
            return ddpg_item_id, ddpg_action,ddpg_state


    def recommend_item_random(self,top_k=False):
        if top_k:
            random_items = numpy.random.choice(self.random_recommend_space, top_k)
            return random_items
        else:
            random_item = numpy.random.choice(self.random_recommend_space, 1)[0]
            return random_item

    # TD学习的目标值也就是网络应该去拟合的label值
    def ddpg_calculate_td_target(self, rewards, dones, batch_states, batch_next_states):
        # newQ = r+γ*nextQ
        # nextQ = t_critic(next_s,next_a)
        # next_a = t_actor(next_s)

        # 计算下一动作用target网络
        target_actions,_,_ = self.ddpg_actor.target_act(batch_next_states)
        # critic的网络和目标网络分别给下一状态动作打分以作为未来期望奖励 Double Q method
        qs = self.ddpg_critic.network([target_actions, batch_next_states])
        target_qs = self.ddpg_critic.target_network([target_actions, batch_next_states])
        min_qs = tensorflow.raw_ops.Min(input=tensorflow.concat([target_qs, qs], axis=1), axis=1, keep_dims=True)
        y_t = numpy.copy(min_qs)
        for i in range(min_qs.shape[0]):
            y_t[i] = rewards[i] + (1 - dones[i]) * (self.discount_factor * min_qs[i])
        return y_t

    def ddpg_ave_calculate_td_target(self, rewards, dones, batch_states, batch_next_states):
        # newQ = r+γ*nextQ
        # nextQ = t_critic(next_s,next_a)
        # next_a = t_actor(next_s)

        # 计算下一动作用target网络
        target_actions = self.ddpg_ave_actor.target_act(batch_next_states)
        # critic的网络和目标网络分别给下一状态动作打分以作为未来期望奖励 Double Q method
        qs = self.ddpg_ave_critic.network([target_actions, batch_next_states])
        target_qs = self.ddpg_ave_critic.target_network([target_actions, batch_next_states])
        min_qs = tensorflow.raw_ops.Min(input=tensorflow.concat([target_qs, qs], axis=1), axis=1, keep_dims=True)
        y_t = numpy.copy(min_qs)
        for i in range(min_qs.shape[0]):
            y_t[i] = rewards[i] + (1 - dones[i]) * (self.discount_factor * min_qs[i])
        return y_t


    # 传入评分获取奖励
    def reward_represent(self,rates):
        rewards = (rates-3)/2
        return rewards


    def train_network(self):
        # 取出一个批量
        ddpg_batch_states, ddpg_batch_actions, ddpg_batch_rewards, ddpg_batch_next_states, ddpg_batch_dones, weight_ddpg_batch, index_ddpg_batch = self.ddpg_buffer.sample(
            self.batch_size)

        # TD学习的目标值即q网络应该去拟合的label值
        td_targets = self.ddpg_calculate_td_target(ddpg_batch_rewards, ddpg_batch_dones, ddpg_batch_states,
                                                   ddpg_batch_next_states)
        # 更新容器的优先级
        for (p, i) in zip(td_targets, index_ddpg_batch):
            self.ddpg_buffer.update_priority(abs(p[0]) + self.epsilon_for_priority, i)

        q_loss = self.ddpg_critic.train([ddpg_batch_actions, ddpg_batch_states], td_targets, weight_ddpg_batch)

        # q对a求导再a对actor的参数求导最后得到分值对actor网络参数的导数，朝着使分值增大的方向优化，传反梯度
        s_grads = self.ddpg_critic.dq_da([ddpg_batch_actions, ddpg_batch_states])
        self.ddpg_actor.train(ddpg_batch_states, s_grads)
        self.ddpg_actor.update_target_network()
        self.ddpg_critic.update_target_network()

        return q_loss

    def train_ave_network(self):
        # 取出一个批量
        ddpg_ave_batch_states, ddpg_ave_batch_actions,ddpg_ave_batch_actions_, ddpg_ave_batch_rewards, \
        ddpg_ave_batch_next_states, ddpg_ave_batch_dones, weight_ddpg_ave_batch, index_ddpg_ave_batch = self.ddpg_ave_buffer.sample(
            self.batch_size)

        # TD学习的目标值即q网络应该去拟合的label值
        td_targets = self.ddpg_ave_calculate_td_target(ddpg_ave_batch_rewards, ddpg_ave_batch_dones, ddpg_ave_batch_states,
                                                   ddpg_ave_batch_next_states)
        # 更新容器的优先级
        for (p, i) in zip(td_targets, index_ddpg_ave_batch):
            self.ddpg_ave_buffer.update_priority(abs(p[0]) + self.epsilon_for_priority, i)

        q_loss = self.ddpg_ave_critic.train([ddpg_ave_batch_actions, ddpg_ave_batch_states], td_targets, weight_ddpg_ave_batch)

        # q对a求导再a对actor的参数求导最后得到分值对actor网络参数的导数，朝着使分值增大的方向优化，传反梯度
        s_grads = self.ddpg_ave_critic.dq_da([ddpg_ave_batch_actions_, ddpg_ave_batch_states])
        self.ddpg_ave_actor.train(ddpg_ave_batch_states, s_grads)
        self.ddpg_ave_actor.update_target_network()
        self.ddpg_ave_critic.update_target_network()

        return q_loss



    def get_rates_standards(self,rates_list):
        positive_sum = 0
        rewards_sum = 0
        for rate in rates_list:
            if rate>=3:
                positive_sum+=1
            if rate==3:
                rate = 3.5
            if rate==2:
                rate = 3
            rewards_sum += (rate-3)
        return positive_sum,rewards_sum


    # 模型训练
    def train(self, max_episode_num ,top_k=False, load_model=False):
        # 重置target网络
        self.ddpg_actor.update_target_network()
        self.ddpg_critic.update_target_network()

        print(load_model)
        train_image = {}
        train_image['rewards_all'] = {}
        train_image['positive_all'] = {}

        train_image['rewards_all']['ours_1700'] = []
        train_image['rewards_all']['ave_3000'] = []
        train_image['rewards_all']['random'] = []

        train_image['positive_all']['ours_1700'] = []
        train_image['positive_all']['ave_3000'] = []
        train_image['positive_all']['random'] = []

        space_rewards_all = 0
        our_rewards_all = 0
        ave_rewards_all = 0
        random_rewards_all = 0

        space_positive_all = 0
        our_positive_all = 0
        ave_positive_all = 0
        random_positive_all = 0

        our_in_all = 1
        ave_in_all = 1
        random_in_all = 1



        step_all = 0

        '''
               给推荐历史（推荐系统可见的）添加一个用户历史记录字典,每轮用户重设时都在这里
               重新赋一个带两个空列表的字典，这样当选择的用户已经被选择过时，会清空之前的历史记录重新当做一个新用户进行训练
               '''
        for user_id in range(1,6041):
            self.ddpg_ave_history_dict_all[user_id] = {}
            self.ddpg_ave_history_dict_all[user_id]["movie_id"] = []
            self.ddpg_ave_history_dict_all[user_id]["rates"] = []

            self.ddpg_history_dict_all[user_id] = {}
            self.ddpg_history_dict_all[user_id]["movie_id"] = []
            self.ddpg_history_dict_all[user_id]["rates"] = []
            # 多级压缩表示，一级就是电影本身
            self.ddpg_history_dict_all[user_id]["second"] = []
            self.ddpg_history_dict_all[user_id]["third"] = []
            self.ddpg_history_dict_all[user_id]["actions"] = []

            # 随机推荐也需要维护历史，用来算指标
            self.random_history_dict_all[user_id] = {}
            self.random_history_dict_all[user_id]["movie_id"] = []
            self.random_history_dict_all[user_id]["rates"] = []



        for episode in range(max_episode_num):
            # episodic reward 每轮置零
            ddpg_episode_reward = 0
            random_episode_reward = 0
            steps = 0
            ddpg_q_loss = 0
            ddpg_ave_q_loss = 0
            ddpg_mean_action = 0


            '''
            # Environment 从环境里随机获得一个用户以及历史，即每轮选一个模拟用户进行一次完整的经历
            # recommender进行推荐，user反馈reward（由评分构成，可复合多种指标进行设置）
            '''
            user_id,done = self.env.reset()
            '''
            给推荐历史（推荐系统可见的）添加一个用户历史记录字典,每轮用户重设时都在这里
            重新赋一个带两个空列表的字典，这样当选择的用户已经被选择过时，会清空之前的历史记录重新当做一个新用户进行训练
            '''
            self.ddpg_ave_history_dict_all[user_id] = {}
            self.ddpg_ave_history_dict_all[user_id]["movie_id"] = []
            self.ddpg_ave_history_dict_all[user_id]["rates"] = []


            self.ddpg_history_dict_all[user_id] = {}
            self.ddpg_history_dict_all[user_id]["movie_id"] = []
            self.ddpg_history_dict_all[user_id]["rates"] = []
            # 多级压缩表示，一级就是电影本身
            self.ddpg_history_dict_all[user_id]["second"] = []
            self.ddpg_history_dict_all[user_id]["third"] = []
            self.ddpg_history_dict_all[user_id]["actions"] = []

            # 随机推荐也需要维护历史，用来算指标
            self.random_history_dict_all[user_id] = {}
            self.random_history_dict_all[user_id]["movie_id"] = []
            self.random_history_dict_all[user_id]["rates"] = []


            """单个用户"""
            # Ddpg方式训练时的推荐空间
            self.ddpg_recommend_space = copy.deepcopy(self.env.recommend_space_train)

            # Ddpg_ave方式训练时的推荐空间
            self.ddpg_ave_recommend_space = copy.deepcopy(self.env.recommend_space_train)
            # 随机方式训练时的推荐空间
            self.random_recommend_space = copy.deepcopy(self.env.recommend_space_train)


            print("当前用户",user_id)
            print("推荐空间长度",len(self.env.recommend_space_train))
            print("推荐空间里的好评记录数",self.env.positive_rates_count)
            space_rewards_all += self.env.positive_rewards_sum
            space_positive_all += self.env.positive_rates_count
            # 全好评和全差评的视作无效用户不训练
            # while not done and self.env.positive_rates_count != 0 and self.env.positive_rates_count!=len(self.env.recommend_space_train):
            while not done :
                # 做出推荐
                ddpg_item_id, ddpg_action, ddpg_state,second_info,third_info = self.recommend_item_ddpg(user_id=user_id)
                ddpg_ave_item_id, ddpg_ave_action, ddpg_ave_state = self.recommend_item_ddpg_ave(user_id=user_id)
                random_item_id = self.recommend_item_random()
                step_all+=1

                #与用户交互得到评价
                random_rates,random_is_in = self.env.step(random_item_id, top_k=top_k)
                ddpg_rates,ddpg_is_in = self.env.step(ddpg_item_id, top_k=top_k)
                ddpg_ave_rates,ddpg_ave_is_in = self.env.step(ddpg_ave_item_id, top_k=top_k)

                if random_is_in:
                    random_in_all+=1
                if ddpg_is_in:
                    our_in_all+=1
                if ddpg_ave_is_in:
                    ave_in_all+=1

                # 奖励计算
                ddpg_rewards = self.reward_represent(ddpg_rates)
                ddpg_ave_rewards = self.reward_represent(ddpg_ave_rates)
                random_rewards = self.reward_represent(random_rates)
                ddpg_episode_reward += ddpg_rewards
                random_episode_reward += random_rewards

                our_rewards_all += ddpg_rewards
                ave_rewards_all += ddpg_ave_rewards
                random_rewards_all += random_rewards





                # 推荐空间删除掉已推荐项目

                self.random_recommend_space = self.random_recommend_space.tolist()
                self.ddpg_recommend_space = self.ddpg_recommend_space.tolist()
                self.ddpg_ave_recommend_space = self.ddpg_ave_recommend_space.tolist()

                self.random_recommend_space.remove(random_item_id)
                self.ddpg_recommend_space.remove(ddpg_item_id)
                self.ddpg_ave_recommend_space.remove(ddpg_ave_item_id)

                self.random_recommend_space = numpy.array(self.random_recommend_space)
                self.ddpg_recommend_space = numpy.array(self.ddpg_recommend_space)
                self.ddpg_ave_recommend_space = numpy.array(self.ddpg_ave_recommend_space)

                # 推荐历史改变
                # 推荐历史改变（即状态发生改变）
                if ddpg_ave_rates>3:
                    self.ddpg_ave_history_dict_all[user_id]["movie_id"].append(ddpg_ave_item_id)
                    ave_positive_all+=1
                self.ddpg_ave_history_dict_all[user_id]["rates"].append(ddpg_ave_rates)

                if ddpg_rates>3:
                    self.ddpg_history_dict_all[user_id]["movie_id"].append(ddpg_item_id)
                    self.ddpg_history_dict_all[user_id]["actions"].append(ddpg_action)
                    self.ddpg_history_dict_all[user_id]["second"].append(second_info)
                    self.ddpg_history_dict_all[user_id]["third"].append(third_info)
                    our_positive_all+=1
                self.ddpg_history_dict_all[user_id]["rates"].append(ddpg_rates)

                if random_rates>3:
                    random_positive_all+=1

                self.random_history_dict_all[user_id]["movie_id"].append(random_item_id)
                self.random_history_dict_all[user_id]["rates"].append(random_rates)
                # 下一状态是为了获取某策略下一个动作，从而获取critic评分的
                # next_state,历史改变了状态就改变了，直接获取最新状态即可



                if top_k:
                    ddpg_rewards = numpy.sum(ddpg_rewards)
                ddpg_next_state = self.get_state(user_id)
                self.ddpg_buffer.append(ddpg_state, ddpg_action, ddpg_rewards, ddpg_next_state, done)
                ddpg_ave_next_state = self.get_state_ave(user_id)
                self.ddpg_ave_buffer.append(ddpg_ave_state, ddpg_ave_action, ddpg_ave_rewards, ddpg_ave_next_state,
                                            done)

                # 把经历加入buffer里有，之后训练网络的时候进行经历重放
                if self.ddpg_buffer.crt_idx > self.batch_size or self.ddpg_buffer.is_full :
                    ddpg_q_loss += self.train_network()
                if self.ddpg_ave_buffer.crt_idx > self.batch_size or self.ddpg_ave_buffer.is_full :
                    ddpg_ave_q_loss += self.train_ave_network()

                ddpg_mean_action += numpy.sum(ddpg_action[0]) / (len(ddpg_action[0]))
                steps += 1
                print(
                    f'recommended items : {steps},ddpg_reward : {ddpg_rewards:+},epsilon : {self.epsilon:0.3f}',
                    end='\r')
                # 是否应该结束推荐
                if steps >= 20:
                    done = True
                # 计算一次经历的推荐精度
                if done:
                    # # 通过推荐系统历史记录算指标
                    # ddpg_positive_precision_mean,ddpg_rewards_precision_mean,random_positive_precision_mean,\
                    # random_rewards_precision_mean,ddpg_ave_positive_precision_mean,ddpg_ave_rewards_precision_mean = \
                    #     self.episode_indicator\
                    #         (user_id,step_all,ddpg_positive_precision_mean,
                    #          ddpg_rewards_precision_mean,random_positive_precision_mean,
                    #          random_rewards_precision_mean,ddpg_ave_positive_precision_mean,ddpg_ave_rewards_precision_mean)
                    #
                    self.train_history_dict["precision_mean"]["our_positive_precision_mean"].append(our_positive_all*100/our_in_all)
                    # self.train_history_dict["precision_mean"]["our_rewards_precision_mean"].append(our_rewards_all*100/space_rewards_all)
                    self.train_history_dict["precision_mean"]["ave_positive_precision_mean"].append(ave_positive_all*100/ave_in_all)
                    # self.train_history_dict["precision_mean"]["ave_rewards_precision_mean"].append(ave_rewards_all*100/space_rewards_all)
                    self.train_history_dict["precision_mean"]["random_positive_precision_mean"].append(random_positive_all*100/random_in_all)
                    # self.train_history_dict["precision_mean"]["random_rewards_precision_mean"].append(random_rewards_all*100/space_rewards_all)

                    train_image['rewards_all']['ours_1700'].append(our_rewards_all)
                    train_image['rewards_all']['ave_3000'].append(ave_rewards_all)
                    train_image['rewards_all']['random'].append(random_rewards_all)

                    train_image['positive_all']['ours_1700'].append(our_positive_all)
                    train_image['positive_all']['ave_3000'].append(ave_positive_all)
                    train_image['positive_all']['random'].append(random_positive_all)








                    print("推荐项目数量",steps)
                    print(f'{episode}/{max_episode_num}')
                    print(f' ddpg_q_loss : {ddpg_q_loss / steps}, '
                          f' ddpg_ave_q_loss : {ddpg_ave_q_loss / steps}, '
                        f'ddpg_mean_action : {ddpg_mean_action / steps}')
                    print()
            if (episode + 1) % 5 == 0:
                plot(self.train_history_dict["precision_mean"], r"./train_image_our1700_ave3000_noview_rates2/precision/",
                     "precision_mean_" + str(episode))
                plot(train_image['rewards_all'], r"./train_image_our1700_ave3000_noview_rates2/rewards_all/",
                     "rewards_all_" + str(episode))
                plot(train_image['positive_all'], r"./train_image_our1700_ave3000_noview_rates2/positive_all/",
                     "positive_all_" + str(episode))

            if (episode + 1) % 100 == 0:
                self.ddpg_actor.save_weights(r'actor_critic_weights/our1700_ave3000_noview_rates2\actor_' + str(episode + 1) + '.h5')
                self.ddpg_critic.save_weights(r'actor_critic_weights/our1700_ave3000_noview_rates2\critic_' + str(episode + 1) + '.h5')
                self.ddpg_ave_actor.save_weights(r'actor_critic_weights/our1700_ave3000_noview_rates2\ave_actor_' + str(episode + 1) + '.h5')
                self.ddpg_ave_critic.save_weights(r'actor_critic_weights/our1700_ave3000_noview_rates2\ave_critic_' + str(episode + 1) + '.h5')


    def get_state_ave_by_list(self,list,user_id):
        # 填充向量
        complex_movies_eb = numpy.ones((100, 100))
        movie_eb = self.embedding_network.get_layer('movie_embedding')(numpy.array(list)[-100:])
        complex_movies_eb = numpy.concatenate([complex_movies_eb, movie_eb], axis=0)
        complex_movies_eb = complex_movies_eb[-100:, :]

        complex_movies_eb = complex_movies_eb.reshape(10, 1000)
        # embedding向量计算
        user_embedding_tensor = self.embedding_network.get_layer('user_embedding')(numpy.array(user_id))
        user_embedding_tensor_1000 = user_embedding_tensor

        for i in range(9):
            user_embedding_tensor_1000 = numpy.concatenate([user_embedding_tensor_1000, user_embedding_tensor], axis=0)

        ## 组合成状态向量
        ddpg_ave_state = self.state_represent_network([numpy.expand_dims(user_embedding_tensor_1000, axis=0),
                                                       numpy.expand_dims(complex_movies_eb, axis=0)])

        return ddpg_ave_state
    def get_state_ave_by_list_n(self,list,user_id):
        movie_eb = self.embedding_network.get_layer('movie_embedding')(numpy.array(list)[-30:])
        movie_eb = numpy.array(movie_eb)
        movie_eb = movie_eb.reshape(1, 3000)

        return movie_eb

    def real_act_train_n(self,hitory,top_k=10):
        # 重置target网络
        self.ddpg_actor.update_target_network()
        self.ddpg_critic.update_target_network()
        user_num = len(self.env.users_history_dict)
        hh = numpy.load(r"data/data_ac_train/users_history_dict_train.npy", allow_pickle=True).item()
        ddpg_ave_q_loss = 0
        for episode in range(user_num):
            user_id,done = self.env.reset(episode+1)
            history_real = hh[user_id]
            print(user_id)
            movie_id_state = []
            state_exit = False
            for i in range(len(history_real)):
                if i==len(history_real)-1:
                    done=True
                if len(movie_id_state)>29:
                    state_exit = True
                    state = self.get_state_ave_by_list_n(movie_id_state,user_id)
                    action = self.embedding_network.get_layer('movie_embedding')(numpy.array(history_real[i][0]))
                    action_ = self.ddpg_ave_actor.act(state)
                    reward = (history_real[i][1]-3)/2
                if history_real[i][1] > 3:
                    movie_id_state.append(history_real[i][0])
                if len(movie_id_state)>29 and state_exit==True:
                    next_state = self.get_state_ave_by_list_n(movie_id_state,user_id)
                    self.ddpg_ave_buffer.append(state, action,action_, reward, next_state,done)
                if self.ddpg_ave_buffer.crt_idx > self.batch_size or self.ddpg_ave_buffer.is_full:
                    ddpg_ave_q_loss += self.train_ave_network()
            if state_exit==True:
                print(action_)
            if (episode + 1) % 100 == 0:
                self.ddpg_ave_critic.save_weights(r'offline/ave_critic_' + str(episode + 1) + '.h5')
                self.ddpg_ave_actor.save_weights(r'offline/ave_actor_' + str(episode + 1) + '.h5')
                print(ddpg_ave_q_loss)
                print(episode + 1)
    def real_act_train(self,hitory,top_k=10):
        # 重置target网络
        self.ddpg_actor.update_target_network()
        self.ddpg_critic.update_target_network()
        user_num = len(self.env.users_history_dict)
        hh = numpy.load(r"data/data_ac_train/users_history_dict_train.npy", allow_pickle=True).item()
        ddpg_ave_q_loss = 0
        for episode in range(user_num):
            user_id,done = self.env.reset(episode+1)
            history_real = hh[user_id]
            print(user_id)
            movie_id_state = []
            for i in range(len(history_real)):
                if i==len(history_real)-1:
                    done=True
                state = self.get_state_ave_by_list(movie_id_state,user_id)
                action = self.embedding_network.get_layer('movie_embedding')(numpy.array(history_real[i][0]))
                action_ = self.ddpg_ave_actor.act(state)
                reward = (history_real[i][1]-3)/2
                if history_real[i][1] > 3:
                    movie_id_state.append(history_real[i][0])
                next_state = self.get_state_ave_by_list(movie_id_state,user_id)
                self.ddpg_ave_buffer.append(state, action,action_, reward, next_state,done)
                if self.ddpg_ave_buffer.crt_idx > self.batch_size or self.ddpg_ave_buffer.is_full:
                    ddpg_ave_q_loss += self.train_ave_network()
            if (episode + 1) % 100 == 0:
                self.ddpg_ave_critic.save_weights(r'critic_train_only/ave_critic_' + str(episode + 1) + '.h5')
                self.ddpg_ave_actor.save_weights(r'critic_train_only/ave_actor_' + str(episode + 1) + '.h5')
                print(episode + 1)





    # 单独训练critic，把历史记录当经历拟合状态动作值函数
    def train_critic_only(self, max_episode_num, top_k=False, load_model=False):
        # 重置target网络
        self.ddpg_actor.update_target_network()
        self.ddpg_critic.update_target_network()

        print(load_model)
        # 训练记录
        ddpg_episodic_precision_history = []
        random_episodic_precision_history = []
        ddpg_real_episodic_precision_history = []


        for episode in range(max_episode_num):
            # episodic reward 每轮置零
            ddpg_episode_reward = 0
            random_episode_reward = 0
            ddpg_correct_count = 0
            random_correct_count = 0

            steps = 0
            ddpg_q_loss = 0
            ddpg_mean_action = 0
            '''
            # Environment 从环境里随机获得一个用户以及历史，即每轮选一个模拟用户进行一次完整的经历
            # recommender进行推荐，user反馈reward（由评分构成，可复合多种指标进行设置）
            '''
            user_id,done = self.env.reset()

            """单个用户"""
            # Ddpg方式训练时的推荐空间
            self.ddpg_recommend_space = copy.deepcopy(self.env.recommend_space_train)
            # 随机方式训练时的推荐空间
            self.random_recommend_space = copy.deepcopy(self.env.recommend_space_train)

            # Ddpg方式训练时的历史记录
            # self.ddpg_recommended_items = copy.deepcopy(self.env.recommended_items_init)
            # 随机方式训练时的历史记录
            # self.random_recommended_items = copy.deepcopy(self.env.recommended_items_init)


            print("当前用户",user_id)
            print("推荐空间长度",len(self.env.recommend_space_train))
            print("推荐空间里的好评记录数",self.env.positive_rates_count)
            # 全好评和全差评的视作无效用户不训练
            while not done and self.env.positive_rates_count != 0 and self.env.positive_rates_count!=len(self.env.recommend_space_train):
                # 做出推荐
                ddpg_item_id, ddpg_action, ddpg_state = self.recommend_item_ddpg(user_id=user_id,state_before=numpy.zeros((1, 100)))
                random_item_id = self.recommend_item_random()

                #与用户交互得到评价
                random_recommend_rates = self.env.step(random_item_id, top_k=top_k)
                ddpg_rates = self.env.step(ddpg_item_id, top_k=top_k)

                # 推荐空间删除掉已推荐项目
                self.random_recommend_space = self.random_recommend_space[self.random_recommend_space != random_item_id]
                self.ddpg_recommend_space = self.ddpg_recommend_space[self.ddpg_recommend_space != ddpg_item_id]

                """
                应该让状态表示模块来维护history
                """

                # 推荐历史改变
                self.random_recommended_items['item_ids_list'].append(random_item_id)
                self.random_recommended_items['rates_list'].append(random_recommend_rates)
                self.ddpg_recommended_items['item_ids_list'].append(ddpg_item_id)
                self.ddpg_recommended_items['rates_list'].append(ddpg_rates)

                # 奖励计算
                ddpg_rewards = self.reward_represent(ddpg_rates)
                random_rewards = self.reward_represent(random_recommend_rates)

                if ddpg_rewards > 0:
                    ddpg_correct_count += 1
                if random_rewards > 0:
                    random_correct_count += 1

                ddpg_episode_reward += ddpg_rewards
                random_episode_reward += random_rewards

                if top_k:
                    ddpg_rewards = numpy.sum(ddpg_rewards)
                # get next_state,获取下一状态

                # 把经历加入buffer里有，之后训练网络的时候进行经历重放
                # self.ddpg_buffer.append(ddpg_state, ddpg_action, ddpg_rewards, ddpg_next_state, done)
                if self.ddpg_buffer.crt_idx > 1 or self.ddpg_buffer.is_full:
                    ddpg_q_loss += self.train_network()
                ddpg_mean_action += numpy.sum(ddpg_action[0]) / (len(ddpg_action[0]))
                steps += 1
                print(
                    f'recommended items : {steps},ddpg_reward : {ddpg_rewards:+},epsilon : {self.epsilon:0.3f}',
                    end='\r')
                # 是否应该结束推荐
                if steps >= self.env.positive_rates_count:
                    done = True
                # 计算一次经历的推荐精度
                if done:
                    print("ddpg推荐被好评：",ddpg_correct_count)
                    print("随机推荐被好评：",random_correct_count)
                    # 积极个数推荐精度
                    # 回报率推荐精度
                    ddpg_precision = int(ddpg_episode_reward / (self.env.positive_rewards_sum ) * 100)
                    random_precision = int(random_episode_reward / (self.env.positive_rewards_sum ) * 100)
                    # 推荐精度之差
                    ddpg_precision_real = ddpg_precision - random_precision

                    print("推荐项目数量",steps)
                    print(
                        f'{episode}/{max_episode_num}, ddpg_precision : {ddpg_precision:2}%,'
                        f' random_precision : {random_precision:2}%, ')
                    print(f'ddpg_reward:{ddpg_episode_reward},random_reward:{random_episode_reward},all_rewards:{self.env.positive_rewards_sum}')
                    print(f' ddpg_q_loss : {ddpg_q_loss / steps}, '
                        f'ddpg_mean_action : {ddpg_mean_action / steps}')
                    ddpg_episodic_precision_history.append(ddpg_precision)
                    random_episodic_precision_history.append(random_precision)
                    ddpg_real_episodic_precision_history.append(ddpg_precision_real)
                    print()
                    self.monitor.record_dict['ddpg_precision_real'] = ddpg_real_episodic_precision_history
                    self.monitor.comparison_record_dict['random_precision'] = random_episodic_precision_history
                    self.monitor.comparison_record_dict['ddpg_precision'] = ddpg_episodic_precision_history


            if (episode + 1) % 10 == 0:
               self.monitor.plot(episode+1,10)
            if (episode + 1) % 200 == 0:
                self.monitor.plot(episode + 1, 200)
            if (episode + 1) % 100 == 0:
                self.ddpg_actor.save_weights(r'actor_critic_weights\ddpg_new\actor' + str(episode + 1) + '.h5')
                self.ddpg_critic.save_weights(r'actor_critic_weights\ddpg_new\critic' + str(episode + 1) + '.h5')


    def evaluation(self, max_user_id, top_k=False, load_model=False):
        print(load_model)
        steps = 0
        ddpg_ndcg = 0
        random_ndcg = 0

        ddpg_precision = 0
        random_precision = 0
        for episode in range(max_user_id):
            # episodic reward 每轮置零
            while episode+1 not in self.env.available_users:
                  episode += 1

            user_id, short_term_ids, done = self.env.reset(episode+1)
            # Ddpg方式训练时的推荐空间
            self.ddpg_recommend_space = copy.deepcopy(self.env.recommend_space_train)
            # 随机方式训练时的推荐空间
            self.random_recommend_space = copy.deepcopy(self.env.recommend_space_train)

            # Ddpg方式训练时的历史记录
            self.ddpg_recommended_items = copy.deepcopy(self.env.recommended_items_init)
            # 随机方式训练时的历史记录
            self.random_recommended_items = copy.deepcopy(self.env.recommended_items_init)
            print("当前用户", user_id)
            # print("推荐空间",self.env.recommend_space_train)
            print("推荐空间长度", len(self.env.recommend_space_train))
            print("推荐空间里的好评记录数", self.env.positive_rates_count)
            while not done :
                # 做出推荐
                ddpg_item_id, ddpg_action, ddpg_state = self.recommend_item_ddpg(user_id=user_id,top_k=10)
                steps += 1
                random_item_id = self.recommend_item_random(top_k=10)

                # 与用户交互得到评价
                random_rates = self.env.step(random_item_id, top_k=top_k)
                ddpg_rates = self.env.step(ddpg_item_id, top_k=top_k)

                # 理想状态下的top——k推荐应有的评价
                rates_list = self.env.recommend_space_train_rates
                rates_list.sort(reverse=True)
                ideal_list = rates_list[:top_k]

                ddpg_ndcg = self._ndcg_calculate(ddpg_rates,ideal_list)/steps+ddpg_ndcg*(steps-1)/steps
                random_ndcg = self._ndcg_calculate(random_rates,ideal_list)/steps+random_ndcg*(steps-1)/steps

                ddpg_precision = (self._cg_calculate(ddpg_rates)/self._cg_calculate(ideal_list))/steps + ddpg_precision*(steps-1)/steps
                random_precision = (self._cg_calculate(random_rates)/self._cg_calculate(ideal_list))/steps + random_precision*(steps-1)/steps
                done = True
                if done:
                    print("ddpg推荐平均精度：", ddpg_precision)
                    print("随机推荐平均精度：", random_precision)
                    print("ddpg推荐平均ndcg：", ddpg_ndcg)
                    print("随机推荐平均ndcg：", random_ndcg)
                    print("推荐用戶数量", steps)
                    print()


    def _cg_calculate(self,rate_list):
        result = 0
        for rate in rate_list:
            result += rate
        return result

    def _dcg_calculate(self,rate_list):
        result = 0
        i = 1
        for rate in rate_list:
            result += rate/(numpy.log2(i+1))
            i += 1
        return result

    def _ndcg_calculate(self,real_list,ideal_list):
        dcg = self._dcg_calculate(real_list)
        idcg = self._dcg_calculate(ideal_list)
        return dcg/idcg

    # 通过推荐系统历史记录算指标
    def episode_indicator(self, user_id,step_all,ddpg_positive_precision_mean,
                          ddpg_rewards_precision_mean,random_positive_precision_mean,
                          random_rewards_precision_mean,ddpg_ave_positive_precision_mean,
                          ddpg_ave_rewards_precision_mean):
        ddpg_rate_list = self.ddpg_history_dict_all[user_id]["rates"]
        ddpg_ave_rate_list = self.ddpg_ave_history_dict_all[user_id]["rates"]
        random_rate_list = self.random_history_dict_all[user_id]["rates"]
        steps = len(ddpg_rate_list)
        # 被好评数和总奖励
        ddpg_correct_count, ddpg_episode_reward = self.get_rates_standards(ddpg_rate_list)
        ddpg_ave_correct_count, ddpg_ave_episode_reward = self.get_rates_standards(ddpg_ave_rate_list)
        random_correct_count, random_episode_reward = self.get_rates_standards(random_rate_list)
        # 积极个数推荐精度
        ddpg_positive_precision = int(ddpg_correct_count / (self.env.positive_rates_count) * 100)
        ddpg_ave_positive_precision = int(ddpg_ave_correct_count / (self.env.positive_rates_count) * 100)
        random_positive_precision = int(random_correct_count / (self.env.positive_rates_count) * 100)
        # 回报率推荐精度
        ddpg_rewards_precision = int(ddpg_episode_reward / (self.env.positive_rewards_sum) * 100)
        ddpg_ave_rewards_precision = int(ddpg_ave_episode_reward / (self.env.positive_rewards_sum) * 100)
        random_rewards_precision = int(random_episode_reward / (self.env.positive_rewards_sum) * 100)

        # 训练指标记录
        self.train_history_dict["steps"].append(steps)
        self.train_history_dict["precision"]["ddpg_positive_precision"].append(ddpg_positive_precision)
        self.train_history_dict["precision"]["ddpg_rewards_precision"].append(ddpg_rewards_precision)
        self.train_history_dict["precision"]["ddpg_ave_positive_precision"].append(ddpg_ave_positive_precision)
        self.train_history_dict["precision"]["ddpg_ave_rewards_precision"].append(ddpg_ave_rewards_precision)
        self.train_history_dict["precision"]["random_positive_precision"].append(random_positive_precision)
        self.train_history_dict["precision"]["random_rewards_precision"].append(random_rewards_precision)
        # 平均精度累进更新
        ddpg_positive_precision_mean = get_mean(step_all,steps,ddpg_positive_precision_mean,ddpg_positive_precision)
        ddpg_rewards_precision_mean = get_mean(step_all,steps,ddpg_rewards_precision_mean,ddpg_rewards_precision)
        ddpg_ave_positive_precision_mean = get_mean(step_all, steps, ddpg_ave_positive_precision_mean, ddpg_ave_positive_precision)
        ddpg_ave_rewards_precision_mean = get_mean(step_all, steps, ddpg_ave_rewards_precision_mean, ddpg_ave_rewards_precision)
        random_positive_precision_mean = get_mean(step_all,steps,random_positive_precision_mean,random_positive_precision)
        random_rewards_precision_mean = get_mean(step_all,steps,random_rewards_precision_mean,random_rewards_precision)
        self.train_history_dict["precision_mean"]["ddpg_positive_precision_mean"].append(ddpg_positive_precision_mean)
        self.train_history_dict["precision_mean"]["ddpg_rewards_precision_mean"].append(ddpg_rewards_precision_mean)
        self.train_history_dict["precision_mean"]["ddpg_ave_positive_precision_mean"].append(ddpg_ave_positive_precision_mean)
        self.train_history_dict["precision_mean"]["ddpg_ave_rewards_precision_mean"].append(ddpg_ave_rewards_precision_mean)
        self.train_history_dict["precision_mean"]["random_positive_precision_mean"].append(random_positive_precision_mean)
        self.train_history_dict["precision_mean"]["random_rewards_precision_mean"].append(random_rewards_precision_mean)

        return ddpg_positive_precision_mean,ddpg_rewards_precision_mean,\
               random_positive_precision_mean,random_rewards_precision_mean,ddpg_ave_positive_precision_mean,ddpg_ave_rewards_precision_mean


    import matplotlib.pyplot as plt
    # 训练历史记录可视化并保存
    def train_history_view(self,episode):
        numpy.save(r"train_history_dict_movielens1M_our1700_ave3000.npy", self.train_history_dict)
        plot(self.train_history_dict["precision_mean"],r"./1700_3000_po_onlly_multiembedding/","precision_mean_"+str(episode))


#绘制一个字典（存的列表）的图像并保存
def plot(dict,save_path,save_name):
        # 画图
        for key in dict.keys():
            plt.plot(dict[key], label=key)
        plt.grid()
        plt.legend()  # 显示上面的label
        plt.title(save_name)  # 标题

        plt.savefig(r''+save_path+ save_name+'.png')
        plt.clf()



# 累进更新平均值
def get_mean(steps_before,steps_now,mean_before,now):
    return (mean_before*steps_before+now*steps_now)/(steps_before+steps_now)























