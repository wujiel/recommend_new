import copy
import random
from ExperienceBuffer import DrrnBuffer_offline
import numpy.random
import numpy
import tensorflow

from EmbeddingNetwork import MultiNetwork
from actor import Actor_DRRN
from critic import Critic_DRRN

train_dict = numpy.load('data/data_ac_train/users_history_dict_train.npy',allow_pickle='True').item()
eval_dict = numpy.load('data/data_ac_eval/users_history_dict_eval.npy',allow_pickle='True').item()
popular_list = numpy.load('popular_list.npy',allow_pickle='True')

embedding_network = MultiNetwork(len_users=6040, len_movies=3900, embedding_dim=100)
# 这是在build网络，传一个数据即可
out = embedding_network([numpy.zeros((1)), numpy.zeros((1))])
embedding_network.summary()
embedding_network.load_weights(r'embedding_weights/muti/multi_network_weights99000.h5')


def train_network_without_priority(buffer, actor, critic, batch_size,discount_factor):
    # 取出一个批量
    batch_states, batch_actions_sample, batch_actions, batch_rewards, \
        batch_next_states, batch_dones = buffer.sample_random(batch_size)

    # TD学习的目标值即q网络应该去拟合的label值
    td_targets = calculate_td_target(rewards=batch_rewards, dones=batch_dones, actor=actor,
                                     critic=critic, batch_next_states=batch_next_states, discount_factor=discount_factor)

    q_loss = critic.train_only([batch_actions_sample, batch_states], td_targets)
    critic.update_target_network()
    # q对a求导再a对actor的参数求导最后得到分值对actor网络参数的导数，朝着使分值增大的方向优化，传反梯度
    s_grads = critic.dq_da([batch_actions, batch_states])
    actor.train(batch_states, s_grads)
    actor.update_target_network()
    critic.update_target_network()

    return q_loss


def get_state_drrn(state_list,n,padding,embedding_network):
    # 填充向量
    # complex_movies_eb = numpy.ones((100, 100))
    movie_record_list = copy.deepcopy(state_list[-n:])
    movie_eb = embedding_network.get_layer('movie_embedding')(numpy.array(movie_record_list))
    complex_movies_eb = numpy.concatenate([padding, movie_eb], axis=0)
    complex_movies_eb = complex_movies_eb[-n:, :]
    state = complex_movies_eb.reshape(1, 100*n)
    return state

def calculate_td_target(rewards, dones,actor,critic, batch_next_states,discount_factor):
        # newQ = r+γ*nextQ
        # nextQ = t_critic(next_s,next_a)
        # next_a = t_actor(next_s)

        # 计算下一动作用target网络
        target_actions = actor.target_act(batch_next_states)
        # critic的网络和目标网络分别给下一状态动作打分以作为未来期望奖励 Double Q method
        qs = critic.network([target_actions, batch_next_states])
        target_qs = critic.target_network([target_actions, batch_next_states])
        min_qs = tensorflow.raw_ops.Min(input=tensorflow.concat([target_qs, qs], axis=1), axis=1, keep_dims=True)
        y_t = numpy.copy(min_qs)
        for i in range(min_qs.shape[0]):
            y_t[i] = rewards[i] + (1 - dones[i]) * (discount_factor * min_qs[i])
        return y_t
def train_network_offline(buffer,actor,critic,batch_size):
        # 取出一个批量
        batch_states, batch_actions_sample,batch_actions, batch_rewards, \
        batch_next_states, batch_dones, weight_batch, index_batch = buffer.sample(batch_size)

        # TD学习的目标值即q网络应该去拟合的label值
        td_targets = calculate_td_target(rewards=batch_rewards, dones=batch_dones,actor=actor,
                                         critic=critic,batch_next_states=batch_next_states,discount_factor=0.9)
        # 更新容器的优先级
        for (p, i) in zip(td_targets, index_batch):
            buffer.update_priority(abs(p[0]) + (1e-6), i)

        q_loss = critic.train([batch_actions_sample, batch_states], td_targets, weight_batch)

        # q对a求导再a对actor的参数求导最后得到分值对actor网络参数的导数，朝着使分值增大的方向优化，传反梯度
        s_grads = critic.dq_da([batch_actions, batch_states])
        actor.train(batch_states, s_grads)
        actor.update_target_network()
        critic.update_target_network()

        return q_loss

def train_drrn(train_dict,embedding_network,n):
    actor = Actor_DRRN(state_dim=n*100, action_dim=100, learning_rate=0.001,
                           target_network_update_rate=0.001)
    # critic
    critic = Critic_DRRN(state_dim=100*n, action_dim=100, learning_rate=0.001,
                             target_network_update_rate=0.001)

    buffer = DrrnBuffer_offline(buffer_size=1000000,state_dim=n*100,action_dim=100)
    batch_size = 32
    q_loss = 0
    for user in train_dict:
        user_state_sample = []
        # zero_padding = complex_movies_eb = numpy.zeros((n, 100))
        # ones_padding = complex_movies_eb = numpy.ones((n, 100))
        popular_padding = embedding_network.get_layer('movie_embedding')(numpy.array(popular_list[:n]))
        done = False
        for i in range(len(train_dict[user])):
            # if i == len(train_dict[user]):
            #     done = True
            state = get_state_drrn(state_list=user_state_sample,n=n,padding=popular_padding,embedding_network=embedding_network)
            action_sample_movie_id = train_dict[user][i][0]
            action_sample = embedding_network.get_layer('movie_embedding')(numpy.array(action_sample_movie_id))
            action = actor.act(state)
            rate = train_dict[user][i][1]
            reward = (rate - 3) / 2
            if rate > 3:
                user_state_sample.append(action_sample_movie_id)
            next_state = get_state_drrn(state_list=user_state_sample,n=n,padding=popular_padding,embedding_network=embedding_network)
            buffer.append(state, action_sample, action, reward, next_state, done)
            if buffer.crt_idx > batch_size or buffer.is_full:
                q_loss += train_network_offline(buffer=buffer,actor=actor,critic=critic,batch_size=batch_size)
        print(user)
        if user%100 == 0:
            actor.save_weights(r'actor_weights/actor_episode' + str(user)+'_statedim_'+str(n) + '.h5')
            critic.save_weights(r'critic_weights/critic_episode' + str(user)+'_statedim_'+str(n) + '.h5')
            print(q_loss)
            print(user)
    return "finish"


'''
生成一个训练所需的容器，以及actor和critic

'''
def sample_create(train_dict,embedding_network,n):
    actor = Actor_DRRN(state_dim=n*100, action_dim=100, learning_rate=0.001,
                           target_network_update_rate=0.001)
    # critic
    critic = Critic_DRRN(state_dim=100*n, action_dim=100, learning_rate=0.001,
                             target_network_update_rate=0.001)
    # 存放经验（根据用户的真实交互）
    buffer = DrrnBuffer_offline(buffer_size=1000000,state_dim=n*100,action_dim=100)
    for user in train_dict:
        user_state_sample = []
        # zero_padding = complex_movies_eb = numpy.zeros((n, 100))
        # ones_padding = complex_movies_eb = numpy.ones((n, 100))、

        # 使用最受欢迎的电影特征向量作为填充
        popular_padding = embedding_network.get_layer('movie_embedding')(numpy.array(popular_list[:n]))
        done = False
        for i in range(len(train_dict[user])):
            # 状态获取
            state = get_state_drrn(state_list=user_state_sample,n=n,padding=popular_padding,embedding_network=embedding_network)
            # 在此状态下用户的真实交互，也就是历史记录中的电影id
            action_sample_movie_id = train_dict[user][i][0]
            # 取出此电影id的特征向量，这个向量将和state向量组成一个状态动作对供critic打分并进行优化
            action_sample = embedding_network.get_layer('movie_embedding')(numpy.array(action_sample_movie_id))
            # 在此状态下actor的输出，也就是actor网络的预测，只需要放进经验容器里，供critic进行优化
            action = actor.act(state)
            # 如果rate大于3，则状态改变
            rate = train_dict[user][i][1]
            # 计算reward
            reward = (rate - 3) / 2
            if rate > 3:
                user_state_sample.append(action_sample_movie_id)
            # 计算下一状态
            next_state = get_state_drrn(state_list=user_state_sample,n=n,padding=popular_padding,embedding_network=embedding_network)
            # 加入经验容器里,直至遍历完训练集所有的记录，约80万条
            buffer.append(state, action_sample, action, reward, next_state, done)
            if buffer.is_full:
                print('full')
        if user%100 == 0:
            print(user)
    return actor,critic,buffer

'''
最大的区别是这里的状态动作对是真实用户的示例，用以优化critic
例：
用户1有如下历史记录
[(3186, 4), (1721, 4), (1270, 5), (1022, 5), (2340, 3), (1836, 5), (3408, 4), (1207, 4)]
假设滑动窗口为2，那么第一个无填充的状态即[3216,1721]这两部影片的特征向量，

若用模拟环境学习，actor产生一个动作向量action，那么我们得到一个状态动作对（[3216,1721]这两部影片的特征向量，action向量）用以优化critic
这里的问题是action向量并不是一条真正的影片特征记录，而且用户历史并无此状态

若用示例学习，我们把1270这条真实记录作为此状态的动作向量，那么我们得到一个状态动作对（[3216,1721]这两部影片的特征向量，1270的特征向量）用以优化critic
同时我们依然让actor产生一个动作向量action并加入经验容器，但它不做为状态动作对优化critic，只是用以梯度计算好让critic可以对actor进行优化

本质上来说，两者的区别只是状态动作对的替换，即，用真实的状态动作对替换了模拟环境下的状态动作对

'''
def train_drrn_without_priority(train_dict,embedding_network,n,discount_factor,max_episodes,batch_size):
    actor,critic,buffer = sample_create(train_dict,embedding_network,n)
    buffer_dict = {'buffer':buffer}
    numpy.save(r'buffer/drrn_buffer' + '_statedim_' + str(n) + '.npy',buffer_dict)

    q_loss_all = 0
    for episode in range(max_episodes):
        q_loss =  train_network_without_priority(buffer=buffer, actor=actor, critic=critic, batch_size=batch_size,discount_factor=discount_factor)
        q_loss_all += q_loss
        if episode % 1000 == 0:
            actor.save_weights(r'actor_weights_random_sample/actor_episode' + str(episode) + '_statedim_' + str(n) + '.h5')
            critic.save_weights(r'critic_weights_random_sample/critic_episode' + str(episode) + '_statedim_' + str(n) + '.h5')
            # print('q_loss:',q_loss)
            # print('q_loss_all:',q_loss_all)
            print('episode:',episode)





# for n in [10,15,20,25,30,35,40,45,50,55,60]:
#     s = train_drrn_without_priority(train_dict=train_dict,
#                                     embedding_network=embedding_network,n = n,discount_factor=0.9,max_episodes=1000000,batch_size=128)

# for n in [20,25,30,35,40,45,50,55,60]:
#     s = train_drrn(train_dict=train_dict,embedding_network=embedding_network,n = n)









