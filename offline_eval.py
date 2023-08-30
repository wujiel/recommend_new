import copy
import random
import numpy.random
import numpy
from matplotlib import pyplot as plt

from EmbeddingNetwork import MultiNetwork
from dict_eval import calculate_dict_precision, calculate_dict_ndcg
from offline_train_drrn import get_state_drrn, popular_list
from actor import Actor_DRRN


def plot(dict, save_path, save_name):
    # 画图
    for key in dict.keys():
        plt.plot(dict[key], label=key)
    plt.grid()
    plt.legend()  # 显示上面的label
    plt.title(save_name)  # 标题
    plt.savefig(r'' + save_path + save_name + '.png')
    plt.clf()


train_dict = numpy.load('data/data_ac_train/users_history_dict_train.npy',allow_pickle='True').item()
eval_dict = numpy.load('data/data_ac_eval/users_history_dict_eval.npy',allow_pickle='True').item()
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
def eval_recommend_random(eval_dict):
    result = {}
    for user in eval_dict:
        result[user] = [data[1] for data in eval_dict[user]]
        random.shuffle(result[user])
    return result
def eval_recommend_popularity(train_dict,eval_dict):
    result = {}
    user_num = len(train_dict)
    mat, mat_01 = getMat(train_dict, user_num + 1, 4000)
    add = numpy.ones(user_num + 1)
    hot_sore_all = numpy.mean(mat, axis=0)
    hot_sore_all_01 = numpy.dot(add, mat_01)
    hot_rank = list(hot_sore_all.argsort()[::-1])
    hot_rank_01 = list(hot_sore_all_01.argsort()[::-1])
    hot_rank_dict = {}
    rank = 1
    for i in hot_rank:
        hot_rank_dict[i] = rank
        rank += 1
    hot_rank_dict_01 = {}
    rank = 1
    for i in hot_rank_01:
        hot_rank_dict_01[i] = rank
        rank += 1
    numpy.save("popular_list.npy", hot_rank_01)
    for user in eval_dict:
        item_set = [data[0] for data in eval_dict[user]]
        user_dict = {}
        for tuple_ in eval_dict[user]:
            user_dict[tuple_[0]] = tuple_[1]
        list_hot_rank = []
        for item in item_set:
            rank = hot_rank_dict_01[item]
            list_hot_rank.append(rank)
        list_hot_rank.sort()
        i_id = []
        popular_item_rate = []
        for rank in list_hot_rank:
            i_id.append(hot_rank_01[rank - 1])
            popular_item_rate.append(user_dict[hot_rank_01[rank - 1]])
        result[user] = popular_item_rate
    return result
# result_random = eval_recommend_random(eval_dict=eval_dict)
# result_popularity = eval_recommend_popularity(train_dict=train_dict,eval_dict=eval_dict)
# numpy.save("./eval_record_dict/result_popularity.npy", result_popularity)
# numpy.save("./eval_record_dict/result_random.npy", result_random)


embedding_network = MultiNetwork(len_users=6040, len_movies=3900, embedding_dim=100)
# 这是在build网络，传一个数据即可
out = embedding_network([numpy.zeros((1)), numpy.zeros((1))])
embedding_network.summary()
embedding_network.load_weights(r'embedding_weights/muti/multi_network_weights99000.h5')

def get_state_by_list_drrn(list):
    movie_eb = embedding_network.get_layer('movie_embedding')(numpy.array(list)[-30:])
    movie_eb = numpy.array(movie_eb)
    movie_eb = movie_eb.reshape(1, 3000)
    return movie_eb

def eval_recommend_drrn_nopadding(actor,train_dict,eval_dict):
    K = 0
    result = {}
    eval_step = {}
    for user in eval_dict:
        user_dict = {}
        eval_list = []
        for tuple_ in eval_dict[user]:
            user_dict[tuple_[0]] = tuple_[1]
            eval_list.append(tuple_[0])
        user_dict[0] = eval_list
        eval_step[user] = user_dict

    for user in train_dict:
        user_eval_list = copy.deepcopy(eval_step[user][0])
        user_state = []
        predict = []
        for tuple_ in train_dict[user]:
            if tuple_[1] > 3:
                user_state.append(tuple_[0])
        if len(user_state) > 30:
            for i in range(len(user_eval_list)):
                K +=1
                print(user,' ',K)
                state = get_state_by_list_drrn(user_state)
                action = actor(state)
                action = numpy.array(action)
                action_T = action.reshape(100, 1)
                recommend_space_ebs = embedding_network.get_layer('movie_embedding')(numpy.array(user_eval_list))
                rank = numpy.dot(recommend_space_ebs, action_T)
                rank = rank.transpose(1, 0)
                item_idx = user_eval_list[numpy.argmax(rank)]
                user_eval_list.remove(item_idx)
                predict.append(eval_step[user][item_idx])
                if eval_step[user][item_idx] > 3:
                    user_state.append(item_idx)
            result[user] = predict
    return result

def eval_recommend_drrn(actor,train_dict,eval_dict,n,embedding_network):
    popular_padding = embedding_network.get_layer('movie_embedding')(numpy.array(popular_list[:n]))
    result = {}
    eval_step = {}
    for user in eval_dict:
        user_dict = {}
        eval_list = []
        for tuple_ in eval_dict[user]:
            user_dict[tuple_[0]] = tuple_[1]
            eval_list.append(tuple_[0])
        user_dict[0] = eval_list
        eval_step[user] = user_dict

    for user in train_dict:
        user_eval_list = copy.deepcopy(eval_step[user][0])
        user_state = []
        predict = []
        for tuple_ in train_dict[user]:
            if tuple_[1] > 3:
                user_state.append(tuple_[0])
        for i in range(len(user_eval_list)):
            state = get_state_drrn(state_list=user_state,n=n,padding=popular_padding,embedding_network=embedding_network)
            action = actor.act(state)
            action = numpy.array(action)
            action_T = action.reshape(100, 1)
            recommend_space_ebs = embedding_network.get_layer('movie_embedding')(numpy.array(user_eval_list))
            rank = numpy.dot(recommend_space_ebs, action_T)
            rank = rank.transpose(1, 0)
            item_idx = user_eval_list[numpy.argmax(rank)]
            user_eval_list.remove(item_idx)
            predict.append(eval_step[user][item_idx])
            if eval_step[user][item_idx] > 3:
                user_state.append(item_idx)
        result[user] = predict
        if user%100 == 0:
            print(user)
    return result







episode_list = []
episode = 100
while episode < 6001:
    episode_list.append(episode)
    episode += 100


for n in [10,15,20]:
    drrn_precision = {}
    drrn_ndcg = {}
    for k in [5, 10, 15, 20]:
        drrn_precision[k] = []
        drrn_ndcg[k] = []
    for episode in episode_list:
        actor = Actor_DRRN(state_dim=n * 100, action_dim=100, learning_rate=0.001,
                       target_network_update_rate=0.001)
        actor.load_weights(r'actor_weights/actor_episode' + str(episode)+'_statedim_'+str(n) + '.h5')
        result_drrn = eval_recommend_drrn(actor=actor, train_dict=train_dict, eval_dict=eval_dict,n=n,embedding_network=embedding_network)
        numpy.save(r'eval_record_dict/result_drrn_actor_episode' + str(episode)+'_statedim_'+str(n) + '.npy', result_drrn)
        for k in [5,10,15,20]:
            precision_topk_mean = calculate_dict_precision(result_drrn,k)
            ndcg_topk_mean = calculate_dict_ndcg(result_drrn,k)
            drrn_precision[k].append(precision_topk_mean)
            drrn_ndcg[k].append(ndcg_topk_mean)
    plot(drrn_precision, r"./eval_image/", "precision_mean_statedim_" + str(n))
    plot(drrn_ndcg, r"./eval_image/", "ndcg_mean_statedim_" + str(n))










