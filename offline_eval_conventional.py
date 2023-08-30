import random
import numpy.random
import numpy
from matplotlib import pyplot as plt

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
def eval_recommend_best(eval_dict):
    result = {}
    for user in eval_dict:
        result[user] = [data[1] for data in eval_dict[user]]
        result[user].sort(reverse=True)
    return result
def eval_recommend_worst(eval_dict):
    result = {}
    for user in eval_dict:
        result[user] = [data[1] for data in eval_dict[user]]
        result[user].sort(reverse=False)
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
# result_best = eval_recommend_best(eval_dict=eval_dict)
# result_worst = eval_recommend_worst(eval_dict=eval_dict)
# numpy.save("./eval_record_dict/result_best.npy", result_best)
# numpy.save("./eval_record_dict/result_worst.npy", result_worst)

# print()
# for k in [5, 10, 15, 20]:
#     precision_topk_mean = calculate_dict_precision(result_best, k)
#     ndcg_topk_mean = calculate_dict_ndcg(result_worst, k)

















