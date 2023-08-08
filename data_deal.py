import numpy
import pandas as pd
import numpy as np
import os
from Dataloader import Dataloader










'''数据处理，训练集，测试集，验证集的分割 
   embedding网络和ac网络 
'''
dataloader = Dataloader()
users_history_dict_train = numpy.load('train_history_dict_movielens1M_our1700_ave3000.npy', allow_pickle=True).item()
eval_dict = numpy.load('eval_dict_movielens1M_our1700_ave3000.npy', allow_pickle=True).item()
users_history_dict_eval = dataloader.users_history_dict_eval


def _dcg_calculate(rate_list):
    result = 0
    i = 1
    for rate in rate_list:
        result += 2**rate / (numpy.log2(i + 1))
        i += 1
    return result

def precision_calculate(rate_list,topk):
    result = 0
    for rate in rate_list:
        if rate>3:
            result += 1
    # return result/len(rate_list)
    return result/topk

def calculate_all(topk):
    predict_all = 0

    min_ = 0
    ndcg_predict_ave_self_all = 0
    ndcg_random_self_all = 0
    ndcg_popular_self_all = 0

    ndcg_predict_ave_self_all_K = 0
    ndcg_random_self_all_K = 0
    ndcg_popular_self_all_K = 0


    precision_predict_ave_all = 0
    precision_random_all = 0
    precision_popular_all = 0

    precision_predict_ave_all_K = 0
    precision_random_all_K = 0
    precision_popular_all_K = 0


    for user_id in range(1, len(eval_dict)):
        rates_predict_ave = eval_dict[user_id]['ave'][0:topk]
        rates_random = eval_dict[user_id]['random'][0:topk]
        rates_popular = eval_dict[user_id]['popular'][0:topk]
        user_history_len = len(eval_dict[user_id]['ave'])
        # if user_history_len<5:
        #     continue

        dcg_predict_ave = _dcg_calculate(rates_predict_ave)
        dcg_predict_random = _dcg_calculate(rates_random)
        dcg_predict_popular = _dcg_calculate(rates_popular)

        rates_predict_ave.sort(reverse=True)
        rates_random.sort(reverse=True)
        rates_popular.sort(reverse=True)

        idcg_predict_ave = _dcg_calculate(rates_predict_ave)
        idcg_predict_random = _dcg_calculate(rates_random)
        idcg_predict_popular = _dcg_calculate(rates_popular)

        ndcg_predict_ave_self = dcg_predict_ave / idcg_predict_ave
        ndcg_random_self = dcg_predict_random / idcg_predict_random
        ndcg_popular_self = dcg_predict_popular / idcg_predict_popular

        ndcg_predict_ave_self_all += ndcg_predict_ave_self
        ndcg_random_self_all += ndcg_random_self
        ndcg_popular_self_all += ndcg_popular_self

        precision_predict_ave = precision_calculate(rates_predict_ave, topk)
        precision_random = precision_calculate(rates_random, topk)
        precision_popular = precision_calculate(rates_popular, topk)

        precision_predict_ave_all += precision_predict_ave
        precision_random_all += precision_random
        precision_popular_all += precision_popular

        if user_history_len >= topk:
            predict_all += topk

            ndcg_predict_ave_self_all_K += ndcg_predict_ave_self
            ndcg_random_self_all_K += ndcg_random_self
            ndcg_popular_self_all_K += ndcg_popular_self

            precision_predict_ave_all_K += precision_predict_ave
            precision_random_all_K += precision_random
            precision_popular_all_K += precision_popular




        else:
            predict_all += user_history_len
            min_ += 1

    return predict_all,min_, \
        ndcg_predict_ave_self_all, ndcg_random_self_all, ndcg_popular_self_all, \
        precision_predict_ave_all, precision_random_all, precision_popular_all,\
        ndcg_predict_ave_self_all_K,ndcg_random_self_all_K,ndcg_popular_self_all_K,\
        precision_predict_ave_all_K,precision_random_all_K,precision_popular_all_K



predict_all,min_,ndcg_predict_ave_self_all,ndcg_random_self_all,\
    ndcg_popular_self_all,precision_predict_ave_all,\
    precision_random_all,precision_popular_all,\
    ndcg_predict_ave_self_all_K,ndcg_random_self_all_K,ndcg_popular_self_all_K,\
    precision_predict_ave_all_K,precision_random_all_K,precision_popular_all_K = calculate_all(5)

print('top5====================================================================')

print('predict_all:',predict_all)
print('min_:',min_)
# print('ndcg_predict_ave_self_all:',ndcg_predict_ave_self_all)
# print('ndcg_random_self_all:',ndcg_random_self_all)
# print('ndcg_popular_self_all:',ndcg_popular_self_all)
# print('precision_predict_ave_all:',precision_predict_ave_all)
# print('precision_random_all:',precision_random_all)
# print('precision_popular_all:',precision_popular_all)
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('ndcg_predict_ave_self_mean:',ndcg_predict_ave_self_all/len(eval_dict))
print('ndcg_random_self_mean:',ndcg_random_self_all/len(eval_dict))
print('ndcg_popular_self_mean:',ndcg_popular_self_all/len(eval_dict))
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('precision_predict_ave_mean:',precision_predict_ave_all/(len(eval_dict)-88))
print('precision_random_mean:',precision_random_all/(len(eval_dict)-88))
print('precision_popular_mean:',precision_popular_all/(len(eval_dict)-88))
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('ndcg_predict_ave_self_mean_K:',ndcg_predict_ave_self_all_K/(len(eval_dict)-min_))
print('ndcg_random_self_mean_K:',ndcg_random_self_all_K/(len(eval_dict)-min_))
print('ndcg_popular_self_mean_K:',ndcg_popular_self_all_K/(len(eval_dict)-min_))



predict_all,min_,ndcg_predict_ave_self_all,ndcg_random_self_all,\
    ndcg_popular_self_all,precision_predict_ave_all,\
    precision_random_all,precision_popular_all,\
    ndcg_predict_ave_self_all_K,ndcg_random_self_all_K,ndcg_popular_self_all_K,\
    precision_predict_ave_all_K,precision_random_all_K,precision_popular_all_K = calculate_all(10)

print('top10====================================================================')

print('predict_all:',predict_all)
print('min_:',min_)
# print('ndcg_predict_ave_self_all:',ndcg_predict_ave_self_all)
# print('ndcg_random_self_all:',ndcg_random_self_all)
# print('ndcg_popular_self_all:',ndcg_popular_self_all)
# print('precision_predict_ave_all:',precision_predict_ave_all)
# print('precision_random_all:',precision_random_all)
# print('precision_popular_all:',precision_popular_all)
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('ndcg_predict_ave_self_mean:',ndcg_predict_ave_self_all/len(eval_dict))
print('ndcg_random_self_mean:',ndcg_random_self_all/len(eval_dict))
print('ndcg_popular_self_mean:',ndcg_popular_self_all/len(eval_dict))
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('precision_predict_ave_mean:',precision_predict_ave_all/(len(eval_dict)-88))
print('precision_random_mean:',precision_random_all/(len(eval_dict)-88))
print('precision_popular_mean:',precision_popular_all/(len(eval_dict)-88))
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('ndcg_predict_ave_self_mean_K:',ndcg_predict_ave_self_all_K/(len(eval_dict)-min_))
print('ndcg_random_self_mean_K:',ndcg_random_self_all_K/(len(eval_dict)-min_))
print('ndcg_popular_self_mean_K:',ndcg_popular_self_all_K/(len(eval_dict)-min_))


predict_all,min_,ndcg_predict_ave_self_all,ndcg_random_self_all,\
    ndcg_popular_self_all,precision_predict_ave_all,\
    precision_random_all,precision_popular_all,\
    ndcg_predict_ave_self_all_K,ndcg_random_self_all_K,ndcg_popular_self_all_K,\
    precision_predict_ave_all_K,precision_random_all_K,precision_popular_all_K = calculate_all(20)

print('top20====================================================================')

print('predict_all:',predict_all)
print('min_:',min_)
# print('ndcg_predict_ave_self_all:',ndcg_predict_ave_self_all)
# print('ndcg_random_self_all:',ndcg_random_self_all)
# print('ndcg_popular_self_all:',ndcg_popular_self_all)
# print('precision_predict_ave_all:',precision_predict_ave_all)
# print('precision_random_all:',precision_random_all)
# print('precision_popular_all:',precision_popular_all)
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('ndcg_predict_ave_self_mean:',ndcg_predict_ave_self_all/len(eval_dict))
print('ndcg_random_self_mean:',ndcg_random_self_all/len(eval_dict))
print('ndcg_popular_self_mean:',ndcg_popular_self_all/len(eval_dict))
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('precision_predict_ave_mean:',precision_predict_ave_all/(len(eval_dict)-88))
print('precision_random_mean:',precision_random_all/(len(eval_dict)-88))
print('precision_popular_mean:',precision_popular_all/(len(eval_dict)-88))
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('ndcg_predict_ave_self_mean_K:',ndcg_predict_ave_self_all_K/(len(eval_dict)-min_))
print('ndcg_random_self_mean_K:',ndcg_random_self_all_K/(len(eval_dict)-min_))
print('ndcg_popular_self_mean_K:',ndcg_popular_self_all_K/(len(eval_dict)-min_))

print()

def calculate(topk):
    k = 0
    k_ndcg = 0
    predict_all = 0

    ndcg_predict_ave_real_all = 0
    ndcg_random_real_all = 0
    ndcg_popular_real_all = 0

    ndcg_predict_ave_self_all = 0
    ndcg_random_self_all = 0
    ndcg_popular_self_all = 0

    precision_predict_ave_all = 0
    precision_random_all = 0
    precision_popular_all = 0

    for user_id in range(1, len(eval_dict)):
        if len(eval_dict[user_id]) > 0:
            k += 1
            rates_real = [data[1] for data in users_history_dict_eval[user_id]]
            ll = len(rates_real)
            rates_real.sort(reverse=True)
            rates_real = rates_real[0:topk]

            rates_predict_ave = eval_dict[user_id]['ave'][0:topk]
            rates_random = eval_dict[user_id]['random'][0:topk]
            rates_popular = eval_dict[user_id]['popular'][0:topk]
            if ll >= topk:
                k_ndcg+=1
                predict_all += topk
                idcg_real = _dcg_calculate(rates_real)

                dcg_predict_ave = _dcg_calculate(rates_predict_ave)
                dcg_predict_random = _dcg_calculate(rates_random)
                dcg_predict_popular = _dcg_calculate(rates_popular)

                rates_predict_ave.sort(reverse=True)
                rates_random.sort(reverse=True)
                rates_popular.sort(reverse=True)
                idcg_predict_ave = _dcg_calculate(rates_predict_ave)
                idcg_predict_random = _dcg_calculate(rates_random)
                idcg_predict_popular = _dcg_calculate(rates_popular)

                ndcg_predict_ave_real = dcg_predict_ave / idcg_real
                ndcg_random_real = dcg_predict_random / idcg_real
                ndcg_popular_real = dcg_predict_popular / idcg_real

                ndcg_predict_ave_self = dcg_predict_ave / idcg_predict_ave
                ndcg_random_self = dcg_predict_random / idcg_predict_random
                ndcg_popular_self = dcg_predict_popular / idcg_predict_popular

                ndcg_predict_ave_real_all += ndcg_predict_ave_real
                ndcg_random_real_all += ndcg_random_real
                ndcg_predict_ave_self_all += ndcg_predict_ave_self
                ndcg_random_self_all += ndcg_random_self
                ndcg_popular_self_all += ndcg_popular_self
                ndcg_popular_real_all += ndcg_popular_real
                print('real_mean:', k, ' ', (ndcg_predict_ave_real_all) / k_ndcg, '  ', (ndcg_random_real_all) / k_ndcg, '  ',
                      (ndcg_popular_real_all) / k_ndcg)
                print('self_mean:', k_ndcg, ' ', (ndcg_predict_ave_self_all) / k_ndcg, '  ', (ndcg_random_self_all) / k_ndcg, '  ',
                      (ndcg_popular_self_all) / k_ndcg)

            else:
                predict_all += ll
            precision_predict_ave = precision_calculate(rates_predict_ave, topk)
            precision_random = precision_calculate(rates_random, topk)
            precision_popular = precision_calculate(rates_popular, topk)


            precision_predict_ave_all += precision_predict_ave
            precision_random_all += precision_random
            precision_popular_all += precision_popular

            # print('real:',ndcg_predict_ave_real,'  ',ndcg_predict_ave_real,'  ',ndcg_popular_real)
            # print('self:',ndcg_predict_ave_self,'  ',ndcg_predict_ave_self,'  ',ndcg_popular_self)

            print('precision:', precision_predict_ave_all / (k), '  ', precision_random_all / (k), '  ',
                  precision_popular_all / (k))

            print(ll,'===================================================')
            print(topk,'topk===================================================')


calculate(10)




print()
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m/')
STATE_SIZE = 10
# 加载数据
ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'ratings.dat'), 'r').readlines()]
users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'users.dat'), 'r').readlines()]
movies_list = [i.strip().split("::") for i in
               open(os.path.join(DATA_DIR, 'movies.dat'), encoding='latin-1').readlines()]

# 用户对电影的评价dataframe，用户id，电影id，评分，时间戳
ratings_dataframe = pd.DataFrame(ratings_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype=np.uint32)
# 电影数量3883
movies_list_len = len(movies_list)
# 空值数量，无，数据已经清洗过
null_sum = ratings_dataframe.isnull().sum()
print(len(set(ratings_dataframe["UserID"])) == max([int(i) for i in set(ratings_dataframe["UserID"])]))
print(max([int(i) for i in set(ratings_dataframe["UserID"])]))
ratings_dataframe = ratings_dataframe.applymap(int)
ratings_dataframe = ratings_dataframe.apply(np.int32)

















# '''ac网络'''
# # 电影id对应的电影信息，字典键是电影id，值是电影信息列表，传进环境里可以检查电影名
# movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}
# np.save("./data/movies_id_to_movies.npy", movies_id_to_movies)
#
#
# # 按用户顺序整理记录用以ac网络
users_dict = {user: [] for user in set(ratings_dataframe["UserID"])}
# 按时间排序
ratings_dataframe = ratings_dataframe.sort_values(by='Timestamp', ascending=True)
#
users_dict1 = {user: [] for user in set(ratings_dataframe["UserID"])}

# 好评历史长度
ratings_df_gen = ratings_dataframe.iterrows()
users_dict_for_history_len = {user: [] for user in set(ratings_dataframe["UserID"])}
users_dict_for_positive_history_len = {user: [] for user in set(ratings_dataframe["UserID"])}
for data in ratings_df_gen:
    users_dict[data[1]['UserID']].append((data[1]['MovieID'], data[1]['Rating']))
    users_dict_for_history_len[data[1]['UserID']].append((data[1]['MovieID'], data[1]['Rating']))
    if data[1]['Rating'] >= 4:
        users_dict_for_positive_history_len[data[1]['UserID']].append((data[1]['MovieID'], data[1]['Rating']))
# 每个用户的观影历史长度，共6040个用户
users_history_lens = [len(users_dict_for_history_len[u]) for u in set(ratings_dataframe["UserID"])]
# 每个用户的好评观影历史长度，共6040个用户
users_positive_history_lens = [len(users_dict_for_positive_history_len[u]) for u in set(ratings_dataframe["UserID"])]


#无后缀
# 用以训练ac网络，全集
# 每个用户的观影记录（键值对，键：用户id   值：元祖（电影和评分）组成的列表）,好评差评都包括，全集
np.save("data/users_history_dict.npy", users_dict)
# # 用户的历史长度列表全集(用户1有53条记录历史45条好评，53)
# np.save("./data/users_history_lens.npy", users_history_lens)
# # 用户的好评历史长度列表(依次序)非全集(用户1有53条记录历史45条好评，45)
# np.save("./data/users_positive_history_lens.npy", users_positive_history_lens)
# # 6041，用户总数是6040
# users_num = max(ratings_dataframe["UserID"]) + 1
# # 3953，电影总数是3952
# items_num = max(ratings_dataframe["MovieID"]) + 1
#
#
# # _train
# #训练集
# # 4832
# users_num_train = int(users_num * 0.8)
# # 3953
# items_num_train = items_num
# users_dict_train = {k: users_dict[k] for k in range(1, users_num_train + 1)}
# np.save("./data/users_dict_train.npy", users_dict_train)
# users_history_lens_train = users_history_lens[:users_num_train]
# np.save("./data/users_history_lens_train.npy", users_history_lens_train)
# users_positive_history_lens_train = users_positive_history_lens[:users_num_train]
# np.save("./data/users_positive_history_lens_train.npy", users_positive_history_lens_train)
#
# # _eval
# # 测试集
# users_num_eval = int(users_num * 0.2)
# items_num_eval = items_num
# users_dict_eval = {k: users_dict[k] for k in range(users_num-users_num_eval, users_num)}
# np.save("./data/users_dict_eval.npy", users_dict_train)
# users_history_lens_eval = users_positive_history_lens[users_num - users_num_eval - 1:]
# np.save("./data/users_history_lens_eval.npy", users_history_lens_eval)
# users_positive_history_lens_eval = users_positive_history_lens[users_num - users_num_eval - 1:]
# np.save("./data/users_positive_history_lens_eval.npy", users_positive_history_lens_eval)




'''用以embedding网络'''
# 用户对电影的评价dataframe，用户id，电影id，评分，转化为好评对后预训练embedding用
user_movie_rating_dataframe = ratings_dataframe[['UserID', 'MovieID', 'Rating']]
user_movie_rating_dataframe = user_movie_rating_dataframe.apply(np.int32)
# 筛选出评分>3的好评记录(drop掉差评记录)，需要pairs和dict
index_negative = user_movie_rating_dataframe[user_movie_rating_dataframe['Rating'] < 4].index
positive_rating_df = user_movie_rating_dataframe.drop(index_negative)
positive_u_m_pairs = positive_rating_df.drop('Rating', axis=1).to_numpy()
np.save("./data/positive_u_m_pairs.npy", positive_u_m_pairs)
positive_user_movie_dict = {u: [] for u in range(1, max(positive_rating_df['UserID']) + 1)}
for data in positive_rating_df.iterrows():
    positive_user_movie_dict[data[1][0]].append(data[1][1])
np.save("./data/positive_user_movie_dict.npy", positive_user_movie_dict)

# 筛选出和ac网络统一的训练记录(drop掉>train_users_num的记录)，pairs和dict  （_train）
index_none_train = positive_rating_df[positive_rating_df['UserID'] > 4832].index
positive_rating_df_train = positive_rating_df.drop(index_none_train)
positive_u_m_pairs_train = positive_rating_df_train.drop('Rating', axis=1).to_numpy()
np.save("./data/positive_u_m_pairs_train.npy", positive_u_m_pairs_train)
positive_user_movie_dict_train = {u: [] for u in range(1, max(positive_rating_df_train['UserID']) + 1)}
for data in positive_rating_df_train.iterrows():
    positive_user_movie_dict_train[data[1][0]].append(data[1][1])
np.save("./data/positive_user_movie_dict_train.npy", positive_user_movie_dict_train)



