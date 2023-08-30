import copy

import numpy
'''数据处理，训练集，测试集，验证集的分割 
   embedding网络和ac网络 
'''
result_drrn = numpy.load('./eval_record_dict/result_drrn.npy', allow_pickle=True).item()
result_random = numpy.load('./eval_record_dict/result_random.npy', allow_pickle=True).item()
result_popularity = numpy.load('./eval_record_dict/result_popularity.npy', allow_pickle=True).item()
result_best = numpy.load('./eval_record_dict/result_best.npy', allow_pickle=True).item()
result_worst = numpy.load('./eval_record_dict/result_worst.npy', allow_pickle=True).item()
all = {'result_best':result_best,
       'result_worst':result_worst,
       'result_random':result_random,
       'result_popularity':result_popularity,
       'result_drrn':result_drrn,
       }
episode_list = []
episode = 100
while episode < 6001:
    episode_list.append(episode)
    episode += 100

for n in [10,15]:
    for episode in episode_list:
        if n==15 and episode > 1900:
            break
        result_drrn_ = numpy.load(r'eval_record_dict/result_drrn_actor_episode' + str(episode)+'_statedim_'+str(n) + '.npy', allow_pickle=True).item()
        all['drrn__episode' + str(episode)+'_dim_'+str(n)] = result_drrn_

print()



def _dcg_calculate(rate_list):
    result = 0
    i = 1
    for rate in rate_list:
        result += 2**rate / (numpy.log2(i + 1))
        i += 1
    return result
def precision_calculate(rate_list,topk):
    result = 0
    for rate in rate_list[:topk]:
        if rate>3:
            result += 1
    # return result/len(rate_list)
    return result/topk
def ndcg_calculate(rate_list,top_k):
    rate_list_i = copy.deepcopy(rate_list[:top_k])
    rate_list_i.sort(reverse=True)
    dcg = _dcg_calculate(rate_list[:top_k])
    idcg = _dcg_calculate(rate_list_i)
    return dcg/idcg
def calculate_dict_precision(dict,top_k):
    precision_topk_all = 0
    all_len = len(dict)
    for user in dict:
        rate_list = dict[user]
        precision_topk = precision_calculate(rate_list,top_k)
        precision_topk_all += precision_topk
    precision_topk_mean = precision_topk_all/all_len
    return precision_topk_mean
def calculate_dict_ndcg(dict,top_k):
    ndcg_topk_all = 0
    all_len = len(dict)
    for user in dict:
        rate_list = dict[user]
        precision_topk = precision_calculate(rate_list,top_k)
        ndcg_topk = ndcg_calculate(rate_list,top_k)
        ndcg_topk_all +=ndcg_topk
    ndcg_topk_mean = ndcg_topk_all/all_len
    return ndcg_topk_mean

eval_record_all = {}
for dict in all:
    eval_record_all[dict] = {}
    eval_record_all[dict]['precision'] = {}
    eval_record_all[dict]['ndcg'] = {}
    for i in [5, 10, 15, 20]:
        precision_topk_mean = calculate_dict_precision(all[dict],i)
        ndcg_topk_mean = calculate_dict_ndcg(all[dict],i)
        eval_record_all[dict]['precision'][i] = precision_topk_mean
        eval_record_all[dict]['ndcg'][i] = ndcg_topk_mean

numpy.save("./eval_compare_dict/compare_drrn_3000_offline.npy",eval_record_all)



# print(eval_record_all)
