import numpy
'''
model          |precision@5 |precision@10|precision@15|precision@20|ndcg@5      |ndcg@10     |ndcg@15     |ndcg@20     |
result__random |0.5867550   |0.5338742   |0.4791170   |0.4338907   |0.8687449   |0.8403910   |0.8332720   |0.8311111   |
result_popular |0.6930132   |0.6074503   |0.5336645   |0.4760348   |0.9008626   |0.8825751   |0.8781703   |0.8768269   |
result____drrn |0.7166538   |0.6927637   |0.6569669   |0.6116628   |0.8998819   |0.8772692   |0.8710887   |0.8692468   |

model                         |precision@5 |precision@10|precision@15|precision@20|ndcg@5      |ndcg@10     |ndcg@15     |ndcg@20     |
result_best                   |0.9138742   |0.7752318   |0.6673731   |0.5837169   |1.0000000   |1.0000000   |1.0000000   |1.0000000   |
result_worst                  |0.2272848   |0.2899338   |0.2976490   |0.2925662   |0.8453137   |0.7842482   |0.7591288   |0.7450057   |
result_random                 |0.5867550   |0.5338742   |0.4791170   |0.4338907   |0.8687449   |0.8403910   |0.8332720   |0.8311111   |
result_popularity             |0.6930132   |0.6074503   |0.5336645   |0.4760348   |0.9008626   |0.8825751   |0.8781703   |0.8768269   |
result_drrn                   |0.7166538   |0.6927637   |0.6569669   |0.6116628   |0.8998819   |0.8772692   |0.8710887   |0.8692468   |



'''


result_drrn3000_compare_random_popular = numpy.load('./eval_compare_dict/compare_drrn_3000_offline.npy', allow_pickle=True).item()
table = {}
print('%-30s'%'model',end='|')
for dict in result_drrn3000_compare_random_popular:
    for key in result_drrn3000_compare_random_popular[dict]:
        for topk in result_drrn3000_compare_random_popular[dict][key]:
            s = key+'@'+str(topk)
            print('%-12s'%s,end='|')
    break
print()
for dict in result_drrn3000_compare_random_popular:
    print('%-30s'%dict,end='|')
    for value in result_drrn3000_compare_random_popular[dict].values():
        for v in value.values():
            print('%.7f'%v,end='   |')
    print()
print()
