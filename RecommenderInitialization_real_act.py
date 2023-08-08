import numpy
from ExperienceBuffer import DdpgBuffer, DdpgAveBuffer_offline
from actor import DdpgActor, DdpgAveActor
from critic import DdpgCritic, DdpgAveCritic
from Dataloader import Dataloader
from EnvironmentSimulator import EnvironmentSimulator
from RecommendSystem_real_act import RecommendSystem
from EmbeddingNetwork import MultiNetwork

'''生成推荐系统所需要的:
1.环境
2.状态表示模块
3.actor
4.critic
5.经历重放的容器
'''


def RecommenderInitialization():
    dataloader = Dataloader()
    # 电影信息
    movies_id_to_movies = dataloader.movies_id_to_movies
    item_len = len(movies_id_to_movies)
    # 训练集
    users_history_dict_train = numpy.load('data/data_ac_train/users_history_dict_train.npy', allow_pickle=True).item()
    user_len = len(users_history_dict_train)
    # 测试集
    users_history_dict_eval = dataloader.users_history_dict_eval

    # 训练环境
    environment_train = EnvironmentSimulator(users_history_dict=users_history_dict_train,movies_information=movies_id_to_movies)
    environment_eval = EnvironmentSimulator(users_history_dict=users_history_dict_eval,movies_information=movies_id_to_movies)
    # 验证环境
    # 经历重放容器
    # actor
    ddpg_actor = DdpgActor(state_dim=300, hidden_dim=128, action_dim=100, learning_rate=0.001,
                         target_network_update_rate=0.001)
    # critic
    ddpg_critic = DdpgCritic(state_dim=300, action_dim=100, hidden_dim=128, learning_rate=0.001,
                           target_network_update_rate=0.001)
    # 经历重放容器
    ddpg_buffer = DdpgBuffer(buffer_size=1000000, embedding_dim=100)

    return environment_train, environment_eval,ddpg_critic, ddpg_actor, ddpg_buffer,item_len,user_len



environment_train, environment_eval, ddpg_critic, ddpg_actor, ddpg_buffer,item_len,user_len = RecommenderInitialization()

ddpg_ave_actor = DdpgAveActor(state_dim=300, hidden_dim=128, action_dim=100, learning_rate=0.001,
                                   target_network_update_rate=0.001)

ddpg_ave_critic = DdpgAveCritic(state_dim=300, action_dim=100, hidden_dim=128, learning_rate=0.001,
                                     target_network_update_rate=0.001)

# ddpg_ave_buffer(经历重放的容器)
ddpg_ave_buffer = DdpgAveBuffer_offline(buffer_size=1000000, embedding_dim=100)

embedding_network = MultiNetwork(len_users=user_len, len_movies=3900, embedding_dim=100)
# 这是在build网络，传一个数据即可
out = embedding_network([numpy.zeros((1)), numpy.zeros((1))])
embedding_network.summary()
embedding_network.load_weights(r'embedding_weights/muti/multi_network_weights99000.h5')
# embedding_network.load_weights(r'embedding_weights/user_movie_embedding_98accu.h5')
print("已加载权重")
embedding_network.summary()

recommend_system_eval = RecommendSystem(env=environment_eval,
                                        ddpg_actor=ddpg_actor,ddpg_critic=ddpg_critic, ddpg_buffer=ddpg_buffer,
                                        ddpg_ave_actor=ddpg_ave_actor,ddpg_ave_critic=ddpg_ave_critic, ddpg_ave_buffer=ddpg_ave_buffer,embedding_network=embedding_network)




"""
top10
real_mean: 4445   0.8238520565426213    0.7876446466227076
self_mean: 4445   0.9257919059925604    0.9240001040792071
precision: 0.2    0.3
precision_mean: 0.648121484814404    0.5750056242969644


real_mean: 4445   0.7827325881738771    0.7897081404972435
self_mean: 4445   0.9239326378025694    0.9253155789749995
precision: 0.3    0.2
precision_mean: 0.5615973003374578    0.5774578177727825

real: 0.8883156189494684    0.8883156189494684    0.8115468423393484
self: 0.9915906821987674    0.9915906821987674    0.9583179400334796
real_mean: 4445   0.8262242754509919    0.788489605015874    0.8472224714655895
self_mean: 4445   0.9252978898733378    0.9250208133120335    0.9439650263471369
precision: 0.6543982002249775    0.5738807649043894    0.6605849268841419
===================================================


real: 0.7708202397732913    0.7708202397732913    0.8115468423393484
self: 0.8604353577766987    0.8604353577766987    0.9583179400334796
real_mean: 4445   0.8277098317995866    0.7885556029937634    0.8534748586814158
self_mean: 4445   0.9253267871978    0.9242403021642204    0.9456879619879791
precision: 0.6560404949381392    0.5784476940382464    0.6713835770528715

real_mean: 4445   0.8267550665784644    0.7868023864997344    0.8534748586814158
self_mean: 4445   0.9249358212314254    0.9232419579226303    0.9456879619879791
precision: 0.6564904386951673    0.5732958380202505    0.6713835770528715

real_mean: 4445   0.8273067753379097    0.7880235958892045    0.8534748586814158
self_mean: 4445   0.92572134444566    0.9244788115151727    0.9456879619879791
precision: 0.6548031496063044    0.5759280089988764    0.6713835770528715












====================================================================================================
top5
self: 0.9415860650813429    0.9415860650813429    0.8932205291807925
self_mean: 5914   0.9388230290177424    0.9400802880863818    0.9465467809935744
precision: 0.6769022658099544    0.5927629354075122    0.6663184211797707


self: 0.8992629920319946    0.8992629920319946    0.8932205291807925
self_mean: 5914   0.9401649271039221    0.938611963217495    0.944809659493162
precision: 0.6794386202232117    0.5958403787622613    0.6582020870573484


self: 0.9625588944427584    0.9625588944427584    0.890584671421597
self_mean: 5914   0.9397411977654342    0.9392093685288091    0.9524672935376863
precision: 0.6763949949273018    0.5949611092323334    0.6863375042272675



real_mean: 5914   0.8261747665152024    0.7849968734417272    0.8500101090467921
self_mean: 5914   0.9393497828743884    0.9396047659314222    0.9539670088075479
precision: 0.6761244504565527    0.592593845113293    0.6957727426445841

real_mean: 5914   0.6108217051701017    0.5823322827519289    0.6238013184405458
self_mean: 5914   0.9400569603499012    0.9397689918749925    0.9528228727363609
precision: 0.6761244504565548    0.5950625634088647    0.6864389584037975

real_mean: 5914   0.6099595968072881    0.5825076821844842    0.6278783293258229
self_mean: 5914   0.9392950879570062    0.9402934838922397    0.9543843727052275
precision: 0.6757862698681198    0.5949611092323329    0.6961109232330182
=============================================================================================



close 
top10
real_mean: 5999   0.7736832154449211    0.8257590945022959    0.878830004172931
self_mean: 5999   0.9035760670304758    0.928917636136861    0.9467752365994442
precision: 0.4685447574595758    0.5338056342723801    0.6077679613268915


self_mean: 4445   0.8045013806508235    0.8421089914344907    0.8793159765541679
precision: 0.4685395131710562    0.5338112704234759    0.6077859286428847

self_mean: 4445   0.804937083422611    0.8425545766046534    0.8791372120335827
precision: 0.46827275758586184    0.5350950316772267    0.6073691230410173


top5
real_mean: 5999   0.7218183190563693    0.7865466378140102    0.8520543621157329
self_mean: 5999   0.9184339138481944    0.939576365826132    0.9543385350690231
precision: 0.49448241373562346    0.5870978496416093    0.6930821803634071


self_mean: 5914   0.8491190708009813    0.8786372103378192    0.9020127385279918
precision: 0.49448241373562346    0.5870978496416093    0.6930821803634071

self_mean: 5914   0.8505519351630639    0.8790478984545278    0.9010487486926169
precision: 0.49171528588098373    0.5916986164360756    0.6927487914652579




self_mean: 5914   0.8708647261170782    0.8774697826041017    0.9010487486926169
precision: 0.681974974636465    0.6030436252959107    0.7027054447074876





"""
recommend_system_eval.real_act_train_n(numpy.load('data/data_ac_train/users_history_dict_train.npy', allow_pickle=True).item(),10)
print("finish")

'''


'''