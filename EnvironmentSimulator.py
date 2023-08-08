import numpy
import numpy as np

'''
是否应该遵循？：
只学有的，不学没的
即：训练时不推荐不在历史记录的项目，不规定其奖励（因为历史记录没有）
不学习没有的短期模式序列（即最近n条交互真的在历史上发生而不是推荐系统推荐后模拟环境就把模拟交互当做短期模式就）

有历史记录的情况下，是否应该直接MC学习而不是TD学习

若TD学习，必引入假设


'''
def count_positive_rates(rates_list):
    count = 0
    for rate in rates_list:
        if rate > 3:
            count += 1
    return count


def sum_positive_reward(rates_list):
    sum = 0
    for rate in rates_list:
        if rate == 4:
            sum += 1
        if rate == 5:
            sum += 2
    return sum


class EnvironmentSimulator(object):

    def __init__(self, users_history_dict,movies_information, specify_user_id=None,top_k=1):

        # 用户历史记录，已经以时间排序
        self.users_history_dict = users_history_dict
        # 可模拟用户列表
        self.available_users = self._generate_available_users(top_k)
        # 指定用户或者从可模拟用户列表中随机选择用户id
        self.user = specify_user_id if specify_user_id else np.random.choice(self.available_users)
        # 当前模拟用户的历史交互项目,包含评价，dict是无序的，但users_history_dict[self.user]是一个元祖列表
        self.user_items = {data[0]: data[1] for data in self.users_history_dict[self.user]}

        # 推荐空间
        self.recommend_space_train_real = numpy.array([data[0] for data in self.users_history_dict[self.user]])
        self.recommend_space_train = numpy.array([data for data in range(1,len(movies_information)+1)])

        # 推荐空间的评价
        self.recommend_space_train_rates = [data[1] for data in self.users_history_dict[self.user]]
        # 推荐空间积极评价的数量
        self.positive_rates_count = count_positive_rates(self.recommend_space_train_rates)

        # 推荐空间积极评价的总回报
        self.positive_rewards_sum = sum_positive_reward(self.recommend_space_train_rates)

        # 是否可用标志位，训练用。即：推荐系统在训练时推荐空间为用户历史记录包含的项目，因为超出历史记录是没有反馈的，人为设置奖励也是不合理的
        self.done = False

        self.movies_information = movies_information

        self.all_history_dict = self._generate_all_dict()

    def _generate_all_dict(self):
        dict = {}
        for user in self.users_history_dict.keys():
            dict[user] = {}
            dict[user]["movie_id"] = [data[0] for data in self.users_history_dict[self.user]]
            dict[user]["rates"] = [data[1] for data in self.users_history_dict[self.user]]
        return dict


    def _generate_available_users(self,top_k=1):
        available_users = []
        for user in self.users_history_dict.keys():
            if len(self.users_history_dict[user])>=top_k:
                available_users.append(user)
        return available_users



    # 模拟用户重设
    def reset(self,id=None):
        self.user = id if id else np.random.choice(self.available_users)
        self.user_items = {data[0]: data[1] for data in self.users_history_dict[self.user]}
        self.recommend_space_train_real = numpy.array([data[0] for data in self.users_history_dict[self.user]])
        self.recommend_space_train = numpy.array([data for data in range(1,len(self.movies_information)+1)])
        # 推荐空间的评价
        self.recommend_space_train_rates = [data[1] for data in self.users_history_dict[self.user]]


        # 以下两个属性是要让推荐系统知道的
        # 推荐空间积极评价的数量
        self.positive_rates_count = count_positive_rates(self.recommend_space_train_rates)
        # 推荐空间积极评价的总回报
        self.positive_rewards_sum = sum_positive_reward(self.recommend_space_train_rates)

        self.done = False
        return self.user, self.done

    ''' 
    返回真正的评价，不做任何归一化
    已推荐项目到达好评历史长度则更换（）
    '''

    def step(self, action, top_k=False):
        # 得到的action应该是一个具体的项目id列表
        is_in = 0
        if top_k:
            rates = []
            for act in action:
                if act in self.user_items.keys():
                    rates.append(self.user_items[act])
                    is_in+=1
                else:
                    rates.append(2)
            rate = rates

        else:
            if action in self.user_items.keys():
                rate = self.user_items[action]
                is_in = 1
            else:
                rate = 2
        return rate,is_in

    def get_items_names(self, items_ids):
        items_names = []
        for id in items_ids:
            try:
                items_names.append(self.movies_information[str(id)])
            except:
                items_names.append(list(['Not in list']))
        return items_names
