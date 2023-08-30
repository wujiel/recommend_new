import tensorflow
class MultiNetwork(tensorflow.keras.Model):
    def __init__(self, len_users, len_movies, embedding_dim):
        super(MultiNetwork, self).__init__()
        self.m_u_input = tensorflow.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        # embedding
        self.u_embedding = tensorflow.keras.layers.Embedding(name='user_embedding', input_dim=len_users,
                                                             output_dim=embedding_dim)
        self.m_embedding = tensorflow.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies,
                                                             output_dim=embedding_dim)
        # output
        # 以后可以多记忆一些特征，多分类
        # self.m_u = tensorflow.keras.layers.Dense(100, activation='relu')

        self.m_u_fc = tensorflow.keras.layers.Dense(5, activation='softmax')

    def call(self, x):
        x = self.m_u_input(x)
        uemb = self.u_embedding(x[0])
        memb = self.m_embedding(x[1])
        m_u = uemb*memb
        return self.m_u_fc(m_u)


