import tensorflow
import tensorflow as tf
import numpy



class EmbeddingNetwork(tensorflow.keras.Model):
    def __init__(self, len_users, len_movies, embedding_dim):
        super(EmbeddingNetwork, self).__init__()
        self.m_u_input = tensorflow.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        # embedding
        self.u_embedding = tensorflow.keras.layers.Embedding(name='user_embedding', input_dim=len_users,
                                                     output_dim=embedding_dim)
        self.m_embedding = tensorflow.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies,
                                                     output_dim=embedding_dim)
        # dot product
        self.m_u_merge = tensorflow.keras.layers.Dot(name='movie_user_dot', normalize=False, axes=1)
        # output
        # 以后可以多记忆一些特征，多分类
        self.m_u_fc = tensorflow.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.m_u_input(x)
        uemb = self.u_embedding(x[0])
        memb = self.m_embedding(x[1])
        m_u = self.m_u_merge([memb, uemb])
        return self.m_u_fc(m_u)


class UserMovieEmbedding(tf.keras.Model):
    def __init__(self, len_users, len_movies, embedding_dim):
        super(UserMovieEmbedding, self).__init__()
        self.m_u_input = tf.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        # embedding
        self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users,
                                                     output_dim=embedding_dim)
        self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies,
                                                     output_dim=embedding_dim)
        # dot product
        self.m_u_merge = tf.keras.layers.Dot(name='movie_user_dot', normalize=False, axes=1)
        # output
        self.m_u_fc = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.m_u_input(x)
        uemb = self.u_embedding(x[0])
        memb = self.m_embedding(x[1])
        m_u = self.m_u_merge([memb, uemb])
        return self.m_u_fc(m_u)


class SingleNetwork(tensorflow.keras.Model):
    def __init__(self, len_users, len_movies, embedding_dim):
        super(SingleNetwork, self).__init__()
        self.m_u_input = tensorflow.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        # embedding
        self.u_embedding = tensorflow.keras.layers.Embedding(name='user_embedding', input_dim=len_users,
                                                             output_dim=embedding_dim)
        self.m_embedding = tensorflow.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies,
                                                             output_dim=embedding_dim)
        self.dense1 = tensorflow.keras.layers.Dense(200, activation='relu')
        self.dense2 = tensorflow.keras.layers.Dense(50, activation='relu')

        self.m_u_fc = tensorflow.keras.layers.Dense(2, activation='softmax')

    def call(self, x):
        x = self.m_u_input(x)
        uemb = self.u_embedding(x[0])
        memb = self.m_embedding(x[1])
        m_u = uemb*memb
        m_u = self.dense1(m_u)
        m_u = self.dense2(m_u)
        return self.m_u_fc(m_u)



class SoftmaxNetwork(tensorflow.keras.Model):
    def __init__(self, len_users, len_movies, embedding_dim):
        super(SoftmaxNetwork, self).__init__()
        self.m_u_input = tensorflow.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        # embedding
        self.u_embedding = tensorflow.keras.layers.Embedding(name='user_embedding', input_dim=len_users,
                                                     output_dim=embedding_dim)
        self.m_embedding = tensorflow.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies,
                                                     output_dim=embedding_dim)
        # dot product
        self.m_u_merge = tensorflow.keras.layers.Dot(name='movie_user_dot', normalize=False, axes=1)
        # output
        # 以后可以多记忆一些特征，多分类

        self.m_u_fc = tensorflow.keras.layers.Dense(5, activation='softmax')


    def call(self, x):
        x = self.m_u_input(x)
        uemb = self.u_embedding(x[0])
        memb = self.m_embedding(x[1])
        m_u = self.m_u_merge([memb, uemb])
        return self.m_u_fc(m_u)


class CnnNetwork(tensorflow.keras.Model):
    def __init__(self, len_users, len_movies, embedding_dim):
        super(CnnNetwork, self).__init__()
        self.m_u_input = tensorflow.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        # embedding
        self.u_embedding = tensorflow.keras.layers.Embedding(name='user_embedding', input_dim=len_users,
                                                             output_dim=embedding_dim)
        self.m_embedding = tensorflow.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies,
                                                             output_dim=embedding_dim)
        # output
        # 以后可以多记忆一些特征，多分类
        # self.m_u = tensorflow.keras.layers.Dense(100, activation='relu')
        self.cnn1 = tensorflow.keras.layers.Conv2D(filters=8, kernel_size=(10, 10), strides=2, activation='relu')
        self.cnn2 = tensorflow.keras.layers.Conv2D(filters=16, kernel_size=(8, 8),strides=2,  activation='relu')
        self.cnn3 = tensorflow.keras.layers.Conv2D(filters=32, kernel_size=(5, 5),  strides=2,activation='relu')

        self.pool = tensorflow.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.flaten = tensorflow.keras.layers.Flatten()

        self.m_u_fc = tensorflow.keras.layers.Dense(5, activation='softmax')

    def call(self, x):
        x = self.m_u_input(x)
        uemb = self.u_embedding(x[0])
        memb = self.m_embedding(x[1])
        u = [uemb[0]]
        m = [memb[0]]
        m_u = tensorflow.matmul(u,m,transpose_a=True)
        m_u = tensorflow.expand_dims(m_u,0)

        length = len(uemb)-1
        for i in range(length):
            m_u_lone = tensorflow.matmul([uemb[i+1]],[memb[i+1]],transpose_a=True)
            m_u_lone = tensorflow.expand_dims(m_u_lone,0)
            m_u = tensorflow.concat([m_u_lone,m_u],axis=0)

        m_u = tensorflow.expand_dims(m_u,3)

        m_u = self.cnn1(m_u)
        m_u = self.cnn2(m_u)
        m_u = self.cnn3(m_u)
        m_u = self.pool(m_u)
        m_u = self.flaten(m_u)

        return self.m_u_fc(m_u)



class ConcatNetwork(tensorflow.keras.Model):
    def __init__(self, len_users, len_movies, embedding_dim):
        super(ConcatNetwork, self).__init__()
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
        m_u = tensorflow.concat([uemb,memb],axis=1)
        return self.m_u_fc(m_u)



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



class MultiLayerNetwork(tensorflow.keras.Model):
    def __init__(self, len_users, len_movies, embedding_dim):
        super(MultiLayerNetwork, self).__init__()
        self.m_u_input = tensorflow.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        # embedding
        self.u_embedding = tensorflow.keras.layers.Embedding(name='user_embedding', input_dim=len_users,
                                                             output_dim=embedding_dim)
        self.m_embedding = tensorflow.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies,
                                                             output_dim=embedding_dim)
        # output
        # 以后可以多记忆一些特征，多分类
        self.dense1 = tensorflow.keras.layers.Dense(200, activation='relu')
        self.dense2 = tensorflow.keras.layers.Dense(50, activation='relu')

        self.m_u_fc = tensorflow.keras.layers.Dense(5, activation='softmax')

    def call(self, x):
        x = self.m_u_input(x)
        uemb = self.u_embedding(x[0])
        memb = self.m_embedding(x[1])
        m_u = uemb*memb
        m_u = self.dense1(m_u)
        m_u = self.dense2(m_u)
        return self.m_u_fc(m_u)
