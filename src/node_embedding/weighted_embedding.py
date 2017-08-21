import numpy as np
import tensorflow as tf

class NodeEmbedding(object):
    def __init__(self, param):
        self.embedding_size = param["embedding_size"]
        self.batch_size = param["batch_size"]
        self.num_nodes = param["num_nodes"]
        self.learn_rate = param["learn_rate"]
        self.optimizer = params["optimizer"] if "optimizer" in params else "GradientDescentOptimizer"
        self.loss_func = params["loss_func"] if "loss_func" in params else "cross_entropy"

        self.tensor_graph = tf.Graph()

        with self.tensor_graph.as_default():
            self.x_nodes = tf.placeholder(tf.int32, shape = [None, 1])
            self.y_ = tf.placeholder(tf.float64, shape = [None, self.num_nodes])

            self.embeddings = tf.Variable(tf.random_uniform([self.num_nodes, self.embedding_size], -1.0, 1.0, dtype = tf.float64), name = "embeddings", dtype = tf.float64)
            self.w = tf.Variable(tf.random_uniform([self.embedding_size, self.num_nodes],-1.0, 1.0, dtype = tf.float64), name = "w", dtype = tf.float64)
            self.b = tf.Variable(tf.zeros([self.num_nodes], dtype = tf.float64), name = "b", dtype = tf.float64)

            self.embed_pre = tf.nn.embedding_lookup(self.embeddings, self.x_nodes)
            self.embed = tf.reshape(self.embed_pre, [-1, self.embedding_size])

            self.y_sig = tf.nn.sigmoid(tf.matmul(self.embed, self.w) + self.b)
            self.y_sof = tf.nn.softmax(tf.matmul(self.embed, self.w) + self.b)

            self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_sof), reduction_indices=[1]))
            self.mean_square = tf.reduce_mean(tf.reduce_sum(tf.square(self.y_ - self.y_sig), reduction_indices = [1]))

            self.train_step = getattr(tf.train, self.optimizer)(self.learn_rate).minimize(getattr(self, self.loss_func))

            #self.train_step = tf.train.AdamOptimizer(self.learnRate).minimize(self.cross_entropy)

    def train(self, get_batch, epoch_num = 10001, save_path = None):
        with tf.Session(graph = self.tensor_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in xrange(epoch_num):
                batch_nodes, batch_y = get_batch(self.batch_size)
                self.train_step.run({self.x_nodes : batch_nodes, self.y_ : batch_y})
                if (i % 100 == 0):
                    print(getattr(self, self.loss_func).eval({self.x_nodes : batch_nodes, self.y_ : batch_y}))
            if save_path is not None:
                saver = tf.train.Saver()
                saver.save(sess, save_path)
            return sess.run(self.embeddings)

    def load_model(self, save_path):
        with tf.Session(graph = self.tensor_graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path)
            return sess.run(self.embeddings)

