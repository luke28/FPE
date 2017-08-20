import numpy as np
import tensorflow as tf

class TransferEmbedding(object):
    def __init__(self, params):
        self.learn_rate = params["learn_rate"]
        self.optimizer = params["optimizer"] if "optimizer" in params else "GradientDescentOptimizer"
        self.embedding_size = params["embedding_size"]
        self.num_nodes = params["num_nodes"]
        self.theta = params["theta"]

        self.tensor_graph = tf.Graph()
        with self.tensor_graph.as_default():
            self.D = tf.placeholder(tf.float64, shape = [self.num_nodes, self.num_nodes])
            self.xc = tf.placeholder(tf.float64, shape = [self.embedding_size])
            self.r = tf.placeholder(tf.float64, shape = [1])

            self.Z = tf.Variable(tf.random_uniform([self.num_nodes, self.embedding_size], -1.0, 1.0, dtype = tf.float64), name = "Z", dtype = tf.float64)

            # shape(a) = [n, 1]
            self.a = tf.square(tf.norm(self.Z, axis = 1, keep_dims = True))
            self.dist = -2 * tf.matmul(self.Z, tf.transpose(self.Z)) + self.a + tf.transpose(self.a)
            self.D_norm = tf.realdiv(self.D, tf.norm(self.D))
            self.f_norm = tf.norm(self.D_norm - tf.realdiv(self.dist, tf.norm(self.dist)))

            #self.constraint = -2 * tf.matmul(self.Z, tf.expand_dims(self.xc, -1)) + self.a + tf.square(tf.norm(self.xc)) - self.r
            self.constraint = tf.norm(self.Z - self.xc, axis = 1, keep_dims = True) - self.r
            #self.lag = tf.reduce_sum(tf.matmul(tf.expand_dims(self.alpha, 0), self.constraint))
            self.lag = tf.norm(self.constraint)
            self.loss = self.f_norm + self.theta * self.lag

            self.train_step = getattr(tf.train, self.optimizer)(self.learn_rate).minimize(self.loss)


    def train(self, D, xc, r, epoch_num = 10001, save_path = None):
        with tf.Session(graph = self.tensor_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in xrange(epoch_num):
                self.train_step.run({self.D : D, self.xc : xc, self.r : r})
                if (i % 100 == 0):
                    print(self.loss.eval({self.D : D, self.xc : xc, self.r : r}))
            if save_path is not None:
                saver = tf.train.Saver()
                saver.save(sess, save_path)
            return sess.run(self.Z), sess.run(self.dist)

    def load_model(self, save_path):
        with tf.Session(graph = self.tensor_graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path)
            return sess.run(self.Z), sess.run(self.dist)

def main():
    params = {'learn_rate': 0.001, 'embedding_size': 2, 'num_nodes': 3, 'theta': 1.0}
    cli = TransferEmbedding(params)
    r = [1]
    X = np.array([[0,0], [3, 0], [0, 4]], dtype = np.float64)
    a = np.square(np.linalg.norm(X, axis = 1, keepdims = True))
    D = -2 * np.dot(X, np.transpose(X)) + a + np.transpose(a)
    xc = [0.0, 0.0]
    Z, dic = cli.train(D, xc, r)
    print Z
    print dic

if __name__ == "__main__":
    main()
