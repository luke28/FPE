import numpy as np
import tensorflow as tf

class TransferEmbedding(object):
    def __init__(self, params):
        self.learn_rate = params["learn_rate"]
        self.optimizer = params["optimizer"] if "optimizer" in params else "GradientDescentOptimizer"
        self.embedding_size = params["embedding_size"]
        self.num_nodes = params["num_nodes"]

        self.tensor_graph = tf.Graph()
        with self.tensor_graph.as_default():
            self.D = tf.placeholder(tf.float32, shape = [self.num_nodes, self.num_nodes])

            self.Z = tf.Variable(tf.random_uniform([self.num_nodes, self.embedding_size], -1.0, 1.0), name = "Z", dtype = tf.float32)
            self.lbd = tf.Variable(tf.random_uniform([self.num_nodes], -1.0, 1.0), name = "lambda", dtype = tf.float32)

            self.first = tf.matmul((self.D + tf.diag(self.lbd)), self.Z)
            self.a = tf.square(tf.norm(self.Z, axis = 1))
            self.loss = tf.norm(self.first) + tf.norm(self.a - 1)
            # shape(a) = [n, 1]

            self.train_step = getattr(tf.train, self.optimizer)(self.learn_rate).minimize(self.loss)


    def train(self, D, epoch_num = 10001, save_path = None):
        with tf.Session(graph = self.tensor_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in xrange(epoch_num):
                self.train_step.run({self.D : D})
                if (i % 1000 == 0):
                    print(self.loss.eval({self.D : D}))
            if save_path is not None:
                saver = tf.train.Saver()
                saver.save(sess, save_path)
            z = sess.run(self.Z)
            return z

    def load_model(self, save_path):
        with tf.Session(graph = self.tensor_graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path)
            z = sess.run(self.Z)
            return z

    def transfer(self, X, xc, r, epoch_num = 10001, save_path= None):
        print "transfer_embeddings:"
        X = np.array(X)
        xc = np.array(xc)
        a = np.square(np.linalg.norm(X, axis = 1, keepdims = True))
        D = -2 * np.dot(X, np.transpose(X)) + a + np.transpose(a)
        D = D / np.linalg.norm(D)
        S = np.sum(D, axis = 1)
        S = np.diag(S)
        print D
        print S
        Z = self.train(S - D, epoch_num, save_path)
        print Z
        print -2 * np.dot(Z, np.transpose(Z)) + np.square(np.linalg.norm(Z, axis = 1, keepdims = True)) + np.transpose(np.square(np.linalg.norm(Z, axis = 1, keepdims = True)))
        Z = Z * r + xc
        return Z


def main():
    params = {'learn_rate': 0.001, 'embedding_size': 2, 'num_nodes': 3, 'clip_min' : 1e-5}
    cli = TransferEmbedding(params)
    X = np.array([[0,0], [3, 0], [0, 4]], dtype = np.float32)
    xc = [1, 1]
    r = [3]
    Z = cli.transfer(X, xc, r)
    print Z

if __name__ == "__main__":
    main()
