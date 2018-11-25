import tensorflow as tf 
import data_utils
import os 


class behaviorCloning:
    def __init__(self,file_name,envname,hidden_size,batch_size,learning_rate,skip_step,training=False):
        self.graph = tf.Graph()
        self.training = training
        self.file_name = file_name
        self.envname = envname
        self.need_restore = True 
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.skip_step = skip_step
        self.sess = tf.InteractiveSession(graph=self.graph)
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)

    def _import_data(self):
        self.X, self.Y = data_utils.read_data(self.file_name)
        with tf.name_scope('data'):
            self.X_placeholder = tf.placeholder(tf.float32,[None,self.X.shape[1]],name='observations')
            self.Y_placeholder = tf.placeholder(tf.float32,[None,self.Y.shape[1]],name='actions')

    def _create_layer(self,x,output_dim,scope):
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable('w',dtype=tf.float32,shape=[x.shape[1],output_dim],
                                      initializer=tf.random_uniform_initializer())
            b = tf.get_variable('b',dtype=tf.float32,shape=[output_dim],
                                      initializer=tf.constant_initializer(0.0))
            return tf.matmul(x,w)+b 

    def _create_model(self):
        # self.h1 = self._create_layer(self.X_placeholder,self.hidden_size*2,'layer1')
        # h11 = tf.nn.relu(self.h1)
        # h2 = self._create_layer(h11,self.hidden_size//2,'layer3')
        # h22 = tf.sigmoid(h2)
        # h3 = self._create_layer(h22,self.Y.shape[1],'outputLayer')
        with tf.name_scope('layer1'):
            hidden1 = tf.contrib.layers.fully_connected(self.X_placeholder, 
            num_outputs=128, activation_fn=tf.nn.relu)
        with tf.name_scope('layer2'):
            hidden2 = tf.contrib.layers.fully_connected(hidden1, 
            num_outputs=256, activation_fn=tf.nn.relu)
        with tf.name_scope('layer3'):
            hidden3 = tf.contrib.layers.fully_connected(hidden2, 
            num_outputs=64, activation_fn=tf.sigmoid)
        with tf.name_scope('output'):
            pred = tf.contrib.layers.fully_connected(hidden3, 
            num_outputs=self.Y.shape[1], activation_fn=None)
        self.output = pred

    def _create_loss(self):
        self.loss = tf.reduce_mean(tf.square(self.output-self.Y_placeholder))
        # self.loss = tf.reduce_sum(tf.square(self.output-self.Y_placeholder))

    def _create_optimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate
            ).minimize(self.loss, global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss',self.loss)
            # tf.summary.histogram('histogram loss',self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        with self.graph.as_default():
            self._import_data()
            self._create_model()
            self._create_loss()
            self._create_optimizer()
            self._create_summaries()

    def train(self,num_train_steps):
        with self.graph.as_default():
            saver = tf.train.Saver()

            initial_step = 0
            try:
                os.mkdir('checkpoints/'+self.envname)
                os.mkdir('graphs/'+self.envname)
            except:
                pass

            self.sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('graphs/'+self.envname+'/lr'+str(self.learning_rate),self.sess.graph)
            initial_step = self.global_step.eval(self.sess)

            total_loss = 0

            for i in range(num_train_steps):
                beg = (i%(self.X.shape[0]//self.batch_size))*self.batch_size
                end = min(beg+self.batch_size,self.X.shape[0])
                loss_batch, _, summary = self.sess.run([self.loss,self.optimizer,self.summary_op],
                                                feed_dict={self.X_placeholder:self.X[beg:end], self.Y_placeholder:self.Y[beg:end]})
                total_loss += loss_batch

                if (i+1)%self.skip_step == 0:
                    print('Average loss at step {}:{:5.1f}'.format(i,total_loss/self.skip_step))
                    total_loss = 0.0 

                saver.save(self.sess,'checkpoints/'+self.envname+'/bc',num_train_steps)
                writer.close()
                self.need_restore = False

    def apply(self,x):
        with self.graph.as_default():
            if self.need_restore:
                saver = tf.train.Saver()
                self.need_restore = False
                ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self.sess,ckpt.model_checkpoint_path)
                else:
                    print("not exist")

            res = self.sess.run(self.output,feed_dict={self.X_placeholder:x})
            return res 





