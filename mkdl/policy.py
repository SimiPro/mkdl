import tensorflow as tf
import numpy as np
from baselines.a2c import utils
from baselines.common.distributions import make_pdtype

class ContCnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.shape[0]
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = utils.conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = utils.conv(h, 'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = utils.conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = utils.conv_to_fc(h3)
            h4 = utils.fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = utils.fc(h4, 'pi', nact, act=lambda x:x, init_scale=0.01)
            vf = utils.fc(h4, 'v', 1, act=lambda x:x)[:,0]
            logstd = tf.get_variable(name='logstd', shape=[1, nact], initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi*0.0+logstd], axis=1)
        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value