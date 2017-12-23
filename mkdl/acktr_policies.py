import numpy as np
import tensorflow as tf
from CapsLayer import capslayer
from baselines.acktr.utils import conv, fc, conv_to_fc, sample


def pool(X, pool_size=2, strides=1, padding='VALID'):
    z = tf.layers.max_pooling2d(X, pool_size, strides, padding=padding)
    return z


class CapsulePolicy(object):
        def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
            nbatch = nenv * nsteps
            nh, nw, nc = ob_space.shape
            ob_shape = (nbatch, nh, nw, nc * nstack)
            nact = ac_space.n
            X = tf.placeholder(tf.uint8, ob_shape)  # obs
            with tf.variable_scope("model", reuse=reuse):
                h = capslayer.layers.conv2d(tf.cast(X,tf.float32)/255., 'relu', )
                h = conv(tf.cast(X, tf.float32) / 255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
                p1 = pool(h)
                h2 = conv(p1, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
                p2 = pool(h2)
                h3 = conv(p2, 'c3', nf=32, rf=3, stride=1, init_scale=np.sqrt(2))
                p3 = pool(h3)
                h3 = conv_to_fc(p3)
                h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
                pi = fc(h4, 'pi', nact, act=lambda x: x)
                vf = fc(h4, 'v', 1, act=lambda x: x)

            v0 = vf[:, 0]
            a0 = sample(pi)
            self.initial_state = []  # not stateful

            def step(ob, *_args, **_kwargs):
                a, v = sess.run([a0, v0], {X: ob})
                return a, v, []  # dummy state

            def value(ob, *_args, **_kwargs):
                return sess.run(v0, {X: ob})

            self.X = X
            self.pi = pi
            self.vf = vf
            self.step = step
            self.value = value


class OurAcktrPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32) / 255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            p1 = pool(h)
            h2 = conv(p1, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            p2 = pool(h2)
            h3 = conv(p2, 'c3', nf=32, rf=3, stride=1, init_scale=np.sqrt(2))
            p3 = pool(h3)
            h3 = conv_to_fc(p3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x: x)
            vf = fc(h4, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = []  # not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X: ob})
            return a, v, []  # dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
