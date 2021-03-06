from collections import defaultdict
import math
import numpy as np
import tensorflow as tf
print(tf.__version__)
tf.compat.v1.enable_eager_execution()
from tensorflow.python.ops import array_ops
import torch
import torch.distributions as dist
from torch.distributions.utils import broadcast_all
from numbers import Number
import argparse



class Sparse_Concrete_tf(tf.distributions.Distribution):
    
    def mean(self):
      #  for 1 sample only !
        gamma = tf.squeeze(self.gamma, axis = 0)

        return (1 - gamma) * self.loc

    
    def stddev(self):
        return self.gamma * self.alpha + (1 - self.gamma) * self.scale

    def __init__(self, gamma, loc, scale):
        self._name = 'Spike-and-Slab Dist'
        self.gamma = tf.convert_to_tensor(gamma, dtype=np.float32)
        self.alpha = tf.constant(0.05, dtype=np.float32)

        self.loc = tf.convert_to_tensor(loc, dtype=np.float32)
        self.scale = tf.convert_to_tensor(scale, dtype=np.float32)
        super(Sparse_Concrete_tf, self).__init__(name=self._name,
                dtype=self.scale.dtype,
                reparameterization_type=tf.distributions.FULLY_REPARAMETERIZED,
                validate_args=False,
                allow_nan_stats=False)

    def _batch_shape_tensor(self):
        return array_ops.broadcast_dynamic_shape(array_ops.shape(self.loc), array_ops.shape(self.scale))
    
    @staticmethod
    def sample_logistic(shape, eps=1e-20):
        U = tf.random_uniform(shape, minval=0, maxval=1)
        return tf.math.log(U+eps) - tf.math.log1p(-U+eps)
    
    def sanity_check_count_ones_zeros(self, sample):
       samples = sample.numpy().tolist()
       #print(sample)
       prob_stat = [defaultdict(lambda:0), defaultdict(lambda:0)]
       print(len(sample[0]))
       for sample in samples:
           for ex_id, example in enumerate(sample):
               for val_id, val in enumerate(example):
                 #  print(val)
                   prob_stat[ex_id][val_id] += val

       for ex_id, prob in enumerate(prob_stat):
               for val_id, _ in enumerate(prob):
                 #  print(val)
                   prob_stat[ex_id][val_id] /= len(samples)


      # for row in sample:
      #     for idx, val in enumerate(row):
      #         prob_stat[idx] += val
      # for idx in prob_stat:
      #      prob_stat[idx] /= len(sample)
       return prob_stat
    
    def sample_binary_concrete(self, shape, probs, temperature=1.5, hard=True, eps=1e-20):
        L = self.sample_logistic(shape, eps=eps)
        scale = tf.math.reciprocal(temperature)

        logits =  tf.math.log(probs+eps) - tf.math.log1p(-probs+eps) # convert probability to logits
        sample = logits + L
        sample = tf.math.sigmoid(sample*scale)
        if hard:
            result = tf.where(sample <= 0.5,  tf.zeros_like(sample),  tf.ones_like(sample))
            sample = tf.stop_gradient(result-sample)+sample
   #     print('sample', sample)
        print('probs', self.sanity_check_count_ones_zeros(sample))
        return sample
    
    def straight_through(self):
        p = tf.compat.v1.distributions.Bernoulli(probs=(self.gamma ),  dtype=tf.dtypes.float32).sample()
        p = tf.stop_gradient(p)
        return p

    def sample(self, n, temperature):
        shape = tf.concat([[n], self._batch_shape_tensor()], axis=0)
        epsilon_2 = tf.keras.backend.random_normal(shape=shape, mean=0., stddev=1)
        epsilon_1 = tf.keras.backend.random_normal(shape=shape, mean=0., stddev=1)
        p = self.sample_binary_concrete(shape, self.gamma, temperature=temperature, hard=True, eps=1e-5)
        res = p * self.alpha * epsilon_1 + (1 - p) * (self.loc + self.scale * epsilon_2)
        return res
  
   
    def log_prob(self, value):
        res = tf.concat([tf.expand_dims((tf.distributions.Normal(loc=0.0, scale=self.alpha).log_prob(value) + tf.math.log(self.gamma+1e-5)), 0),\
            tf.expand_dims((tf.distributions.Normal(loc=self.loc, scale=self.scale).log_prob(value) + tf.math.log(1-self.gamma+1e-5)), 0)], axis=0)
        
        res= tf.math.reduce_logsumexp(res, 0)
        return res








class Sparse_tf(tf.distributions.Distribution):
    @property
    def mean(self):
        return (1 - self.gamma) * self.loc

    @property
    def stddev(self):
        return self.gamma * self.alpha + (1 - self.gamma) * self.scale

    def __init__(self, gamma, loc, scale):
        self._name = 'Sparse Dist'
        self.gamma = tf.convert_to_tensor(gamma, dtype=np.float32)
        self.alpha = tf.convert_to_tensor(0.05, dtype=np.float32)
        self.loc = tf.convert_to_tensor(loc, dtype=np.float32)
        self.scale = tf.convert_to_tensor(scale, dtype=np.float32)
       
        super(Sparse_tf, self).__init__(name=self._name,
                dtype=self.scale.dtype,
                reparameterization_type=tf.distributions.NOT_REPARAMETERIZED,
                validate_args=False,
                allow_nan_stats=False)

    def _batch_shape_tensor(self):
      return array_ops.broadcast_dynamic_shape(array_ops.shape(self.loc), array_ops.shape(self.scale))

    
    def sample(self, n): 
        shape = tf.concat([[n], self._batch_shape_tensor()], axis=0)
        epsilon_2 = tf.keras.backend.random_normal(shape=shape, mean=0., stddev=1)
        epsilon_1 = tf.keras.backend.random_normal(shape=shape, mean=0., stddev=1)

        p = tf.compat.v1.distributions.Bernoulli(probs=(self.gamma * tf.ones(shape)),  dtype=tf.dtypes.float32).sample()
        res = p * self.alpha * epsilon_1 + (1 - p) * (self.loc + self.scale * epsilon_2)
        return res
    
    def log_prob(self, value):
        res = tf.concat([tf.expand_dims((tf.distributions.Normal(loc=0.0, scale=self.alpha).log_prob(value) + tf.math.log(self.gamma)), 0),\
                        tf.expand_dims((tf.distributions.Normal(loc=self.loc, scale=self.scale).log_prob(value) + tf.math.log(1 - self.gamma)), 0)], axis=0)
        
        return tf.math.reduce_logsumexp(res, 0)



def sparsity_in_batch(data, norm=True):
    z_prev = []
    for batch_data in data:
            z = batch_data.numpy()
            if len(z_prev) > 0:
                z_prev = np.concatenate((z_prev, z), axis =0)
            else:
                z_prev =z
    
        
    print('z shape', z_prev.shape)
#    print('z:', z_prev)
    return compute_sparsity_tf(tf.convert_to_tensor(z_prev), norm=norm)



def compute_sparsity_tf(zs, norm):
    '''
    Hoyer metric
    norm: normalise input along dimension to avoid that dimension collapse leads to good sparsity
    '''
    latent_dim = float(zs.shape[-1].value)
    #print('norm', norm)
    if norm:
        zs = zs / tf.math.reduce_std(zs, axis = 0)
    l1 = tf.math.reduce_sum(tf.math.abs(zs), axis=-1)
   
    l2 = tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(zs,2), axis=-1))
    l1_l2 = tf.math.reduce_mean(l1/l2)
    return (tf.math.sqrt(latent_dim) - l1_l2) / (tf.math.sqrt(latent_dim) - 1)



def test_hoyer_for_n_inputs(n):
    dims = 32
    gamma = 0.8
    loc = np.full((1,dims), 0, dtype=np.float32)
    scale = np.full((1,dims), 1., dtype=np.float32)
    gamma_tf = np.full((1,dims), gamma, dtype=np.float32)
    p_tf = Sparse_tf(tf.convert_to_tensor(gamma_tf), loc, scale)
    for i in range(n):
        print('run:', i + 1)
        sample = p_tf.sample(512)
        sample = tf.squeeze(sample, axis=1).numpy()
        data = tf.data.Dataset.from_tensor_slices(sample)
        data = data.batch(4)
        
        hoyer_tf = compute_sparsity_tf(tf.convert_to_tensor(sample), norm=True)
        print('Hoyer tensorflow normalised', round(float(hoyer_tf), 3) )
        hoyer_tf = compute_sparsity_tf(tf.convert_to_tensor(sample), norm=False)
        print('Hoyer tensorflow (un)normalised', round(float(hoyer_tf), 3)  )

        print('****************')
        
        hoyer_tf = sparsity_in_batch(data, norm=True)
        print('Hoyer batch tensorflow normalised', round(float(hoyer_tf), 3) )
        hoyer_tf = sparsity_in_batch(data, norm=False)
        print('Hoyer batch tensorflow (un)normalised', round(float(hoyer_tf), 3)  )

        print('================')




def test_concrete_distribution(n_sample):
    dims = 5
    gamma =np.array( [[0.8,0.,0.1,0.5,0.3],  [0.1,0.2,0.9,0.8,0.1]], dtype=np.float32 )
    loc = np.full((1,dims), 0, dtype=np.float32)
    scale = np.full((1,dims), 1., dtype=np.float32)

    sp = Sparse_Concrete_tf(gamma, loc, scale)
    sp.sample(n_sample, temperature=0.5)


            
            
if __name__ == '__main__':
    test_hoyer_for_n_inputs(5)
    test_concrete_distribution(1000)
