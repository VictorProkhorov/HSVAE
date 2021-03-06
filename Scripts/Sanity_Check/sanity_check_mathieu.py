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


class Sparse_torch(dist.Distribution):
    has_rsample = False

    @property
    def mean(self):
        return (1 - self.gamma) * self.loc

    @property
    def stddev(self):
        return self.gamma * self.alpha + (1 - self.gamma) * self.scale

    def __init__(self, gamma, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        self.gamma = gamma
        self.alpha = torch.tensor(0.05).to(self.loc.device)
        if isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.scale.size()
        super(Sparse_torch, self).__init__(batch_shape)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.bernoulli(self.gamma * torch.ones(shape).to(self.loc.device))
        res = p * self.alpha * torch.randn(shape).to(self.loc.device) +  (1 - p) * (self.loc + self.scale * torch.randn(shape).to(self.loc.device))
        return res

    def log_prob(self, value):
        res = torch.cat([(dist.Normal(torch.zeros_like(self.loc), self.alpha).log_prob(value) + self.gamma.log()).unsqueeze(0),
                         (dist.Normal(self.loc, self.scale).log_prob(value) + (1 - self.gamma).log()).unsqueeze(0)],
                        dim=0)
        return torch.logsumexp(res, 0)



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
        #print('gamma',self.gamma)
        self.alpha = tf.convert_to_tensor(0.05, dtype=np.float32)
        self.loc = tf.convert_to_tensor(loc, dtype=np.float32)
        self.scale = tf.convert_to_tensor(scale, dtype=np.float32)
       
        super(Sparse_tf, self).__init__(name=self._name,
                dtype=self.scale.dtype,
                reparameterization_type=tf.distributions.NOT_REPARAMETERIZED,
                validate_args=False,
                allow_nan_stats=True)

    def _batch_shape_tensor(self):
      return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.loc),
        array_ops.shape(self.scale))

    
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





def imq_dim_kernel_torch(X, Y):
    assert X.shape == Y.shape
    batch_size, latent_dim = X.shape
    Xb = X.expand(batch_size, *X.shape)
    Yb = Y.expand(batch_size, *Y.shape)
    dists_x = (Xb - Xb.transpose(0, 1)).pow(2)
    dists_y = (Yb - Yb.transpose(0, 1)).pow(2)
    dists_c = (Xb - Yb.transpose(0, 1)).pow(2)
    stats = 0
    off_diag = 1 - torch.eye(batch_size, device=X.device)
    off_diag = off_diag.unsqueeze(-1).expand(*off_diag.shape, latent_dim)

    for scale in [.1, .2, .5, 1., 2., 5.]:
        C = 2 * scale  # 2 * latent_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)
        res1 = off_diag * res1
        res1 = res1.sum(0).sum(0) / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum(0).sum(0) * 2. / (batch_size)
        stats += (res1 - res2).sum()
    return stats / batch_size



def imq_dim_kernel_tf(Z_q, Z_p):
       
        batch_size = Z_q.shape[0].value
        latent_dim = Z_q.shape[1].value
        Yb = tf.broadcast_to(Z_q,[batch_size, Z_q.shape[0], Z_q.shape[1]])
        Xb = tf.broadcast_to(Z_p,[batch_size, Z_p.shape[0], Z_p.shape[1]])
        
        dists_x = tf.math.pow((Xb - tf.transpose(Xb, perm=[1, 0, 2])), 2)
        dists_y = tf.math.pow((Yb - tf.transpose(Yb, perm=[1, 0, 2])), 2)
        dists_c = tf.math.pow((Xb - tf.transpose(Yb, perm=[1, 0, 2])), 2)
        stats = 0.0
      
        off_diag = 1 - tf.eye(batch_size)
        off_diag =  tf.broadcast_to(tf.expand_dims(off_diag,-1), [off_diag.shape[0], off_diag.shape[1], latent_dim])
        for scale in [.1, .2, .5, 1., 2., 5.]:
            C = 2 * scale  # 2 * latent_dim * 1.0 * scale
            res1 = C / (C + dists_x)
            res1 += C / (C + dists_y)
            res1 = off_diag * res1
            res1 = tf.reduce_sum(tf.reduce_sum(res1, axis=0), axis=0) / (batch_size - 1)
            res2 = C / (C + dists_c)
            res2 = tf.reduce_sum(tf.reduce_sum(res2, axis=0), axis=0) * 2. / (batch_size)
           
            stats += tf.reduce_sum((res1 - res2))
        return stats / batch_size





def test_mmd_component_for_n_inputs(n):
    """
    This function tests the correctness of the reimplementation of the MMD term of the objective function introduced in Mathieu et al.
    """
    def test_mmd_component(X, Y):
        res_torch = imq_dim_kernel_torch(torch.tensor(X), torch.tensor(Y))
        res_tf =  imq_dim_kernel_tf(tf.convert_to_tensor(Y), tf.convert_to_tensor(X))

        print('mmd torch', round(float(res_torch), 3))
        print('mmd tensorflow', round(float(res_tf), 3))
    
    print('Result for N random normal vectors')
    for i in range(n):
        print('run:', i + 1)
        X = tf.keras.backend.random_normal(shape=[512,32], mean=0., stddev=1).numpy() 
        Y = tf.keras.backend.random_normal(shape=[512,32], mean=0., stddev=1).numpy()
        test_mmd_component(X, Y)
        print('================')

    print('Result for toy vectors')
    X = np.array([[1, 7.3,1], [-1.9, 0.2,0.8]], dtype=np.float32)
    Y = np.array([[9, -0.3,2], [-1, 1.2,-1]], dtype=np.float32)
    test_mmd_component(X, Y)

def test_log_prob_for_n_inputs(n):
    """
    This function tests the correctness of the reimplementation of the log_prob  of the Spike-and-Slab prior introduced in Mathieu et al.
    """
    dims = 32
    gamma = 0.8
    loc = np.full((1,dims), 0, dtype=np.float32)
    scale = np.full((1,dims), 1., dtype=np.float32)
    
    gamma_torch = torch.tensor(gamma)
    prior_torch = Sparse_torch(gamma_torch, torch.tensor(loc), torch.tensor(scale))
    
    gamma_tf = np.full((1,dims), gamma, dtype=np.float32)
    prior_tensorflow = Sparse_tf(tf.convert_to_tensor(gamma_tf), loc, scale)
    def test_log_prob(value):
        log_prob_tf = prior_tensorflow.log_prob(tf.convert_to_tensor(value))
        print('tensorflow log_prob matrix', log_prob_tf)
        log_prob_torch = prior_torch.log_prob(torch.tensor(value))
        print('torch log_prob matrix', log_prob_torch)
        print('****************')
        print('tensorflow log_prob mean of all values', round(float(tf.math.reduce_mean(log_prob_tf)), 3) )
        print('torch log_prob mean of all values', round(float(log_prob_torch.mean()), 3)  )


    for i in range(n):
        print('run:', i + 1)
        loc = tf.convert_to_tensor(np.random.choice([0.1, 0.2],  size=(512,dims), p=[0.5, 0.5]), dtype=np.float32)
        std = tf.convert_to_tensor(np.random.choice([1., 2.],  size=(512,dims), p=[0.5, 0.5]), dtype=np.float32)
        gaus = tf.distributions.Normal(loc=loc, scale=std)
        value = gaus.sample(1).numpy()
        test_log_prob(value)
        print('================')



def has_analytic_kl(type_p, type_q):
    return (type_p, type_q) in torch.distributions.kl._KL_REGISTRY

def kl_divergence_torch(p, q, samples=None):
    if has_analytic_kl(type(p), type(q)):
        return dist.kl_divergence(p, q)
    else:
        if samples is None:
            K = 10
            samples = p.rsample(torch.Size([K])) if p.has_rsample else p.sample(torch.Size([K]))
       
        ent = -p.log_prob(samples)
        kl_loss = (-ent - q.log_prob(samples)).mean(0)
        print('torch kl_loss matrix', kl_loss)
        return kl_loss

def kl_divergence_tf(samples, pz, qz_x):
    ent = -qz_x.log_prob(samples)
    p = pz.log_prob(samples)
    kl_loss = tf.math.reduce_mean(-ent -p, axis = 0)
    print('tensorflow kl_loss matrix', kl_loss)
    kl_loss = tf.math.reduce_mean(tf.math.reduce_sum(kl_loss, axis=-1), axis=-1)
    return kl_loss

def test_kl_div_for_n_inputs(n):
    """
    This function tests the correctness of the reimplementation of the kl divergence term of the objective function introduced in Mathieu et al.
    """
    dims = 32
    gamma = 0.5
    loc = np.full((1,dims), 0, dtype=np.float32)
    scale = np.full((1,dims), 1., dtype=np.float32)
    
    gamma_torch = torch.tensor(gamma)
    p_torch = Sparse_torch(gamma_torch, torch.tensor(loc), torch.tensor(scale))
    
    gamma_tf = np.full((1,dims), gamma, dtype=np.float32)
    p_tf = Sparse_tf(tf.convert_to_tensor(gamma_tf), loc, scale)

  
    
    for i in range(n):
        print('run:', i + 1)
        loc = np.random.choice([0.1, 0.2],  size=(512,dims), p=[0.5, 0.5])
        std =np.random.choice([1., 2.],  size=(512,dims), p=[0.5, 0.5])
        
        q_tf = tf.distributions.Normal(loc=tf.convert_to_tensor(loc, dtype=np.float32), scale=tf.convert_to_tensor(std, dtype=np.float32))
        q_torch = dist.Normal(torch.tensor(loc), torch.tensor(std))
        value = q_tf.sample(1).numpy()
        
        kl_div_torch =  kl_divergence_torch(q_torch, p_torch, samples=torch.tensor(value)).sum(-1).mean(-1)
        
        kl_div_tf =  kl_divergence_tf(tf.convert_to_tensor(value), p_tf, q_tf)
        print('torch kl_div loss value', round(float(kl_div_torch), 3))
        print('tensorflow kl_div loss value', round(float(kl_div_tf), 3))
        print('================')


def compute_sparsity_torch(zs, norm):
    '''
    Hoyer metric
    norm: normalise input along dimension to avoid that dimension collapse leads to good sparsity
    '''
    latent_dim = zs.size(-1)
    if norm:
        zs = zs / zs.std(0)
    l1_l2 = (zs.abs().sum(-1) / zs.pow(2).sum(-1).sqrt()).mean()
    return (math.sqrt(latent_dim) - l1_l2) / (math.sqrt(latent_dim) - 1)

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
        hoyer_torch = compute_sparsity_torch(torch.tensor(sample), True)
        print('Hoyer torch normalised', round(float(hoyer_torch), 3) )
        hoyer_tf = compute_sparsity_tf(tf.convert_to_tensor(sample), norm=True)
        print('Hoyer tensorflow normalised', round(float(hoyer_tf), 3) )
        print('****************')
        hoyer_torch = compute_sparsity_torch(torch.tensor(sample), False)
        print('Hoyer torch (un)normalised', round(float(hoyer_torch), 3) )
        hoyer_tf = compute_sparsity_tf(tf.convert_to_tensor(sample), norm=False)
        print('Hoyer tensorflow (un)normalised', round(float(hoyer_tf), 3)  )
        print('================')


if __name__ == '__main__':
    print(tf.__version__)
    descr = "Sanity check for implemented Mathieu et al. 2019 functions. In all experiments Tensorflow (GPU) 1.13.1 and python 3.5.1 were used."
    epil  = "See:  [V. Prokhorov, Y. Li, E. Shareghi, N. Collier (EMNLP 2020)]"
    parser = argparse.ArgumentParser(description=descr, epilog=epil)
    parser.add_argument('--mmd_test', required=False, type=int,
                         help='Third term of the objective function of Mathieu et al. 2019')
    parser.add_argument('--log_prob_test', required=False, type=int,
                         help='Log probability of Spike-and-Slab')
    parser.add_argument('--kl_div_test', required=False, type=int,
                         help='Second term of the objective function of Mathieu et al. 2019')
    parser.add_argument('--hoyer_test', required=False, type=int,
                         help='Hoyer of Mathieu et al. 2019')
    args = parser.parse_args()

    if args.mmd_test == 1:
        test_mmd_component_for_n_inputs(5)
    elif args.log_prob_test == 1:
        test_log_prob_for_n_inputs(5)
    elif args.kl_div_test == 1:
        test_kl_div_for_n_inputs(5)
    elif args.hoyer_test == 1:
        test_hoyer_for_n_inputs(5)







