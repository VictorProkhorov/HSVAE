import os
import sys
import math
import time
import glob
import tensorflow as tf
print(tf.__version__)
tf.enable_eager_execution()
tf.reset_default_graph()
import pickle
from tensorflow.keras import backend as K
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
import tensorflow.contrib.eager as tfe
import numpy as np
from scipy.linalg import orth
import six
from tensorflow.python.ops import array_ops
import argparse
#np.set_printoptions(threshold=sys.maxsize)

# many parts of the code were taken from https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb

is_gpu = tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)

if is_gpu == True:
    print("Using GPU")
else:
    print("Using CPU")

#### LOADING DATA ####

class LanguageIndex():
    def  __init__(self, lang=None, vocab=set(), is_vocab=False):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = vocab
        if is_vocab == False:
            self.create_index_with_sentences()
        else:
            self.vocab = sorted(self.vocab)
            self.create_index_with_vocab()
    
    def create_index_with_vocab(self):
        
        self.vocab.append('<EOS>')
        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1
        for word, index in self.word2idx.items():
            self.idx2word[index] = word
  
    def create_index_with_sentences(self):
        for sentence in self.lang:
            self.vocab.update(sentence.split(' '))
        self.vocab = sorted(self.vocab)
        self.vocab.append('<EOS>')
        if '_UNK' not in self.vocab:
            self.vocab.append('_UNK')
        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            # + 1 to take into account <pad>
            self.word2idx[word] = index + 1
    
        for word, index in self.word2idx.items():
            self.idx2word[index] = word
    

def preprocess_sentence(sentence):
    #w = unicode_to_ascii(w.strip())
   # sentence =  sentence
    return sentence


def create_dataset(path):
    
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    word_pairs = [preprocess_sentence(line.lstrip().rstrip()) for line in lines if len(line) > 0]
    
    return word_pairs


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0., eos=1.):
    """
    Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.
        eos = end of sentence index to end each sentence
    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}:\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen+1) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen] + [float(eos)]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x




def get_vocab(vocab_file):
    vocab = set()
    with open(vocab_file, 'r') as f:
        for idx,word in enumerate(f):
           # if not word.rstrip():
           #     print ('hi', word, idx+1)
            word = word.rstrip()
            vocab.add(word)
    return vocab


def load_dataset(path, mean_length_text=None, text_lang=None, is_test_data=False, vocab_file=None):
    # creating cleaned input, output pairs
    sentences = create_dataset(path)
    print('Number of sentences:', len(sentences))
    # index language using the class defined above   
    if text_lang == None: 
        if vocab_file is not None:
            vocab = get_vocab(vocab_file)
            print('loaded vocab:', len(vocab))
            text_lang = LanguageIndex(vocab=vocab, is_vocab=True)
        else:
            text_lang = LanguageIndex(lang=sentences)
    
    print('Number of unique words in text:', len(text_lang.vocab))
    
   
    # text definitions of concepts
   # unk = '<unk>'
    unk = '_UNK'
    text_tensor = [[text_lang.word2idx[word] if word in text_lang.word2idx else text_lang.word2idx[unk] for word in txt.split(' ')] for txt in sentences]
    
    
    def max_length(tensor):
        return max([len(t) for t in tensor])

    max_length_text = max_length(text_tensor)

    # Padding the input and output tensor to the maximum length
    text_tensor = pad_sequences(text_tensor, maxlen=max_length_text, padding='post', truncating='post', eos=text_lang.word2idx['<EOS>'])
    
    max_length_text = max_length_text + 1 # add one to compensate for <EOS>
    print('the longest sentence is of N symbols:', max_length_text)
    return text_tensor, text_lang, max_length_text 


#### MODELS ####

def last_relevant(output, length):
    "taken form: https://danijar.com/variable-sequence-lengths-in-tensorflow/"
    #print('out', output.shape)
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length -1 )
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant



def log_sum_exp(value, dim=None):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    #print('value', value.shape)
    m = tf.keras.backend.max(value, axis = dim, keepdims=True)
  
    value0 = value - m
  
    log_qz = tf.math.log(tf.reduce_sum(tf.keras.backend.exp(value0), axis=dim, keepdims=True))
  
    
    return m + log_qz


class Sparse(tf.distributions.Distribution):
    
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
        super(Sparse, self).__init__(name=self._name,
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
    
    
    
    def sample_binary_concrete(self, shape, probs, temperature=1.5, hard=True, eps=1e-20):
        L = self.sample_logistic(shape, eps=eps)
        scale = tf.math.reciprocal(temperature)

        logits =  tf.math.log(probs+eps) - tf.math.log1p(-probs+eps) # convert probability to logits
        sample = logits + L
        sample = tf.math.sigmoid(sample*scale)
        if hard:
            result = tf.where(sample <= 0.5,  tf.zeros_like(sample),  tf.ones_like(sample))
            sample = tf.stop_gradient(result-sample)+sample
       # print('sample',sample)
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


       




class Encoder(tf.keras.Model):
    def __init__(self, rnn_dim, z_dim, z_reg_weight, gamma_reg_weight, beta, alpha):
        super(Encoder, self).__init__()
        
        # Prior 
        self.pz_loc = np.full((1, z_dim), 0.)
        self.pz_scale = np.full((1, z_dim), 1.)     
        
        self.p_alpha = [alpha]*z_dim # 30
        self.p_beta = [beta]*z_dim #10
        print("prior alpha", alpha)
        print("prior beta", beta)
        self.p_gamma = tf.distributions.Beta(self.p_alpha, self.p_beta)
        print('prior over gamma', self.p_gamma)
        
        # variational parameters 
        self.rnn = self._init_rnn(rnn_dim)
        self.mean = tf.keras.layers.Dense(z_dim, activation='linear')
        self.log_var = tf.keras.layers.Dense(z_dim, activation='linear')
        self.q_alpha = tf.keras.layers.Dense(z_dim, activation='softplus')
        self.q_beta = tf.keras.layers.Dense(z_dim, activation='softplus')
        
        # weights on losses
        self.z_reg_weight = z_reg_weight# regularisation weight on z
        print('z reg weight', self.z_reg_weight)
        self.gamma_reg_weight = gamma_reg_weight# regularisation weight on gamma
        print('gamma reg weight', self.gamma_reg_weight)
        

    
    def call(self, x, embeddings, temperature):
        outputs = self.rnn(embeddings)
        original_sentences_length = tf.count_nonzero(x, 1, keepdims=False, dtype=tf.int32)
        output = last_relevant(outputs, original_sentences_length)
        
        
        
        q_alpha = self.q_alpha(output)+1e-5 
        q_beta = self.q_beta(output) +1e-5
        mean = self.mean(output)
        log_var = self.log_var(output)
        std = tf.math.exp(0.5*log_var) + 1e-5
        q_gamma_x = tf.distributions.Beta(q_alpha, q_beta)
        with tf.device('cpu'):
            gamma = q_gamma_x.sample(1)
    
        kl_gamma = tf.distributions.kl_divergence(q_gamma_x, self.p_gamma)
        kl_gamma = tf.math.reduce_mean(tf.math.reduce_sum(kl_gamma, axis=-1), axis=0)


        pz_gamma = Sparse(gamma, self.pz_loc, self.pz_scale) 
        qz_x_gamma = Sparse(gamma, mean, std) 
        
        # for now assume one sample of gamma
        zs = qz_x_gamma.sample(1, temperature) 
        kl_z = self.approx_kl_div_z_loss(zs, qz_x_gamma, pz_gamma)
        
    
        return tf.squeeze(zs, axis=0), self.gamma_reg_weight*kl_gamma, self.z_reg_weight*kl_z
    
    def _init_rnn(self, units):
        return tf.compat.v1.keras.layers.CuDNNGRU(units,kernel_initializer='lecun_normal',recurrent_initializer='lecun_normal' , return_state=False, return_sequences=True)
    
    @staticmethod
    def approx_kl_div_z_loss(samples, qz_x, pz):
        log_q = qz_x.log_prob(samples)
        log_p = pz.log_prob(samples) 
        # calculate the KL for one sample of z i.e z~(z|x)
        kl_loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.reduce_mean(log_q -log_p, axis=0), axis=-1), axis=-1)
    
        return kl_loss
                                                                                                         
    
    
 
  
        
    
class Decoder_RNN(tf.keras.Model):
    def __init__(self, rnn_dim, vocab_size):
        super(Decoder_RNN, self).__init__()
        self.rnn = self._init_rnn(rnn_dim)
        self.z_expander = tf.keras.layers.Dense(rnn_dim, activation='linear')
        self.out = tf.keras.layers.Dense(vocab_size)

    def call(self, x, embeddings):
        
        result = self.rnn(embeddings)
        hs = result[0]
        predictions = self.out(hs)
        loss = self.reconstruction_loss_function(x, predictions)
        return loss
    
    def _init_rnn(self, units):
        return tf.compat.v1.keras.layers.CuDNNGRU(units, kernel_initializer='lecun_normal',recurrent_initializer='lecun_normal', return_state=True, return_sequences=True)
        
    @staticmethod
    def reconstruction_loss_function(real, pred):
        mask = 1 - np.equal(real, 0)
        x_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        xe_loss = tf.reduce_sum(x_, axis=-1)
        return xe_loss





class Sentence_VAE(tf.keras.Model):
    def __init__(self, embed_dim, vocab_size, rnn_dim, z_dim, z_reg_weight, gamma_reg_weight, beta, alpha):
        super(Sentence_VAE, self).__init__()
        self.embeddings = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=False)
        ### ENCODER ###
        self.encoder = Encoder(rnn_dim, z_dim, z_reg_weight, gamma_reg_weight, beta, alpha)
        ### DECODER ###
        self.decoder = Decoder_RNN(rnn_dim,  vocab_size)
            
   

    def call(self, x, batch_size, text_lang, temperature):
        ### ENCODING ###
        embeddings = self.embeddings(x)
        z, kl_gamma, kl_z = self.encoder(x, embeddings,temperature )
        

        ### DECODING ###
        dec_input = tf.expand_dims([text_lang.word2idx['<EOS>']] * batch_size, 1) 
        in_ = tf.concat([dec_input, x], axis=-1)
        embeddings = self.embeddings(in_[:,:-1])
        embeddings = tf.keras.layers.concatenate([embeddings,tf.keras.layers.RepeatVector(embeddings.shape[1])(z)])
        loss = self.decoder(x, embeddings)
        
        return loss, kl_gamma, kl_z

    def get_zs(self, x, temperature, is_sample=False, is_sp=False):
        embeddings = self.embeddings(x)
        outputs = self.encoder.rnn(embeddings)
        original_sentences_length = tf.count_nonzero(x, 1, keepdims=False, dtype=tf.int32)
        output = last_relevant(outputs, original_sentences_length)
        
        q_alpha = self.encoder.q_alpha(output)
        q_beta = self.encoder.q_beta(output)
        mean = self.encoder.mean(output)
        log_var = self.encoder.log_var(output)
        std = tf.math.exp(0.5*log_var)
        # distributions over gamma
        if is_sample and not is_sp:
            q_gamma_x = tf.distributions.Beta(q_alpha, q_beta)
            with tf.device('cpu'):
                gamma = q_gamma_x.sample(1)
            gamma = tf.squeeze(gamma, axis = 0)
            # distributions over z
            pz_gamma = Sparse(gamma, self.encoder.pz_loc, self.encoder.pz_scale) 
            qz_x_gamma = Sparse(gamma, mean, std) 
        
            # losses
            # for now assume one sample of gamma
            # sample here
            zs = qz_x_gamma.sample(1, temperature) 
            zs = tf.squeeze(zs, axis=0)
            return zs
        elif not is_sample and is_sp:
            q_gamma_x = tf.distributions.Beta(q_alpha, q_beta)
            with tf.device('cpu'):
                gamma = q_gamma_x.sample(1)
           # gamma = tf.squeeze(gamma, axis = 0)
            # distributions over z
            qz_x_gamma = Sparse(gamma, mean, std) 
            mean_sp =  qz_x_gamma.mean()
            return mean_sp

        elif not is_sample and not is_sp:
            return mean
        else:
            print('ERROR: Cannot get sample')
            exit()
       
    def get_alpha_beta_params(self, x):
        embeddings = self.embeddings(x)
        outputs = self.encoder.rnn(embeddings)
        original_sentences_length = tf.count_nonzero(x, 1, keepdims=False, dtype=tf.int32)
        output = last_relevant(outputs, original_sentences_length)
        
        q_alpha = self.encoder.q_alpha(output)
        q_beta = self.encoder.q_beta(output)
        return q_alpha, q_beta
     
    def get_mean_of_beta_distribution(self, x):
        embeddings = self.embeddings(x)
        outputs = self.encoder.rnn(embeddings)
        original_sentences_length = tf.count_nonzero(x, 1, keepdims=False, dtype=tf.int32)
        output = last_relevant(outputs, original_sentences_length)
        
        q_alpha = self.encoder.q_alpha(output)
        q_beta = self.encoder.q_beta(output)
        q_gamma_x = tf.distributions.Beta(q_alpha, q_beta)
        return q_gamma_x.mean()

    def sample_k_zs(self, x, temperature, n_samples=5):
        embeddings = self.embeddings(x)
        outputs = self.encoder.rnn(embeddings)
        original_sentences_length = tf.count_nonzero(x, 1, keepdims=False, dtype=tf.int32)
        output = last_relevant(outputs, original_sentences_length)
        
        q_alpha = self.encoder.q_alpha(output)
        q_beta = self.encoder.q_beta(output)
        mean = self.encoder.mean(output)
        log_var = self.encoder.log_var(output)
        std = tf.math.exp(0.5*log_var)
        # distributions over gamma
        q_gamma_x = tf.distributions.Beta(q_alpha, q_beta)
        with tf.device('cpu'):
            gamma = q_gamma_x.sample(n_samples)
        
            # distributions over z 
    #    print('gamma', gamma)
        qz_x_gamma = Sparse(gamma, mean, std) 
        zs = qz_x_gamma.sample(n_samples, temperature) 
        #print('zs', zs)
        return zs
        




global_step = tf.train.get_or_create_global_step()


def train(epochs, buffer_size, vae, dataset, optimizer, text_lang, n_batch, checkpoint, valid_data, target_temperature ,gpu=None):
    device =gpu 
   
    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data).shuffle(len(valid_data))
    valid_batch_size = 128
    valid_dataset = valid_dataset.batch(valid_batch_size, drop_remainder=False)
    annealing_initial_step = 0.
    for epoch in range(1, epochs + 1):
        global_step.assign_add(1)
        start = time.time()
        total_rec_loss = 0.
        total_kl_z_loss = 0. 
        total_kl_gamma_loss = 0. 
        dataset = dataset.shuffle(buffer_size)
        for (batch, sentences) in enumerate(dataset):
            batch_size = sentences.shape[0]
            annealing_initial_step += 1.
            with tf.device(device):
                with tf.GradientTape() as tape:
                    rec_loss, kl_gamma, kl_z = vae(sentences, batch_size, text_lang,target_temperature)
                    loss = tf.reduce_mean(rec_loss) +  kl_gamma + kl_z
    
                variables = vae.trainable_variables 
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables), global_step)
    
            batch_loss = loss 
            total_rec_loss += tf.reduce_mean(rec_loss) 
            total_kl_z_loss += kl_z
            total_kl_gamma_loss += kl_gamma
            if batch % 100 == 0:
                print('Epoch {} Batch {} Train Loss {:.4f}'.format(epoch, batch, batch_loss.numpy()))
                
    
        print('Epoch {}, Recon. Loss {:.4f},  KL Gamma {:.4f}, KL Z {:.4f}'.format(epoch, total_rec_loss/n_batch,  total_kl_gamma_loss/n_batch, total_kl_z_loss/n_batch ))


            
        if epoch % 1  == 0 or epoch == 1:
            with tf.device(device):
                valid_rec_loss, valid_reg_gamma_loss, valid_reg_z_loss = evaluate_nnl_and_ppl_batch(valid_dataset, vae, text_lang, target_temperature)

            is_sample = True
            norm = True
            is_sp =False
            hoyer_norm_sample = sparsity_in_batch(vae, valid_dataset, device, target_temperature, norm=norm,is_sample=is_sample, is_sp=is_sp)
            print('sample: sparsity with std norm:', hoyer_norm_sample)
            norm = False
            hoyer_sample = sparsity_in_batch(vae, valid_dataset, device, target_temperature, norm=norm, is_sample=is_sample, is_sp = is_sp)
            print('sample: sparsity without std norm:', hoyer_sample)
            
            is_sample = False
            is_sp =False
            norm = True
            hoyer_norm_mean = sparsity_in_batch(vae, valid_dataset, device, target_temperature, norm=norm,is_sample=is_sample, is_sp=is_sp)
            print('mean: sparsity with std norm:', hoyer_norm_mean)
            norm = False
            hoyer_mean = sparsity_in_batch(vae, valid_dataset, device, target_temperature, norm=norm, is_sample=is_sample, is_sp=is_sp)
            print('mean: sparsity without std norm:', hoyer_mean)
                
            is_sample = False
            is_sp =True
            norm = True
            hoyer_norm_sp_mean = sparsity_in_batch(vae, valid_dataset, device, target_temperature, norm=norm,is_sample=is_sample, is_sp=is_sp)
            print('sp mean: sparsity with std norm:', hoyer_norm_sp_mean)
            norm = False
            hoyer_sp_mean = sparsity_in_batch(vae, valid_dataset, device, target_temperature, norm=norm, is_sample=is_sample, is_sp=is_sp)
            print('sp mean: sparsity without std norm:', hoyer_sp_mean)
                

        print('Epoch {}; Valid Rec Loss {:.4f};  Valid Gamma KL-loss {:.4f}; Valid Z KL-loss {:.4f} '.format(epoch,valid_rec_loss, valid_reg_gamma_loss, valid_reg_z_loss))
                
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
 


#### BASIC EVALUATION ####


def evaluate_nnl_and_ppl_batch(data, vae, text_lang, temperature, is_random_evaluate=False):

    valid_rec_loss = 0.
    valid_reg_gamma_loss = 0.
    valid_reg_z_loss = 0.
    n_epochs = 0
    for x in data:
    
        batch_size = x.shape[0]
        rec_loss, reg_gamma, reg_z  = vae(x, batch_size, text_lang, temperature )
        valid_rec_loss += tf.reduce_mean(rec_loss)
        valid_reg_gamma_loss += reg_gamma
        valid_reg_z_loss += reg_z
        n_epochs += 1
        
    valid_rec_loss /= n_epochs
    valid_reg_gamma_loss /= n_epochs
    valid_reg_z_loss /= n_epochs
    return float(valid_rec_loss), float( valid_reg_gamma_loss), float(valid_reg_z_loss)



def compute_sparsity(zs, norm):
    '''
    Hoyer metric
    norm: normalise input along dimension to avoid that dimension collapse leads to good sparsity
    '''
    latent_dim = float(zs.shape[-1].value)
    if norm:
        zs = zs / tf.math.reduce_std(zs, axis = 0)
    
    l1 = tf.math.reduce_sum(tf.math.abs(zs), axis=-1)
    l2 = tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(zs,2), axis=-1))
    l1_l2 = tf.math.reduce_mean(l1/l2)
    return (tf.math.sqrt(latent_dim) - l1_l2) / (tf.math.sqrt(latent_dim) - 1)





def sparsity_in_batch(vae, data, gpu, temperature, norm=True, is_sample=False, is_sp=False):
    z_prev = []
    for batch_data in data:
         with tf.device(gpu):
            z = vae.get_zs(batch_data, temperature, is_sample=is_sample, is_sp=is_sp).numpy()
            if len(z_prev) > 0:
                z_prev = np.concatenate((z_prev, z), axis =0)
            else:
                z_prev =z
    
        
    print('z shape', z_prev.shape)
    print('z:', z_prev)
    return compute_sparsity(tf.convert_to_tensor(z_prev), norm=norm)





if __name__ == "__main__":
    print(tf.__version__)
    descr = "Tensorflow (Eager) implementation for HSVAE   In all experiments Tensorflow (GPU) 1.13.1 and python 3.5.1 were used."
    epil  = "None"
    parser = argparse.ArgumentParser(description=descr, epilog=epil)
    parser.add_argument('--z_reg_weight', required=True, type=float, default=1. ,help='weight (psi) of  the first KL term')
    parser.add_argument('--gamma_reg_weight', required=True, type=float, default=1. ,help='weight (lambda) of the second KL term')
    parser.add_argument('--temperature', required=True, type=float, default=0.5 ,help='temperature of the Binary Concrete Distribution')
    parser.add_argument('--alpha', required=True, type=float, default=30. ,help='alpha parameter of the Beta (prior) distribution')
    parser.add_argument('--beta', required=True, type=float, default=30. ,help='beta parameter of the Beta (prior) distribution')
    parser.add_argument('--iter', required=True, type=int, default=1 ,help='run of the model')
    parser.add_argument('--corpus', required=True, type=str, default='DBpedia' ,help='Name of a corpus')

    args = parser.parse_args()
    corpus = args.corpus 
    is_load = False
   
    training_data_path = '../Data/New_DBPedia/train.txt'
    valid_data_path='../Data/New_DBPedia/valid.txt'
    vocab_path = '../Data/New_DBPedia/vocab.txt'
    


    text_tensor, text_lang, mean_length_text = load_dataset(training_data_path, vocab_file=vocab_path)
    valid_text_tensor, _, _ = load_dataset(valid_data_path, 
    mean_length_text=mean_length_text, text_lang=text_lang, is_test_data=True)
    
    epochs = 15
    buffer_size = len(text_tensor)
    batch_size = 512
    n_batch = math.ceil(buffer_size/batch_size)
    vocab_size = len(text_lang.word2idx)
    print('vocab size:', vocab_size)
    embedding_dim = 256
    encoder_dim = 512
    z_dim =  32#16
    z_reg_weight = args.z_reg_weight
    gamma_reg_weight = args.gamma_reg_weight
    target_temperature = args.temperature
    beta = args.beta
    alpha = args.alpha
    
    # Load Data #
    dataset = tf.data.Dataset.from_tensor_slices(text_tensor).shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    # Set an optimiser #
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0008)


    iteration = args.iter

    gpu = '/gpu:0'
    # Creating the Model #
    vae = Sentence_VAE(embedding_dim, vocab_size, encoder_dim, z_dim, z_reg_weight, gamma_reg_weight, beta, alpha)
    name_of_experiment ='none_1' # z_dim was not added when trained with 16 dims 
    # Save Model #
    checkpoint_dir = 'none_2'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tfe.Checkpoint(optimizer=optimizer,
                                 vae=vae,
                                 optimizer_step=tf.train.get_or_create_global_step())
    experiment_dir = './model'
    
    
    


    # Select between loading the loading an existing Model and training a new model #
    if is_load == True:
        load_path = tf.train.latest_checkpoint(checkpoint_dir)
        print('load path', load_path)
        load = checkpoint.restore(load_path)
    else:
        train(epochs, buffer_size ,vae, dataset, optimizer, text_lang, n_batch, checkpoint, valid_text_tensor, target_temperature, gpu=gpu)
    
    test_data_path='../Data/New_DBPedia/test.txt'

    test_data, _, _ = load_dataset(test_data_path, 
    mean_length_text=mean_length_text, text_lang=text_lang, is_test_data=True)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
    test_batch_size = 128
    test_dataset = test_dataset.batch(test_batch_size, drop_remainder=False)
    

            

    with tf.device(gpu):
        test_rec_loss, test_reg_gamma_loss, test_reg_z_loss = evaluate_nnl_and_ppl_batch(test_dataset, vae, text_lang, target_temperature)


    print('BATCH: Test Rec Loss {:.4f};  Test KL-loss z {:.4f}, Test KL-loss Gamma {:.4f}'.format(test_rec_loss, test_reg_z_loss, test_reg_gamma_loss))
    print(vae.summary()) 
    
