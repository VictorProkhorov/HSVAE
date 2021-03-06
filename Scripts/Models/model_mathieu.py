import os
import sys
import math
import time
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
    unk = '<unk>'
   # unk = '_UNK'
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
        super(Sparse, self).__init__(name=self._name,
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
        


       




class Encoder(tf.keras.Model):
    def __init__(self, rnn_dim, z_dim, pz, alpha, beta):
        super(Encoder, self).__init__()
        self.alpha = alpha
        self.beta = beta
        print('alpha', self.alpha)
        print('beta', self.beta)
        self.pz = pz
        print('pz', self.pz)
        print('pz slab mean', pz.loc)
        print('pz alpha', pz.alpha)
        print('pz gamma', pz.gamma)
        self.q_density =  tf.distributions.Normal
        self.rnn = self._init_rnn(rnn_dim)
        self.mean = tf.keras.layers.Dense(z_dim, activation='linear')
        self.log_var = tf.keras.layers.Dense(z_dim, activation='linear')


    def call(self, x, embeddings):
        outputs = self.rnn(embeddings)
        original_sentences_length = tf.count_nonzero(x, 1, keepdims=False, dtype=tf.int32)
        output = last_relevant(outputs, original_sentences_length)
        mean = self.mean(output)
        log_var = self.log_var(output)
        std = tf.math.exp(0.5*log_var)
       
        qz_x =  tf.distributions.Normal(loc = mean, scale = std)
        z = qz_x.sample(1)
        
        kl = self.approx_kl_div_loss(z, self.pz, qz_x)
        z = tf.squeeze(z, axis=0)
        z_p =  tf.squeeze(self.pz.sample(z.shape[0].value), axis=1)
        reg = self.regulariser_cauchy(z, z_p)

        return z, kl*self.beta, reg*self.alpha
    
    def _init_rnn(self, units):
        return tf.compat.v1.keras.layers.CuDNNGRU(units,kernel_initializer='lecun_normal',recurrent_initializer='lecun_normal' , return_state=False, return_sequences=True)
        
    
    @staticmethod
    def approx_kl_div_loss(samples, pz, qz_x):
        
        ent = -qz_x.log_prob(samples)
        p = pz.log_prob(samples)
        # calculate the KL for one sample of z i.e z~(z|x)
        kl_loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.reduce_mean(-ent -p, axis=0), axis=-1), axis=-1)

        
        return kl_loss
    

    
    def regulariser_cauchy(self, Z_q, Z_p):
       # alpha =self.alpha
        batch_size = Z_q.shape[0].value
        latent_dim = Z_q.shape[1].value
        Yb = tf.broadcast_to(Z_q,[batch_size, Z_q.shape[0], Z_q.shape[1]])
        Xb = tf.broadcast_to(Z_p,[batch_size, Z_p.shape[0], Z_p.shape[1]])

        dists_x = tf.math.pow((Xb - tf.transpose(Xb, perm=[1, 0, 2])), 2)
        dists_y = tf.math.pow((Yb - tf.transpose(Yb, perm=[1, 0, 2])), 2)
        dists_c = tf.math.pow((Xb - tf.transpose(Yb, perm=[1, 0, 2])), 2)
        stats = 0.
    
        off_diag = 1 - tf.eye(batch_size)
        off_diag =  tf.broadcast_to(tf.expand_dims(off_diag, -1), [off_diag.shape[0], off_diag.shape[1], latent_dim])
        for scale in [0.1,0.2,0.5, 1.,2.,5. ]:
            C =  2*scale# 2 * latent_dim * 1 * scale
            res1 = C / (C + dists_x)
            res1 += C / (C + dists_y)
            res1 = off_diag * res1
            res1 = tf.reduce_sum(tf.reduce_sum(res1, axis=0), axis=0) / (batch_size - 1)
            res2 = C / (C + dists_c)
            res2 = tf.reduce_sum(tf.reduce_sum(res2, axis=0), axis=0) *2. / (batch_size)
            stats += tf.reduce_sum((res1 - res2))   
         
        return stats / batch_size


  
        
    
class Decoder_RNN(tf.keras.Model):
    def __init__(self, rnn_dim, vocab_size):
        super(Decoder_RNN, self).__init__()
        self.rnn = self._init_rnn(rnn_dim)
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
    def __init__(self, embed_dim, vocab_size, rnn_dim, z_dim, pz, alpha, beta):
        super(Sentence_VAE, self).__init__()
        self.embeddings = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=False)
        ### ENCODER ###
        self.encoder = Encoder(rnn_dim, z_dim,  pz, alpha, beta)
        ### DECODER ###
        self.decoder = Decoder_RNN(rnn_dim,  vocab_size)
        self.alpha = alpha
        self.beta = beta
            
   

    def call(self, x, batch_size, text_lang):
        ### ENCODING ###
        embeddings = self.embeddings(x)
        z, kl, reg = self.encoder(x, embeddings )
        

        ### DECODING ###
        dec_input = tf.expand_dims([text_lang.word2idx['<EOS>']] * batch_size, 1) 
        in_ = tf.concat([dec_input, x], axis=-1)
        embeddings = self.embeddings(in_[:,:-1])
        embeddings = tf.keras.layers.concatenate([embeddings,tf.keras.layers.RepeatVector(embeddings.shape[1])(z)])
        loss = self.decoder(x, embeddings)
        
        return loss, kl, reg

    def get_zs(self, x, is_sample=False):
        embeddings = self.embeddings(x)
        outputs = self.encoder.rnn(embeddings)
        original_sentences_length = tf.count_nonzero(x, 1, keepdims=False, dtype=tf.int32)
        output = last_relevant(outputs, original_sentences_length)
        mean = self.encoder.mean(output)

        if is_sample:
            log_var = self.encoder.log_var(output)
            std = tf.math.exp(0.5*log_var)
            qz_x =  tf.distributions.Normal(loc = mean, scale = std)
            z = qz_x.sample(1)
            z = tf.squeeze(z, axis=0)
            return z
        else:
            return mean




global_step = tf.train.get_or_create_global_step()


def train(epochs, buffer_size ,vae, dataset, optimizer, text_lang, n_batch, checkpoint, valid_data, valid_loss_file,train_loss_file, mmd_valid_plot_file, mmd_train_plot_file,kl_valid_plot_file, kl_train_plot_file, hoyer_sample_std_file, hoyer_sample_no_std_file, hoyer_mean_std_file, hoyer_mean_no_std_file,  gpu=None):
    device =gpu 
   
    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data).shuffle(len(valid_data))
    valid_batch_size = 128
    valid_dataset = valid_dataset.batch(valid_batch_size, drop_remainder=False)
    annealing_initial_step = 0
    with open(valid_loss_file, 'w') as valid_rec_l_f, open(train_loss_file, 'w') as train_rec_l_f, open(mmd_valid_plot_file, 'w') as mmd_valid_plot_f, open(mmd_train_plot_file, 'w') as mmd_train_plot_f, open(kl_valid_plot_file,'w') as kl_valid_plot_f, open(kl_train_plot_file,'w') as kl_train_plot_f, open(hoyer_sample_std_file,'w') as hoyer_s_std_f, open(hoyer_sample_no_std_file,'w') as hoyer_s_no_std_f, open(hoyer_mean_std_file,'w') as hoyer_m_std_f, open(hoyer_mean_no_std_file,'w') as hoyer_m_no_std_f:


    
        for epoch in range(1, epochs + 1):
            global_step.assign_add(1)
            start = time.time()
            total_loss = 0
            total_reg_loss = 0 # difference between C and KL
            total_kl_loss = 0 # actual KL
            dataset = dataset.shuffle(buffer_size)
            for (batch, sentences) in enumerate(dataset):
                batch_size = sentences.shape[0]
                annealing_initial_step += 1
                with tf.device(device):
                    with tf.GradientTape() as tape:
                        rec_loss, kl, reg = vae(sentences, batch_size, text_lang)
                        loss = tf.reduce_mean(rec_loss) +  reg +kl
            
                    variables = vae.trainable_variables 
                    gradients = tape.gradient(loss, variables)
                    optimizer.apply_gradients(zip(gradients, variables), global_step)
    
                batch_loss = loss 
                total_loss += tf.reduce_mean(rec_loss) #batch_loss
                total_reg_loss += reg
                total_kl_loss += kl

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Train Loss {:.4f}'.format(epoch,
                                                         batch,
                                                         batch_loss.numpy()))
                
    
            print('Epoch {}, NLL Loss {:.4f},  MMD Loss {:.4f}, KL Loss {:.4f}'.format(epoch, total_loss/n_batch,  total_reg_loss/n_batch, total_kl_loss/n_batch ))

            train_rec_l_f.write(str(epoch) +' ' + str(round((total_loss/n_batch).numpy(), 2) )+'\n')
            mmd_train_plot_f.write(str(epoch) +' ' + str(round((((total_reg_loss/n_batch)/vae.alpha)).numpy(), 2) )+'\n')
            kl_train_plot_f.write(str(epoch) +' ' + str(round(((total_kl_loss/n_batch)/vae.beta).numpy(), 2) )+'\n')
    

            if epoch % 1  == 0 or epoch == 1:
                with tf.device(device):
                    valid_rec_loss, valid_kl_loss, valid_reg_loss = evaluate_nnl_and_ppl_batch(valid_dataset, vae, text_lang)

            
                valid_rec_l_f.write(str(epoch) +' ' + str(round(valid_rec_loss,  2) )+'\n')
                mmd_valid_plot_f.write(str(epoch) +' ' + str(round(valid_reg_loss/vae.alpha, 2) )+'\n')
                kl_valid_plot_f.write(str(epoch) +' ' + str(round(valid_kl_loss/vae.beta, 2) )+'\n')


                is_sample = True
                norm = True
                hoyer_norm_sample = sparsity_in_batch(vae, valid_dataset, device, norm=norm, is_sample=is_sample)
                print('sparsity with std norm:', hoyer_norm_sample)
                norm = False
                hoyer_sample = sparsity_in_batch(vae, valid_dataset, device, norm=norm, is_sample=is_sample)
                print('sparsity without std norm:', hoyer_sample)
            
                is_sample =False
                norm = True
                hoyer_norm_mean = sparsity_in_batch(vae, valid_dataset, device, norm=norm, is_sample=is_sample)
                print('mean sparsity with std norm:', hoyer_norm_mean)
                norm = False
                hoyer_mean = sparsity_in_batch(vae, valid_dataset, device, norm=norm, is_sample=is_sample)
                print('mean sparsity without std norm:', hoyer_mean)
            
                hoyer_s_std_f.write(str(epoch) +' ' + str(round((hoyer_norm_sample).numpy(), 2) )+'\n')
                hoyer_s_no_std_f.write(str(epoch) +' ' + str(round((hoyer_sample).numpy(), 2) )+'\n')


                hoyer_m_std_f.write(str(epoch) +' ' + str(round((hoyer_norm_mean).numpy(), 2) )+'\n')
                hoyer_m_no_std_f.write(str(epoch) +' ' + str(round((hoyer_mean).numpy(), 2) )+'\n')


            
            
            
            
            
           

            print('Epoch {}; Valid Rec Loss {:.4f}; Valid KL-loss {:.4f}; Valid MMD-loss {:.4f}'.format(epoch, valid_rec_loss, valid_kl_loss, valid_reg_loss))
        

            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            save = checkpoint.save(file_prefix=checkpoint_prefix)
            print('save', save)


#### BASIC EVALUATION ####


def evaluate_nnl_and_ppl_batch(data, vae, text_lang, is_random_evaluate=False):

    valid_rec_loss = 0
    valid_reg_loss = 0
    valid_kl_loss= 0
    n_epochs = 0
    for x in data:
       # print('x:', x.shape)
        batch_size = x.shape[0]
        rec_loss, kl_loss, reg_loss  = vae(x, batch_size, text_lang )
        valid_rec_loss += tf.reduce_mean(rec_loss)
        valid_reg_loss += reg_loss
        valid_kl_loss += kl_loss
        n_epochs += 1
        
    valid_rec_loss /= n_epochs
    valid_reg_loss /= n_epochs
    valid_kl_loss /= n_epochs

    return float(valid_rec_loss), float(valid_kl_loss), float( valid_reg_loss)



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





def sparsity_in_batch(vae, data, gpu, norm=True, is_sample=False):
    z_prev = []
    for batch_data in data:
         with tf.device(gpu):
            z = vae.get_zs(batch_data, is_sample=is_sample).numpy()
            if len(z_prev) > 0:
                z_prev = np.concatenate((z_prev, z), axis =0)
            else:
                z_prev =z
    
        
    print('z shape', z_prev.shape)
    print('z:', z_prev)
    return compute_sparsity(tf.convert_to_tensor(z_prev), norm=norm)





if __name__ == "__main__":
    print(tf.__version__)
    descr = "Tensorflow (Eager) implementation of 'Disentangling Disentaglement in Variational Autoencoders by Mathieu et al (2019)'. In all experiments Tensorflow (GPU) 1.13.1 and python 3.5.1 were used."
    epil  = "None"
    parser = argparse.ArgumentParser(description=descr, epilog=epil)
    
    parser.add_argument('--corpus', required=True, type=str, default='CBT' ,help='')
    parser.add_argument('--alpha', required=True, type=float, default=1.0, help='')
    parser.add_argument('--beta', required=True, type=float, default=1.0, help='')

    parser.add_argument('--iter', required=True, type=int, default=1, help='')

    args = parser.parse_args()
    corpus = args.corpus 


    is_load = False

   
#    training_data_path = '../../Data/New_DBPedia/train.txt'
#    valid_data_path='../../Data/New_DBPedia/valid.txt'
 #   vocab_path = '../../Data/New_DBPedia/vocab.txt'
 

    # Yelp (Figure 1)
    training_data_path = '../../Data/Yelp/yelp.train_.txt'
    valid_data_path = '../../Data/Yelp/yelp.valid_.txt'
    vocab_path = None

    text_tensor, text_lang, mean_length_text = load_dataset(training_data_path, vocab_file=vocab_path)
    valid_text_tensor, _, _ = load_dataset(valid_data_path, 
    mean_length_text=mean_length_text, text_lang=text_lang, is_test_data=True)
    
    epochs = 15
    buffer_size = len(text_tensor)

    batch_size =  256#512
    n_batch = math.ceil(buffer_size/batch_size)
    vocab_size = len(text_lang.word2idx)
    print('vocab size:', vocab_size)
    embedding_dim = 256
    encoder_dim = 512
    z_dim =  32
    alpha = args.alpha
    beta = args.beta
    corpus = args.corpus
    iteration = args.iter 
    # Load Data #
    dataset = tf.data.Dataset.from_tensor_slices(text_tensor).shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    # Set an optimiser #
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0008)

    gamma = np.full((1,z_dim), 0.8)

    print('gamma', gamma)
    loc = np.full((1,z_dim), 0.)
    scale = np.full((1,z_dim), 1.)

    gpu = '/gpu:0'
    sparse_pz = Sparse(gamma, loc, scale) 
    # Creating the Model #
    vae = Sentence_VAE(embedding_dim, vocab_size, encoder_dim, z_dim,  sparse_pz, alpha, beta)
    name_of_experiment ='WAE_KL_alpha_'+str(alpha)+'_beta_'+str(beta)+'_GRU_corpus_'+corpus
    # Save Model #
    checkpoint_dir = '../../Data/Trained_Models/'+ name_of_experiment+'_iter_'+str(iteration)

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tfe.Checkpoint(optimizer=optimizer,
                                 vae=vae,
                                 optimizer_step=tf.train.get_or_create_global_step())
    
    
    experiment_dir = './WAE_KL_alpha_'+str(alpha)+'_beta_'+str(beta)+'_corpus_'+corpus+'/'
    
    
    valid_loss_file = experiment_dir+name_of_experiment + '_valid_rec_loss_'+str(iteration)+'.txt'
    train_loss_file = experiment_dir+name_of_experiment + '_train_rec_loss_'+str(iteration)+'.txt'
    mmd_train_plot_file = experiment_dir+name_of_experiment + '_valid_mmd_z_loss_'+str(iteration)+'.txt'
    mmd_valid_plot_file = experiment_dir+name_of_experiment + '_train_mmd_z_loss_'+str(iteration)+'.txt'
    kl_train_plot_file = experiment_dir+name_of_experiment + '_valid_kl_z_loss_'+str(iteration)+'.txt'
    kl_valid_plot_file = experiment_dir+name_of_experiment + '_train_kl_z_loss_'+str(iteration)+'.txt'
     
    hoyer_loss_sample_std_file = experiment_dir+name_of_experiment + '_hoyer_loss_sample_std_'+str(iteration)+'.txt'
    hoyer_loss_sample_no_std_file = experiment_dir+name_of_experiment + '_hoyer_loss_sample_no_std_'+str(iteration)+'.txt'
    hoyer_loss_mean_std_file = experiment_dir+name_of_experiment + '_hoyer_loss_mean_std_'+str(iteration)+'.txt'
    hoyer_loss_mean_no_std_file = experiment_dir+name_of_experiment + '_hoyer_loss_mean_no_std_'+str(iteration)+'.txt'


    
    
    # Select between loading the loading an existing Model and training a new model #
    if is_load == True:
        load_path = tf.train.latest_checkpoint(checkpoint_dir)
        print('load path', load_path)
        load = checkpoint.restore(load_path)
    else:
        train(epochs, buffer_size ,vae, dataset, optimizer, text_lang, n_batch, checkpoint, valid_text_tensor, valid_loss_file,train_loss_file, mmd_valid_plot_file, mmd_train_plot_file,kl_valid_plot_file, kl_train_plot_file, hoyer_loss_sample_std_file, hoyer_loss_sample_no_std_file, hoyer_loss_mean_std_file, hoyer_loss_mean_no_std_file,  gpu=gpu)
    

    
    # Report NLL and Perplexity on the test data #

    test_data_path = '../../Data/Yelp/yelp.test_.txt'
    
#    test_data_path='../../Data/New_DBPedia/test.txt'


    test_data, _, _ = load_dataset(test_data_path, 
    mean_length_text=mean_length_text, text_lang=text_lang, is_test_data=True)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
    test_batch_size = 128
    test_dataset = test_dataset.batch(test_batch_size, drop_remainder=False)
    

    with tf.device(gpu):
        test_rec_loss, test_kl_loss,  test_mmd_loss = evaluate_nnl_and_ppl_batch(test_dataset, vae, text_lang)
    print('BATCH: Test Rec Loss {:.4f}; Test KL-loss {:.4f}; Test MMD-loss {:.4f}'.format(test_rec_loss, test_kl_loss,  test_mmd_loss))
    
           


