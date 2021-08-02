import os
import sys
import math
import time
import tensorflow as tf
print(tf.__version__)
import pickle
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
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
    sentences = [preprocess_sentence(line.lstrip().rstrip()) for line in lines if len(line) > 0]
    
    return sentences




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
    unk = '_UNK'
    text_tensor = [[text_lang.word2idx[word] if word in text_lang.word2idx else text_lang.word2idx[unk] for word in sentence.split(' ')+['<EOS>']] for sentence in sentences]
    

    
    def max_length(tensor):
        return max([len(t) for t in tensor])

    max_length_text = max_length(text_tensor)

    # Padding the input and output tensor to the maximum length
    text_tensor = tf.keras.preprocessing.sequence.pad_sequences(text_tensor, maxlen=max_length_text, padding='post', truncating='post')

    
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




class Sparse(tfp.distributions.Distribution):
    @property
    def mean(self):
        return (1 - self.gamma) * self.loc

    @property
    def stddev(self):
        return self.gamma * self.alpha + (1 - self.gamma) * self.scale

    def __init__(self, gamma, loc, scale):
        self._name = 'Spike-and-Slab Dist'
        self.gamma = tf.convert_to_tensor(gamma, dtype=np.float32)
        self.alpha = tf.convert_to_tensor(0.05, dtype=np.float32)

        self.loc = tf.convert_to_tensor(loc, dtype=np.float32)
        self.scale = tf.convert_to_tensor(scale, dtype=np.float32)
        super(Sparse, self).__init__(name=self._name,
                dtype=self.scale.dtype,
                reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
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
              
        res = tf.concat([tf.expand_dims((tfp.distributions.Normal(loc=0.0, scale=self.alpha).log_prob(value) + tf.math.log(self.gamma)), 0),\
                     tf.expand_dims((tfp.distributions.Normal(loc=self.loc, scale=self.scale).log_prob(value) + tf.math.log(1 - self.gamma)), 0)], axis=0)
        
      
        res= tf.math.reduce_logsumexp(res, 0)
        return res


       




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
        #self.q_density =  tfp.distributions.Normal
        self.rnn = self._init_rnn(rnn_dim)
        self.mean = tf.keras.layers.Dense(z_dim, activation='linear')
        self.std = tf.keras.layers.Dense(z_dim, activation='softplus')


    def call(self, x, embeddings):
        outputs = self.rnn(embeddings)
        original_sentences_length = tf.math.count_nonzero(x, 1, keepdims=False, dtype=tf.int32)
        output = last_relevant(outputs, original_sentences_length)
        mean = self.mean(output)
        std = self.std(output)
        
       
        qz_x =  tfp.distributions.Normal(loc = mean, scale = std)
        z = qz_x.sample(1)
        
        kl = self.approx_kl_div_loss(z, self.pz, qz_x)
        z = tf.squeeze(z, axis=0)
        z_p =  tf.squeeze(self.pz.sample(z.shape[0]), axis=1)
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
        batch_size = Z_q.shape[0]
        latent_dim = Z_q.shape[1]
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
        original_sentences_length = tf.math.count_nonzero(x, 1, keepdims=False, dtype=tf.int32)
        output = last_relevant(outputs, original_sentences_length)
        mean = self.encoder.mean(output)

        if is_sample:
            std = self.encoder.std(output)
            qz_x =  tfp.distributions.Normal(loc = mean, scale = std)
            z = qz_x.sample(1)
            z = tf.squeeze(z, axis=0)
            return z
        else:
            return mean






def train(epochs, buffer_size ,vae, dataset, optimizer, text_lang, n_batch, checkpoint, valid_data, device=None):
   
    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data).shuffle(len(valid_data))
    valid_batch_size = 128
    valid_dataset = valid_dataset.batch(valid_batch_size, drop_remainder=False)
    
    for epoch in range(1, epochs + 1):
        start = time.time()
        total_loss = 0.
        total_reg_loss = 0.
        total_kl_loss = 0. 
        dataset = dataset.shuffle(buffer_size)
        for (batch, sentences) in enumerate(dataset):
            batch_size = sentences.shape[0]
            with tf.device(device):
                with tf.GradientTape() as tape:
                    rec_loss, kl, reg = vae(sentences, batch_size, text_lang)
                    loss = tf.reduce_mean(rec_loss) +  reg +kl
            
                variables = vae.trainable_variables 
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables)) 
    
            batch_loss = loss 
            total_loss += tf.reduce_mean(rec_loss)
            total_reg_loss += reg
            total_kl_loss += kl

            if batch % 100 == 0:
                print('Epoch {} Batch {} Train Loss {:.4f}'.format(epoch, batch, batch_loss.numpy()))
                
    
        print('Epoch {}, NLL Loss {:.4f},  MMD Loss {:.4f}, KL Loss {:.4f}'.format(epoch, total_loss/n_batch,  total_reg_loss/n_batch, total_kl_loss/n_batch ))

            

        if epoch % 1  == 0 or epoch == 1:
            with tf.device(device):
                valid_rec_loss, valid_kl_loss, valid_reg_loss = evaluate_nnl_and_kls_batch(valid_dataset, vae, text_lang)

            
            

            is_sample = True
            norm = True
            hoyer_norm_sample = sparsity_in_batch(vae, valid_dataset, device, norm=norm, is_sample=is_sample)
            print('(gaussian dist) sample sparsity with std norm:', hoyer_norm_sample)
            norm = False
            hoyer_sample = sparsity_in_batch(vae, valid_dataset, device, norm=norm, is_sample=is_sample)
            print('(gaussian dist) sample sparsity without std norm:', hoyer_sample)
            
            is_sample =False
            norm = True
            hoyer_norm_mean = sparsity_in_batch(vae, valid_dataset, device, norm=norm, is_sample=is_sample)
            print('(gaussain dist) mean sparsity with std norm:', hoyer_norm_mean)
            norm = False
            hoyer_mean = sparsity_in_batch(vae, valid_dataset, device, norm=norm, is_sample=is_sample)
            print('(gaussian dist) mean sparsity without std norm:', hoyer_mean)
            
                                       
           

        print('Epoch {}; Valid Rec Loss {:.4f}; Valid KL-loss {:.4f}; Valid MMD-loss {:.4f}'.format(epoch, valid_rec_loss, valid_kl_loss, valid_reg_loss))
        

        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        

#### BASIC EVALUATION ####


def evaluate_nnl_and_kls_batch(data, vae, text_lang, is_random_evaluate=False):

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
    latent_dim = float(zs.shape[-1])
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
    
        

    return compute_sparsity(tf.convert_to_tensor(z_prev), norm=norm)





if __name__ == "__main__":
    print(tf.__version__)
    descr = "Tensorflow (Eager) implementation of MAT-VAE model. In all experiments Tensorflow (GPU) 2.3.0 and python 3.8.5 were used."
    epil  = ""
    parser = argparse.ArgumentParser(description=descr, epilog=epil)
    
    parser.add_argument('--alpha', required=True, type=float, default=1.0, help='')
    parser.add_argument('--beta', required=True, type=float, default=1.0, help='')
    parser.add_argument('--iter', required=True, type=int, default=1, help='')
    args = parser.parse_args()
 

    is_load = False
    training_data_path = '../Data/New_Yahoo/train.txt'
    valid_data_path='../Data/New_Yahoo/valid.txt'
    vocab_path = '../Data/New_Yahoo/vocab.txt'
    
 
    text_tensor, text_lang, mean_length_text = load_dataset(training_data_path, vocab_file=vocab_path)
    valid_text_tensor, _, _ = load_dataset(valid_data_path, mean_length_text=mean_length_text, text_lang=text_lang, is_test_data=True)
    epochs = 15
    buffer_size = len(text_tensor)
    batch_size =  256
    n_batch = math.ceil(buffer_size/batch_size)
    vocab_size = len(text_lang.word2idx)
    print('vocab size:', vocab_size)
    embedding_dim = 256
    encoder_dim = 512
    z_dim =  768
    alpha = args.alpha
    beta = args.beta
    iteration = args.iter 
    # Load Data #
    dataset = tf.data.Dataset.from_tensor_slices(text_tensor).shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    # Set an optimiser #
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0008)

    gamma = np.full((1,z_dim), 0.8) 

    print('gamma', gamma)
    loc = np.full((1,z_dim), 0.)
    scale = np.full((1,z_dim), 1.)

    gpu = '/gpu:0'
    sparse_pz = Sparse(gamma, loc, scale) 
    # Creating the Model #
    vae = Sentence_VAE(embedding_dim, vocab_size, encoder_dim, z_dim,  sparse_pz, alpha, beta)
    name_of_experiment ='WAE_KL_alpha_'+str(alpha)+'_beta_'+str(beta)+'_GRU_'+'_z_dim_'+str(z_dim)
    # Save Model #
    checkpoint_dir = '../Data/Trained_Models/'+ name_of_experiment+'_iter_'+str(iteration)

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, vae=vae)
    
    
    
    
    
    # Select between loading the loading an existing Model and training a new model #
    if is_load == True:
        load_path = tf.train.latest_checkpoint(checkpoint_dir)
        print('load path', load_path)
        load = checkpoint.restore(load_path)
    else:
        train(epochs, buffer_size ,vae, dataset, optimizer, text_lang, n_batch, checkpoint, valid_text_tensor, device=gpu)
    
    

    test_data_path='../Data/New_Yahoo/test.txt'

    
    test_data, _, _ = load_dataset(test_data_path, mean_length_text=mean_length_text, text_lang=text_lang, is_test_data=True)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
    test_batch_size =256# 512
    test_dataset = test_dataset.batch(test_batch_size, drop_remainder=False)
    

    with tf.device(gpu):
        test_rec_loss, test_kl_loss,  test_mmd_loss = evaluate_nnl_and_kls_batch(test_dataset, vae, text_lang)
    print('BATCH: Test Rec Loss {:.4f}; Test KL-loss {:.4f}; Test MMD-loss {:.4f}'.format(test_rec_loss, test_kl_loss,  test_mmd_loss))
    
    print(vae.summary()) 


