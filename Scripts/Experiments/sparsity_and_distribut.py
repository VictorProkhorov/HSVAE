


def store_alpha_beta(data, vae, alpha_param_file, beta_param_file, gpu ):
    with open(alpha_param_file, 'w') as alpha_param_f, open(beta_param_file, 'w') as beta_param_f:
        for x in data:
            #print('here')
            with tf.device(gpu):
                alpha_params, beta_params = vae.get_alpha_beta_params(x)
                alpha_params = alpha_params.numpy().tolist()
                beta_params = beta_params.numpy().tolist()
                for alpha_param, beta_param in zip(alpha_params, beta_params):
                    alpha_param = [round(elem, 2) for elem in alpha_param]
                    beta_param = [round(elem, 2) for elem in beta_param]
                  #  print('alpha;',alpha_param)
                  #  print('beta;',beta_param)
                    alpha_param_f.write(' '.join(map(str, alpha_param)) + '\n')
                    beta_param_f.write(' '.join(map(str, beta_param)) + '\n')

    return None

def store_alpha_beta_dif(data, vae, alpha_beta_dif_param_file,  gpu ):
    with open(alpha_beta_dif_param_file, 'w') as f:
        for x in data:
            #print('here')
            with tf.device(gpu):
                alpha_params, beta_params = vae.get_alpha_beta_params(x)
                difs = (alpha_params-beta_params).numpy().tolist()
                
                for dif in difs:
                    dif= [round(elem, 2) for elem in dif]
                
                    f.write('\t'.join(map(str, dif)) + '\n')

    return None

def store_beta_means(data, vae, beta_mean_file, gpu ):
    all_means = []
    with open(beta_mean_file, 'w') as f:
        for x in data:
            with tf.device(gpu):
                means = vae.get_mean_of_beta_distribution(x)
                means = means.numpy().tolist()
                for mean in means:
                    all_means.append(mean)
                    mean = [round(elem, 2) for elem in mean]
                    f.write('\t'.join(map(str, mean)) + '\n')
    
    means = np.array(all_means)
    print(means)
    print('mean', np.mean(means, axis=0))
    print('std', np.std(means, axis=0))
    return None

def binary_z_bar(z_bar, eps=1e-5):
    z_bar_abs = tf.math.abs(z_bar)
    b_z_bar = tf.where(z_bar_abs > eps, tf.fill(z_bar.shape,1.) , tf.fill(z_bar.shape,0.) )
    return b_z_bar
def get_percentile_threshold_value(vector, percentile = 1):
    n = vector.shape[-1].value
    vector_sorted = tf.sort(vector, axis=-1, direction='ASCENDING')
    percentile_idx = int(tf.math.ceil((percentile/100)*n))
    return vector_sorted[percentile_idx]
def binary_m(vector, threshold):
    return tf.where(vector > threshold, tf.fill(vector.shape,1.) , tf.fill(vector.shape,0.))


 
def compute_hoyer_ratio(dataset, vae, temperature, n_samples=5, threshold=None):
    zs = []
    #step 1 obtaining zs
    for batch_data in dataset:
       # print(batch_data) 
        with tf.device(gpu):
            z = vae.sample_k_zs(batch_data, temperature, n_samples=n_samples).numpy()
            if len(zs) > 0:
               # print('z', z.shape)
               # print('zs', zs.shape)
                zs = np.concatenate((zs, z), axis =1)
            else:
                zs = z
    zs = tf.convert_to_tensor(zs, dtype = tf.float32)
    print(zs.shape)
    # step 2: compute std(z) and normalise zs by the std. 
   # zs = zs / tf.math.reduce_std(zs, axis = 0)
    # step 3 marginalise over gamma
    m_zs = tf.math.reduce_mean(zs, axis=0)
    m_zs = m_zs / tf.math.reduce_std(m_zs, axis = 0)

    hoyer_numerator = compute_sparsity(m_zs, False)
    print('numerator', hoyer_numerator)
    binary = True
    if binary == True:
    #    print('min', tf.math.reduce_min(tf.math.abs(m_zs), axis=-1))
    #    print('m_zs', m_zs.shape)
        binary_m_zs = binary_z_bar(m_zs, eps = 1e-2)
    #    print('binary_mzs', binary_m_zs)
        b_bar = tf.math.reduce_mean(binary_m_zs, axis=0)
     #   print('b_bar', b_bar)
        if threshold == None:
            threshold = get_percentile_threshold_value(b_bar, percentile=5)
        print('threshold', threshold)
        b_m = binary_m(b_bar, threshold)
      #  print('binary m', b_m)
        b_m = tf.expand_dims(b_m, axis=0)
        m = tf.math.reduce_mean(m_zs, axis=0, keepdims=True)
       # print('m', m)
        m = m * b_m
    else:

        m = tf.math.reduce_mean(m_zs, axis=0, keepdims=True)
   # print('m', m)
    hoyer_denominator = compute_sparsity(m, False)

    print('den', hoyer_denominator)
    return hoyer_numerator/hoyer_denominator, threshold







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



