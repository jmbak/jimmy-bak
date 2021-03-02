import tensorflow as tf
from gen_meta_omni import Gen_data_Meta
import numpy as np
import os
from tensorflow.contrib.layers.python import layers as tf_layers
## Error not show!!
import warnings
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
data_folder='data/omniglot/data/'
num_in=5 # K of K-shot N-Way
num_in_and_out=50 # total number of data in D_{test}
dim_hidden=[64,64,64,64]
num_batch=4 # meta-batch size
num_fake_img = 3  # number of fake image (M)

meta_lr=1e-3 # meta-traing learning rate (gamma)
update_lr=1e-1 # learning rate for inner gradient update (alpha)

# 전체 Data에서 k=1000개는 training, 그 외는 test dataset으로 분할
gen_class=Gen_data_Meta(data_folder=data_folder,num_in=num_in,mode='train',
                            num_in_and_out=num_in_and_out,k=1000)
# # 전체 1623개 중 1000 + 623으로 분할됨
# print(len(gen_class.character_folders))
# print(len(gen_class.character_folders_train))
# print(len(gen_class.character_folders_test))

X_in=tf.placeholder('float',[num_batch,num_in, 28,28,1])
X_in_and_out=tf.placeholder('float',[num_batch,num_in_and_out, 28,28,1])
Y_in_and_out=tf.placeholder('float',[num_batch,num_in_and_out])

def loss_softmax(input, par, input_fake=None, reuse=True, label=None):
    repr=input
    for i in range(len(dim_hidden)):
        stride,no_stride=[1,2,2,1],[1,1,1,1]
        conv_output=tf.nn.conv2d(repr,par['w'+str(i+1)],strides=stride,padding='SAME')+par['b'+str(i+1)]
        repr = tf_layers.batch_norm(conv_output,activation_fn=tf.nn.elu,reuse=reuse,scope=str(i+1))
    input_fc = tf.layers.flatten(repr)
    input_loss = tf.matmul(input_fc, par['w_fc1']) + par['b_fc1']

    if label == None:
        input_shape = input.shape.as_list()
        input_fake_shape = input_fake.shape.as_list()

        repr_fake=tf.sigmoid(input_fake)
        for i in range(len(dim_hidden)):
            stride,no_stride=[1,2,2,1],[1,1,1,1]
            conv_output=tf.nn.conv2d(repr_fake,par['w'+str(i+1)],strides=stride,padding='SAME')+par['b'+str(i+1)]
            repr_fake = tf_layers.batch_norm(conv_output,activation_fn=tf.nn.elu,reuse=reuse,scope=str(i+1))
        input_fc = tf.layers.flatten(repr_fake)
        fake_loss = tf.matmul(input_fc, par['w_fc1']) + par['b_fc1']
        
        label_fake = tf.concat((tf.ones([input_fake_shape[0], 1]), tf.zeros([input_fake_shape[0], 1])), axis=1)
        label_real = tf.concat((tf.zeros([input_shape[0], 1]), tf.ones([input_shape[0], 1])), axis=1)

        result_img = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_real, logits=input_loss))
        result_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_fake, logits=fake_loss))
        result = result_img + result_fake
    else:

        label_re = tf.reshape(label, [-1, 1])
        label_real = tf.concat((label_re, 1.0 - label_re), axis=1)
        result = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_real, logits=input_loss))

    return result

def loss_adv(input_fake, par):
    repr_fake=tf.sigmoid(input_fake)
    for i in range(len(dim_hidden)):
        stride,no_stride=[1,2,2,1],[1,1,1,1]
        conv_output=tf.nn.conv2d(repr_fake,par['w'+str(i+1)],strides=stride,padding='SAME')+par['b'+str(i+1)]
        repr_fake = tf_layers.batch_norm(conv_output,activation_fn=tf.nn.elu,reuse=True,scope=str(i+1))
    input_fc = tf.layers.flatten(repr_fake)
    input_fake = tf.matmul(input_fc, par['w_fc1']) + par['b_fc1']
    logit_fake = tf.concat((tf.zeros([tf.shape(input_fake)[0], 1]), tf.ones([tf.shape(input_fake)[0], 1])), axis=1)

    result_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=logit_fake, logits=input_fake))

    return result_fake


def Task_learn(inp):
    X_i, X_target2_i, Y_target2_i = inp
    num_iter=1
    loss_real = loss_softmax(input=X_i, input_fake=fake_img, par=par_real1, reuse=True) # L_{\theta}(D_{train}, \theta_{fake})
    grad_real = tf.gradients(loss_real, list(par_real1.values())) # \partial(L) / \partial(\theta)
    gradient_real = dict(zip(par_real1.keys(), grad_real)) # dict(zip([a1,a2,..],[b1,b2,..]))는 'a1':b1, 'a2':b2..로 dictionary를 연결해준다. 
    # Eq. (1)
    par_real_update1 = dict(zip(par_real1.keys(), [par_real1[key] - update_lr * gradient_real[key] for key in par_real1.keys()]))

    for k in range(num_iter-1):
        loss_real = loss_softmax(input=X_i, input_fake=fake_img, par=par_real_update1, reuse=True) # same with above
        grad_real = tf.gradients(loss_real, list(par_real_update1.values())) # same with above
        gradient_real = dict(zip(par_real_update1.keys(), grad_real)) # same with above
        # Eq. (1)
        par_real_update1 = dict(zip(par_real_update1.keys(), [par_real_update1[key] - update_lr * gradient_real[key] for key in par_real_update1.keys()]))
    
    loss_real_adv = loss_adv(input_fake=fake_img,  par=par_real_update1) # L_{\theta}(\theta_{fake})
    grad_real_adv = tf.gradients(loss_real_adv, [fake_img]) # \partial(L) / \partial(\theta_{fake})
    grad_real_adv = tf.stop_gradient(grad_real_adv) 
    # Eq. (2)
    input_fake_update = fake_img - tf.multiply(tf.nn.softplus(beta_fake) , tf.sign(grad_real_adv[0])) # \theta_{fake} - \beta_{fake}\odot sign(grad)
    # softplus : log(exp(x) + 1)  => 음수가 나오지 않도록 넣은듯 => 음수일때는 0
    # multiply : point wise product

    for k in range(2):
        loss_real_adv = loss_adv(input_fake=input_fake_update, par=par_real_update1) # update된 input_fake 입력
        grad_real_adv = tf.gradients(loss_real_adv, [input_fake_update])
        grad_real_adv = tf.stop_gradient(grad_real_adv) #괄호 내의 파란미터는 학습되지 않는다.
        # Eq. (2)
        input_fake_update = input_fake_update - tf.multiply(tf.nn.softplus(beta_fake) , tf.sign(grad_real_adv[0]))

    loss_adv_final = loss_adv(input_fake=input_fake_update,par=par_real_update1)
    fake_stack = tf.concat((fake_img, input_fake_update), axis=0)
    
    loss_real = loss_softmax(input=X_i, input_fake=fake_stack, par=par_real1, reuse=True)
    grad_real = tf.gradients(loss_real, list(par_real1.values()))
    gradient_real = dict(zip(par_real1.keys(), grad_real))
    # Eq. (3)
    par_real_update = dict(zip(par_real1.keys(), [par_real1[key] - update_lr * gradient_real[key] for key in par_real1.keys()]))
    
    num_iter=3
    for k in range(num_iter-1):
        loss_real = loss_softmax(input=X_i, input_fake=fake_stack, par=par_real_update, reuse=True)
        grad_real = tf.gradients(loss_real, list(par_real_update.values()))
        gradient_real = dict(zip(par_real_update.keys(), grad_real))
        # Eq. (3)
        par_real_update = dict(zip(par_real_update.keys(),[par_real_update[key] - update_lr * gradient_real[key] for key in par_real_update.keys()]))
    
    # Eq. (4)
    loss_real_tar = loss_softmax(input=X_target2_i, par=par_real_update, label=Y_target2_i, reuse=True)
            #     def loss_softmax(input,             par,  input_fake=None, reuse=True, label=None):

    return loss_real_tar

# def construct_conv_weights():
initializer_w=tf.contrib.layers.xavier_initializer_conv2d()
initializer_b=tf.contrib.layers.xavier_initializer()
par_real1={}

with tf.variable_scope("par_real",reuse=tf.AUTO_REUSE):
    shape_w=[3,3,1,dim_hidden[0]]  # 필터 64개?
    shape_b=[dim_hidden[0]]        # 64개

    par_real1['w1'] = tf.Variable(initializer_w(shape=shape_w),name='w1')
    par_real1['b1'] = tf.Variable(initializer_b(shape=shape_b),name='b1')
    for i in range(1, len(dim_hidden)): # i = 1,2,3
        shape_w=[3,3,dim_hidden[i - 1], dim_hidden[i]]
        shape_b=[dim_hidden[i]]

        par_real1['w' + str(i + 1)] = tf.Variable(initializer_w(shape=shape_w), name='w' + str(i + 1))
        par_real1['b' + str(i + 1)] = tf.Variable(initializer_b(shape=shape_b), name='b' + str(i + 1))

with tf.variable_scope("par_fake", reuse=tf.AUTO_REUSE):
    initializer_fake = tf.random_normal_initializer()
    initializer_alpha = tf.ones_initializer()
    fake_img_shape = [num_fake_img, 28,28, 1]
    fake_img_par = tf.Variable(initializer_fake(shape=fake_img_shape), name='fake_img')
    beta_fake = tf.Variable(initializer_alpha(shape=fake_img_shape), name='alpha')

fake_img=fake_img_par
# Vaiable : par_real1 : [w1~w4, b1~b4]
# Vaiable : fake_img(shape([num_fake_img=3, 28, 28, 1]))
# Vaiable : beta_fake(shape([num_fake_img=3, 28, 28, 1])) = 1로 초기화
# par_real1, fake_img=fake_img_par, beta_fake  # theta, theta_{fake}, beta_fake

#######################################################################################

with tf.variable_scope('par_real/', reuse=tf.AUTO_REUSE):
    dim_w1 = [64*4, 2]
    dim_b1 = [2]
    initializer_fc=tf.contrib.layers.xavier_initializer()
    # 파라미터 초기화
    par_real1['w_fc1']=tf.Variable(name='w_fc1',initial_value=initializer_fc(shape=dim_w1))
    par_real1['b_fc1']=tf.Variable(name='b_fc1',initial_value=initializer_fc(shape=dim_b1))

# Vaiable : par_real1 : [w1~w4, b1~b4, w_fc1, b_fc1]

unused=X_in[0]
for i in range(len(dim_hidden)):
    stride,no_stride=[1,2,2,1],[1,1,1,1]
    conv_output=tf.nn.conv2d(unused,par_real1['w'+str(i+1)],strides=stride,padding='SAME')+par_real1['b'+str(i+1)]
    unused = tf_layers.batch_norm(conv_output,activation_fn=tf.nn.elu,reuse=False,scope=str(i+1))
loss_class=tf.map_fn(Task_learn, 
                   elems=(X_in, X_in_and_out, Y_in_and_out), 
                   dtype=tf.float32, 
                   parallel_iterations=num_batch)
loss_class_final=tf.reduce_mean(loss_class)
#######################################################################################

Train_op0=tf.train.AdamOptimizer(learning_rate=meta_lr).minimize(loss_class_final)

#######################################################################################



sess = tf.Session()
# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

max_iteration = 10
for iter in range(max_iteration):
    for j in range(num_batch):
        X_in_j, X_in_and_out_j, Y_in_and_out_j=gen_class.construction_unknown_single_train()
        X_in_j = np.expand_dims(np.expand_dims(X_in_j, 0),-1)
        X_in_and_out_j = np.expand_dims(np.expand_dims(X_in_and_out_j, 0),-1)
        if j == 0:
            X_in_feed = X_in_j
            X_in_and_out_feed = X_in_and_out_j
            Y_in_and_out_feed = Y_in_and_out_j
        else:
            # np.vstack 세로 결합
            X_in_feed = np.vstack((X_in_feed, X_in_j))
            X_in_and_out_feed = np.vstack((X_in_and_out_feed, X_in_and_out_j))
            Y_in_and_out_feed = np.vstack((Y_in_and_out_feed, Y_in_and_out_j))
    loss_r1,_= sess.run((loss_class_final,Train_op0),
                        feed_dict={X_in        : X_in_feed, 
                                   X_in_and_out: X_in_and_out_feed,
                                   Y_in_and_out: Y_in_and_out_feed})
    if iter %10 ==0:
        print(loss_r1,iter)

