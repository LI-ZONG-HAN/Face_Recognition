import tensorflow as tf
import numpy as np
from Net.Loss import triplet_loss as triplet
from Net.Loss import face_losses as L_layer
#from tensorflow.contrib.slim import nets
from Net.Resnet_50 import resnet_v1
#from Net.Resnet_50 import resnet_v2
slim = tf.contrib.slim
import scipy.stats
import math
import os
import horovod.tensorflow as hvd

class Res50_Arc_loss(object):
    def __init__(self,hyper_para,ToTal_IDs, graph):
        super(Res50_Arc_loss, self).__init__()
        self.session = None
        hvd.init()
        self._config = tf.ConfigProto()
        self._config.gpu_options.allow_growth = True
        self._config.gpu_options.visible_device_list = str(hvd.local_rank())
        self._channels = 3
        self._share_layers_training = hyper_para["is_share_layers_training"]
        self._weight_decay_share = hyper_para["weight_decay_share"] #0.0015
        self._graph = graph
        self._crop_height = hyper_para["imgage_width"]
        self._crop_width = hyper_para["imgage_width"]
        self._FR_Emb_Dim = hyper_para["FR_Emb_Dim"]
        self._Gender_hinden_output = hyper_para["Gender_hinden_layers_dims"]
        self._Gender_class = 2
        self._Age_hinden_output = hyper_para["Age_hinden_layers_dims"]
        self._Age_Class = 7
        self._task = ["FR","Gender","Age"]
        self._weight_decay = {self._task[0]:hyper_para["weight_decay"][0], self._task[1]:hyper_para["weight_decay"][1], self._task[2]:hyper_para["weight_decay"][2]}
        self._l2_norm = True
        self._initial_lr_FR = hyper_para["initial_lr"][0] * hvd.size()#initial learning rate
        self._initial_lr_Gender = hyper_para["initial_lr"][1] * hvd.size()#initial learning rate
        self._initial_lr_Age = hyper_para["initial_lr"][2] * hvd.size()#initial learning rate
        self._lr_shrink_rate = 0.1
        self._ToTal_IDs = ToTal_IDs
        self._step_without_progress_thresh = hyper_para["step_without_progress_thresh"]
        self._loss_slop_check_budget = {}
        self._privious_loss_dump_amount = self._step_without_progress_thresh / 6
        self._loss_container = {}
        self._save_loss = {}
        self._steps_without_progress = {}
        for key in self._task:
            self._loss_container[key] = []
            self._save_loss[key] = []
            self._steps_without_progress[key] = 0
            self._loss_slop_check_budget[key] = 0
        self._soft_loss = {}
        self._l2_loss = {}
        self._total_loss = {}
        self._softmax_op = {}

        with self._graph.as_default():
            self._is_training = tf.placeholder(name="is_training",shape = (), dtype=tf.bool)
            self._imgs = tf.placeholder(name="input_imgs", shape=[None,self._crop_height,self._crop_width,self._channels], dtype=tf.float32)
            self._labels = {}
            self._labels[self._task[0]] = tf.placeholder(name="input_labels", shape=[None], dtype=tf.int32)
            self._labels[self._task[1]] = tf.placeholder(name="input_labels", shape=[None,self._Gender_class], dtype=tf.float32)
            self._labels[self._task[2]] = tf.placeholder(name="input_labels", shape=[None,self._Age_Class], dtype=tf.float32)
            self._lr_rate = {}
            with tf.variable_scope("lr_FR", reuse=tf.AUTO_REUSE):
                self._lr_rate[self._task[0]] = tf.get_variable("learning_rate",shape = (), dtype=tf.float32
				                                     , initializer = tf.constant_initializer(self._initial_lr_FR)) 
				
            with tf.variable_scope("lr_Gender", reuse=tf.AUTO_REUSE):
                self._lr_rate[self._task[1]] = tf.get_variable("learning_rate",shape = (), dtype=tf.float32
				                                      , initializer = tf.constant_initializer(self._initial_lr_Gender)) 
				
            with tf.variable_scope("lr_Age", reuse=tf.AUTO_REUSE):
                self._lr_rate[self._task[2]] = tf.get_variable("learning_rate",shape = (), dtype=tf.float32
				                                      , initializer = tf.constant_initializer(self._initial_lr_Age)) 
			
            with tf.variable_scope("step_count", reuse=tf.AUTO_REUSE):
                self._step_cnt_op = tf.get_variable("step_cnt", shape=(), dtype=tf.int64, initializer=tf.constant_initializer(0))
                self._incr_step_cnt_op = tf.assign_add(self._step_cnt_op, 1)
				
            
            self._images = self.Image_Preprocess(self._imgs, self._is_training)
			
            if self._share_layers_training:
                self._share_output = self._model_fn_share(self._images, self._weight_decay_share, self._is_training)
                self._share_l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                self._update_ops_share = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            else:
                self._weight_decay_share = 0.0
                self._share_output = self._model_fn_share(self._images, self._weight_decay_share, False)
                self._share_l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                self._share_output = tf.stop_gradient(self._share_output)
                self._share_l2_loss = tf.stop_gradient(self._share_l2_loss)
                self._update_ops_share = []
			
            self.All_l2_ten = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self._update_ops_ALL = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            #FR
            self._FR_embedding  = self._model_fn_FR    (self._share_output, self._weight_decay[self._task[0]], self._l2_norm, self._is_training)
            #self._FR_embedding = self._model_fn_FR    (self._share_output, 0.0, self._l2_norm, False)
            #self._FR_embedding = tf.stop_gradient(self._FR_embedding)
            w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
            self._FR_logit = L_layer.arcface_loss(self._FR_embedding, self._labels[self._task[0]], self._ToTal_IDs, w_init_method)
            self._softmax_op[self._task[0]] = None
            self._soft_loss[self._task[0]]  = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._FR_logit, labels=self._labels[self._task[0]]))
            self._FR_l2_ten = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self._FR_l2_ten = [l2 for l2 in self._FR_l2_ten if l2 not in self.All_l2_ten]
            self._l2_loss[self._task[0]]    = self._share_l2_loss + tf.reduce_sum(self._FR_l2_ten)
            self._total_loss[self._task[0]] = self._soft_loss[self._task[0]] + self._l2_loss[self._task[0]]
            self.All_l2_ten = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self._update_ops_FR = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self._update_ops_FR = [op for op in self._update_ops_FR if op not in self._update_ops_ALL]
            self._update_ops_ALL = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            

            #Gender
            self._Gender_logit = self._model_fn_Gender(self._share_output, self._weight_decay[self._task[1]], self._is_training)
            self._softmax_op[self._task[1]] = tf.nn.softmax(self._Gender_logit)
            self._soft_loss[self._task[1]]= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._Gender_logit, labels=self._labels[self._task[1]]))
            self._Gender_l2_ten = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self._Gender_l2_ten = [l2 for l2 in self._Gender_l2_ten if l2 not in self.All_l2_ten]
            self._l2_loss[self._task[1]] = self._share_l2_loss + tf.reduce_sum(self._Gender_l2_ten)
            self._total_loss[self._task[1]] = self._soft_loss[self._task[1]] + self._l2_loss[self._task[1]]
            self.All_l2_ten = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self._update_ops_Gender = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self._update_ops_Gender = [op for op in self._update_ops_Gender if op not in self._update_ops_ALL]
            self._update_ops_ALL = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            #Age
            self._Age_logit = self._model_fn_Age   (self._share_output, self._weight_decay[self._task[2]], self._is_training)
            self._softmax_op[self._task[2]] = tf.nn.softmax(self._Age_logit)
            self._soft_loss[self._task[2]]  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._Age_logit, labels=self._labels[self._task[2]]))
            self._Age_l2_ten = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self._Age_l2_ten = [l2 for l2 in self._Age_l2_ten if l2 not in self.All_l2_ten]
            self._l2_loss[self._task[2]] = self._share_l2_loss + tf.reduce_sum(self._Age_l2_ten)
            self._total_loss[self._task[2]] = self._soft_loss[self._task[2]] + self._l2_loss[self._task[2]]
            self._update_ops_Age = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self._update_ops_Age = [op for op in self._update_ops_Age if op not in self._update_ops_ALL]			

                       
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self._variables_to_restore = tf.global_variables()
            training_vars = tf.trainable_variables()
            #self._small_lr = tf.multiply(self._learning_rate, 1)
            self._optimize_op = {}
            with tf.control_dependencies(self._update_ops_share + self._update_ops_FR):
                op_FR = tf.train.MomentumOptimizer( learning_rate = self._lr_rate[self._task[0]], momentum=0.9 )
                op_FR = hvd.DistributedOptimizer(op_FR)
                self._optimize_op[self._task[0]] = op_FR.minimize(self._total_loss[self._task[0]])
            with tf.control_dependencies(self._update_ops_share + self._update_ops_Gender):
                op_Gender = tf.train.MomentumOptimizer( learning_rate = self._lr_rate[self._task[1]], momentum=0.9 )
                op_Gender = hvd.DistributedOptimizer(op_Gender)
                self._optimize_op[self._task[1]] = op_Gender.minimize(self._total_loss[self._task[1]])
            with tf.control_dependencies(self._update_ops_share + self._update_ops_Age):
                op_Age = tf.train.MomentumOptimizer( learning_rate = self._lr_rate[self._task[2]], momentum=0.9 )
                op_Age = hvd.DistributedOptimizer(op_Age)
                self._optimize_op[self._task[2]] = op_Age.minimize(self._total_loss[self._task[2]])
												  

            self._eval_ops = {}
            self._small_batch_output = {}
            self._small_batch_intput = {}
            #FR eval
            self._small_batch_intput[self._task[0]] = self._FR_embedding
            self._small_batch_output[self._task[0]] = tf.placeholder(name="emb_input", shape=[None,self._FR_Emb_Dim], dtype=tf.float32)
            self._eval_ops[self._task[0]] = {}
            eval_FR_logit = L_layer.arcface_loss(self._small_batch_output[self._task[0]], self._labels[self._task[0]], self._ToTal_IDs)
            self._eval_FR_arc_loss= tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=eval_FR_logit, labels=self._labels[self._task[0]]))
            self._eval_ops[self._task[0]]["loss"] = self._eval_FR_arc_loss + self._l2_loss[self._task[0]]
            pairwise_dist = triplet._pairwise_distances(self._small_batch_output[self._task[0]], False)
            self.Same_eval_matrix = self.Same_person_eval(self._labels[self._task[0]],pairwise_dist)
            self.Diff_eval_matrix = self.Diff_person_eval(self._labels[self._task[0]],pairwise_dist)
            self._eval_ops[self._task[0]]["Same_eval_matrix"] = self.Same_eval_matrix
            self._eval_ops[self._task[0]]["Diff_eval_matrix"] = self.Diff_eval_matrix
            #Gender eval
            self._small_batch_intput[self._task[1]] = self._Gender_logit
            self._small_batch_output[self._task[1]] = tf.placeholder(name="Gender_logit", shape=[None,self._Gender_class], dtype=tf.float32)
            self._eval_ops[self._task[1]] = {}
            self._eval_ops[self._task[1]]["softmax"] = tf.nn.softmax(self._small_batch_output[self._task[1]])
            self._eval_ops[self._task[1]]["loss"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			                                        logits=self._small_batch_output[self._task[1]], labels=self._labels[self._task[1]])) + self._l2_loss[self._task[1]]
            #Age eval
            self._small_batch_intput[self._task[2]] = self._Age_logit
            self._small_batch_output[self._task[2]] = tf.placeholder(name="Gender_logit", shape=[None,self._Age_Class], dtype=tf.float32)
            self._eval_ops[self._task[2]] = {}
            self._eval_ops[self._task[2]]["softmax"] = tf.nn.softmax(self._small_batch_output[self._task[2]])
            self._eval_ops[self._task[2]]["loss"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			                                        logits=self._small_batch_output[self._task[2]], labels=self._labels[self._task[2]])) + self._l2_loss[self._task[2]]
            

    def data_augmentation (self,input_tensors):
        with tf.name_scope('data_augmentation'):    
            distorted_image = tf.cast(input_tensors, tf.float32)
            distorted_image = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), distorted_image)
            #distorted_image = tf.image.random_brightness(distorted_image,max_delta=35)
            #distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)
            #distorted_image = tf.maximum(distorted_image, 0.0)
            #distorted_image = tf.minimum(distorted_image, 255.0)

        return (distorted_image - 255.0/2) / (255.0/2)

    def Preprocess_test (self,input_tensors):
        return (input_tensors - 255.0/2) / (255.0/2)

    def Image_Preprocess (self,input_tensors,Train):
        images = tf.cond(Train,
                         lambda: self.data_augmentation (input_tensors), 
                         lambda: self.Preprocess_test (input_tensors))
        
        return images
        #return self.Preprocess_test (input_tensors)
		
    def _model_fn_share(self,input_batch,weight_decay,is_training = False,num_classes = None):
        #with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay = weight_decay)):
        #    net, endpoints = resnet_v2.resnet_v2_50(input_batch, num_classes=None
        #                                                 ,is_training=is_training,reuse = tf.AUTO_REUSE)
														 
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay = weight_decay)):
            net, endpoints = resnet_v1.resnet_v1_50(input_batch, num_classes=None
                                                         ,is_training=is_training,reuse = tf.AUTO_REUSE)
														 
        net = tf.reduce_mean(net, axis=[1, 2])
        return net
    
    def _model_fn_FR(self,net,weight_decay,l2_normalized = True,is_training = False):
        net = tf.layers.dropout(net, rate = 0.4,training = is_training, name = "drop_out")
        with tf.variable_scope("fc_FR", reuse=tf.AUTO_REUSE) as scope:
            embedding = slim.fully_connected(net, num_outputs=self._FR_Emb_Dim
                                             ,weights_regularizer = slim.l2_regularizer(weight_decay)
                                             ,activation_fn=None, scope=scope)

        print("FR_output_net ",embedding.shape)
        if l2_normalized:
            embedding = tf.nn.l2_normalize(embedding, axis=1)    
        return embedding
		
    def _model_fn_Gender(self,net,weight_decay,is_training = False):
        net = tf.layers.dropout(net, rate = 0.5,training = is_training, name = "drop_out")
        count = 0
        for dim in self._Gender_hinden_output:
            #with tf.variable_scope("fc_Gender_"+str(count), reuse=tf.AUTO_REUSE) as scope:
            #    net = slim.fully_connected(net, num_outputs=dim
            #                              ,weights_regularizer = slim.l2_regularizer(weight_decay)
            #                              ,normalizer_fn=tf.layers.batch_normalization
			#							  ,normalizer_params={'training':is_training}
			#							  ,scope=scope)
			
            with tf.variable_scope("fc_Gender"+str(count), reuse=tf.AUTO_REUSE) as scope:
                net = slim.fully_connected(net, num_outputs=dim
                                          ,weights_regularizer = slim.l2_regularizer(weight_decay)
                                          ,normalizer_fn=None
                                          ,activation_fn=None
										  ,scope=scope)
										  
                net = tf.contrib.layers.batch_norm(net, is_training=is_training, activation_fn = tf.nn.relu, scope='postnorm')
            count += 1
            #print("gender_net ",net.shape)
        with tf.variable_scope("fc_Gender_"+str(count), reuse=tf.AUTO_REUSE) as scope:
            output = slim.fully_connected(net, num_outputs=self._Gender_class
                                         ,weights_regularizer = slim.l2_regularizer(weight_decay)
                                         ,activation_fn=None, scope=scope)
        #print("gender_output_net ",output.shape)
        return output
    
    def _model_fn_Age(self,net,weight_decay,is_training = False):
        net = tf.layers.dropout(net, rate = 0.5,training = is_training, name = "drop_out")
        count = 0
        for dim in self._Age_hinden_output:
            #with tf.variable_scope("fc_Age_"+str(count), reuse=tf.AUTO_REUSE) as scope:
            #    net = slim.fully_connected(net, num_outputs=dim
            #                              ,weights_regularizer = slim.l2_regularizer(weight_decay)
			#							  ,normalizer_fn=tf.layers.batch_normalization
			#							  ,normalizer_params={'training':is_training}
            #                              , scope=scope)
			
            with tf.variable_scope("fc_Age_"+str(count), reuse=tf.AUTO_REUSE) as scope:
                net = slim.fully_connected(net, num_outputs=dim
                                          ,weights_regularizer = slim.l2_regularizer(weight_decay)
										  ,normalizer_fn=None
                                          ,activation_fn=None
                                          , scope=scope)
                #net = tf.layers.batch_normalization(net, training=is_training, activation_fn = tf.nn.relu, name='postnorm')
                net = tf.contrib.layers.batch_norm(net, is_training=is_training, activation_fn = tf.nn.relu, scope='postnorm')

                count += 1
                #print("Age_net ",net.shape)
        with tf.variable_scope("fc_Age_"+str(count), reuse=tf.AUTO_REUSE) as scope:
            output = slim.fully_connected(net, num_outputs=self._Age_Class
                                         ,weights_regularizer = slim.l2_regularizer(weight_decay)
                                         ,activation_fn=None, scope=scope)
            #print("Age_output_net ",output.shape)
        return output

    def Same_person_eval(self, labels, pairwise_dist):
    
        mask_anchor_positive = triplet._get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)

		# We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
        return anchor_positive_dist

    def Diff_person_eval(self, labels, pairwise_dist):
		
        mask_anchor_negative = triplet._get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)

		# We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_negative_dist = tf.multiply(mask_anchor_negative, pairwise_dist)
        return anchor_negative_dist
    
    def _incr_step_cnt(self):
        self.session.run(self._incr_step_cnt_op)
		
    def _shrink_lr_rate(self,lr):
        self.session.run(tf.assign(lr, tf.multiply(lr, self._lr_shrink_rate)))
        
    def step_cnt(self):
        run_res = self.session.run([self._step_cnt_op])
        return run_res[0]
    
    def train_eval(self, input_batch,key):
        feed_dict = {self._imgs:input_batch["img"], self._is_training : True }
        small_batch_output = self.session.run(self._small_batch_intput[key],feed_dict = feed_dict)
        feed_dict = {self._small_batch_output[key]:small_batch_output, 
                     self._labels[key]: input_batch["label"]}	
        run_res = self.session.run(self._eval_ops[key],feed_dict = feed_dict)
        run_res["prefix"] = key+"_train"
        run_res["step_without_progress"] = self._steps_without_progress[key]
        if key == "FR":
            Same_mean, Same_std, Diff_mean, Diff_std, Gap = self.get_result(run_res)
            run_res["Same_mean"] = Same_mean
            run_res["Same_std"] = Same_std
            run_res["Diff_mean"] = Diff_mean
            run_res["Diff_std"] = Diff_std
            run_res["Gap"] = Gap
		
		
        return run_res
    
    def valid_eval(self,input_batch, key):
        small_batch_output = self.run_by_samll_batch(input_batch["img"],self._small_batch_intput[key], False)
        feed_dict = {self._small_batch_output[key]:small_batch_output, 
                     self._labels[key]: input_batch["label"]}	
        run_res = self.session.run(self._eval_ops[key],feed_dict = feed_dict)
        run_res["prefix"] = key+"_valid"
        
        if key == "FR":
            Same_mean, Same_std, Diff_mean, Diff_std, Gap = self.get_result(run_res)
            run_res["Same_mean"] = Same_mean
            run_res["Same_std"] = Same_std
            run_res["Diff_mean"] = Diff_mean
            run_res["Diff_std"] = Diff_std
            run_res["Gap"] = Gap
        
        return run_res
    
    def train_one_step(self,input_batch,key):
        self._incr_step_cnt()
        self.check_lr_rate(key)
        
        feed_dict = {
            self._is_training : True,
            self._imgs: input_batch["img"],
            self._labels[key]: input_batch["label"], 
        }
        run_ops = {
            "optimize": self._optimize_op[key],
            "_soft_loss": self._soft_loss[key],
            "l2_loss": self._l2_loss[key]
        }
        run_res = self.session.run(run_ops, feed_dict=feed_dict)
        loss = run_res["_soft_loss"] + run_res["l2_loss"]

        self._loss_container[key].append(loss)
        self._save_loss[key].append(loss)
        self._loss_slop_check_budget[key] += 20
		
    def check_lr_rate(self,key):
        if self._loss_slop_check_budget[key] >= self._step_without_progress_thresh and self._lr_shrink_rate != 1. :
            print (key+"_loss_check")
            if len(self._loss_container[key]) > self._step_without_progress_thresh:
                self._loss_container[key] = self._loss_container[key][ len(self._loss_container[key]) - self._step_without_progress_thresh :] #drop old loss

            self._loss_slop_check_budget[key] = 0
            self.Save_loss(self._save_loss[key], key)
            self._save_loss[key] = []

            
            self._steps_without_progress[key] = self.count_steps_without_decrease( np.array(self._loss_container[key]) )
            if  self._steps_without_progress[key] >=  self._step_without_progress_thresh:
                print (key+"_count_steps_without_decrease over")
                self._steps_without_progress[key] = self.count_steps_without_decrease_robust( np.array(self._loss_container[key]) )
                if  self._steps_without_progress[key] >=  self._step_without_progress_thresh:
                    self._shrink_lr_rate(self._lr_rate[key])
                    print (self._privious_loss_dump_amount+self._step_without_progress_thresh*0.2, len(self._loss_container[key]))
                    previous_drop_amount = min (self._privious_loss_dump_amount+self._step_without_progress_thresh*0.1, len(self._loss_container[key]))
                    print (previous_drop_amount)
                    self._loss_container[key] = self._loss_container[key][int(previous_drop_amount):]
                    print (len(self._loss_container[key]))
                    print (key+"_step:{:7d} lr_shrink, without_progress for{:7d} step".format(self.step_cnt(),self._steps_without_progress[key]))
                    self._steps_without_progress[key] = 0
		
    def run_by_samll_batch(self, imgs, output, is_training, interval = 10):
        total_num = imgs.shape[0]
        emb_np = np.zeros(shape=[total_num, output.shape[-1]], dtype=np.float32)
        if total_num > 0:
            cut_ind = np.arange(0,total_num,interval)
            if cut_ind[-1] != total_num:
                cut_ind = np.append(cut_ind,total_num)
            for i in range (cut_ind.shape[0]-1):
                start = cut_ind[i]
                end = cut_ind[i+1]
                feed_dict = {self._imgs:imgs[start:end], self._is_training : is_training }
                temp = self.session.run(output,feed_dict = feed_dict)
                emb_np[start:end] = temp
        else:
            print("eval_batch size equal to 0")
        return  emb_np
		
    
    def load_pre_trained_ckpt(self,ckpt_path):
        with self._graph.as_default():
            pre_trained_variables = [var for var in self._variables_to_restore
                                     if not (var.name.startswith('step') or var.name.startswith('fc') 
									 or var.name.startswith('lr') or var.name.find('embedding_weights') >= 0 )]
            
            saver_ckpt = tf.train.Saver(pre_trained_variables)
            saver_ckpt.restore(self.session,ckpt_path) 
    
    def freeze_deploy_graph(self,model_path):
        #exclude image_preprocess
        input_name = "input"
        output_name = "embedding,Gender,Age"
        with self._graph.as_default():
            img_input = tf.placeholder(name=input_name, shape=(None,self._crop_height,self._crop_width,self._channels)
                                       , dtype=tf.float32)

            share_output = self._model_fn_share(img_input, 0.1, False)
            output_FR = self._model_fn_FR(share_output, 0.1, self._l2_norm, False)
            output_Gender = tf.nn.softmax(self._model_fn_Gender(share_output, 0.1, False))
            output_Age = tf.nn.softmax(self._model_fn_Age(share_output, 0.1, False))

            output_FR = tf.identity(output_FR, name="embedding")
            output_Gender = tf.identity(output_Gender, name="Gender")
            output_Age = tf.identity(output_Age, name="Age")
			
			
        
        print (output_name.split(","))
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.session,
            self._graph.as_graph_def(),
            output_name.split(",")
        )
        with tf.gfile.GFile(model_path,"wb") as f:
            f.write(output_graph_def.SerializeToString())
        print ("%d ops in final graph." %len(output_graph_def.node))
		
    def count_steps_without_decrease(self, Y, probability_of_decrease = 0.51):
        steps_without_decrease = 0
        n = len(Y)
        if (n < 3):
            return steps_without_decrease
        for i in reversed(range(n-2)):
            #print (i)
            if self.Slope_P(Y[i:n],0) < probability_of_decrease:
                steps_without_decrease = n-i
        return steps_without_decrease

    def count_steps_without_increase(self, X, Y, probability_of_decrease = 0.51):
        steps_without_decrease = 0
        n = len(Y)
        last_step = X[-1]
        if (n < 3):
            return steps_without_decrease
        
        for i in reversed(range(n-2)):
            #print (i)
            if ( 1 - self.Slope_P(Y[i:n],0) ) < probability_of_decrease:
                steps_without_decrease = last_step - X[i]
        return steps_without_decrease

    def count_steps_without_decrease_robust(self, Y, probability_of_decrease = 0.51, quantile_discard = 0.1):
        steps_without_decrease = 0
        n = len(Y)
        if (n < 3):
            return steps_without_decrease
    
        loss_thresh = Y[Y.argsort()[int(Y.shape[0]*(1-quantile_discard))]]
        indexs = np.arange(n)
        indexs = indexs[Y < loss_thresh]
        Y = Y[indexs]  # discarding the top 10% of loss
        new_n = len(Y)
        #print (new_n , n)
        for i in reversed(range(new_n-2)):
            #print (i)
            if self.Slope_P(Y[i:new_n],0) < probability_of_decrease:
                steps_without_decrease = n-indexs[i]
        return steps_without_decrease
		
    def count_steps_without_increase_robust(self, X, Y, probability_of_decrease = 0.51, quantile_discard = 0.1):
        steps_without_decrease = 0
        n = len(Y)
        last_step = X[n-1]
        if (n < 3):
            return steps_without_decrease
    
        loss_thresh = Y[Y.argsort()[int(Y.shape[0]*quantile_discard)]]
        indexs = np.arange(n)
        indexs = indexs[Y > loss_thresh]
        Y = Y[indexs]  # discarding the top 10% of loss
        X = X[indexs]
        new_n = len(Y)
        #print (new_n , n)
        for i in reversed(range(new_n-2)):
            #print (i)
            if (1- self.Slope_P(Y[i:new_n],0)) < probability_of_decrease:
                steps_without_decrease = last_step-X[i]
        return steps_without_decrease
		
    def Slope_P(self, Y, smaller_than = 0):
        n = len(Y)
        x = np.arange(n)
        A = np.vstack([x, np.ones(n)]).T
        m_p,c_p = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(Y))
        Y_p = m_p*x+c_p
        sigma = 1/(n-2)*np.sum((Y-Y_p)**2)
        slope_sigma = 12*sigma/(n**3 -n)
        p1 = scipy.stats.norm.cdf(smaller_than, m_p, math.sqrt(slope_sigma))
        return p1

    def Save_loss(self, Y1 , scope):
        dir = "loss_debug"
        if not os.path.exists(dir):
            os.makedirs(dir)
        path1 = scope+"_loss.txt"

        with open(os.path.join(dir,path1),"a") as f:
            for loss in Y1:
                f.write(str(loss)+",")	

				
    def Save_valid_results(self, X, Y):
        dir = "loss_debug"
        if not os.path.exists(dir):
            os.makedirs(dir)
        path1 = "valid-result.txt"
        
        with open(os.path.join(dir,path1),"a") as f:
            f.write(str(X)+","+str(Y)+"\n")
        

    def get_result(self, res):
        Same = np.triu(res["Same_eval_matrix"],1).flatten()
        Diff = np.triu(res["Diff_eval_matrix"],1).flatten()
        Same = Same[Same>0]
        Diff = Diff[Diff>0]
        Total_Same_count = Same.shape[0]
        Same_mean = np.mean(Same)
        Same_std = np.std(Same,ddof=1)
        Total_Diff_count = Diff.shape[0]
        Diff_mean = np.mean(Diff)
        Diff_std = np.std(Diff,ddof=1)

        Gap = (Diff_mean-Same_mean)/Diff_std
        return Same_mean, Same_std, Diff_mean, Diff_std, Gap				
	 
		
		
