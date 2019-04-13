from PIL import Image
import os
import numpy as np
import tensorflow as tf
import time
import json
import threading
from queue import Queue
from Net import Net_Arcface_multi_task as ArcFace
from tensorflow.contrib.slim import nets
slim = tf.contrib.slim
#import dlib

import io
import random


def _log_trainable_variables():
    total_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()

    var_params = 1
    for dim in shape:
        var_params *= dim.value
    tf.logging.info("%s %s: %s", variable.name, shape, var_params)

    total_params += var_params

    tf.logging.info("total parameters: %s", total_params)
	
def _randomize(lists,seed):
    permutation = np.random.RandomState(seed=seed).permutation(lists.shape[0])
    shuffled_lists = lists[permutation]
    return shuffled_lists

def load_path_lists(data_dir):
    lists = os.listdir(data_dir)
    lists = [os.path.join(data_dir,f) for f in lists]
    lists = [f for f in lists if os.path.isdir(f)]
    results = []
    for f in lists:
        temp_array = _randomize(np.array([os.path.join(f,path) for path in os.listdir(f)]),int(time.time()))
        results.append(temp_array)
    return results

def load_path_lm_lists(data_dir,flie_name):
    with open(os.path.join(data_dir,flie_name),"r") as f:
        lines = f.readlines()
        lines = [f.strip() for f in lines]   
        path_list = np.array([data_dir+"/"+f.strip().split(" ",1)[0] for f in lines])
        lm_list = np.array([f.strip().split(" ",1)[1].split(" ")[1:] for f in lines],dtype = np.float32)
        label_list = [f.split("/")[2] for f in path_list]
        hush_table = {}
        path_list_by_id = []
        lm_list_by_id =[]
        keys_list_temp=[]
        for i,l in enumerate(label_list):
            if l not in hush_table.keys():
                hush_table[l] = [i]
                keys_list_temp.append(l)
            else:
                hush_table[l] += [i]
        #for k,v in hush_table.items():
        #    path_list_by_id.append(path_list[v])
        #    lm_list_by_id.append(lm_list[v])
        for k in keys_list_temp:
            path_list_by_id.append(path_list[hush_table[k]])
            lm_list_by_id.append(lm_list[hush_table[k]])
        
        labels = []
        for i, ls in enumerate(path_list_by_id):
            labels += [i for num in range(ls.shape[0])]
        labels = np.array(labels, dtype = np.int32)
        del label_list
        del hush_table
    return path_list_by_id, lm_list_by_id, path_list, lm_list, labels
def load_index_lists(data_dir,flie_name):
    with open(os.path.join(data_dir,flie_name),"r") as f:
        lines = f.readlines()
        lines = [f.strip() for f in lines]   
        path_list = np.array([data_dir+"/"+f.strip().split(" ",1)[0] for f in lines])
        index_list = np.array([f.strip().split(" ")[1:] for f in lines],dtype = np.uint64)
        label_list = [f.split("/")[2] for f in path_list]
        hush_table = {}
        path_list_by_id = []
        index_list_by_id =[]
        keys_list_temp=[]
        for i,l in enumerate(label_list):
            if l not in hush_table.keys():
                hush_table[l] = [i]
                keys_list_temp.append(l)
            else:
                hush_table[l] += [i]

        for k in keys_list_temp:
            path_list_by_id.append(path_list[hush_table[k]])
            index_list_by_id.append(index_list[hush_table[k]])
        
        labels = []
        for i, ls in enumerate(path_list_by_id):
            labels += [i for num in range(ls.shape[0])]
        labels = np.array(labels, dtype = np.int32)
        del label_list
        del hush_table
    return path_list_by_id, index_list_by_id, path_list, index_list, labels


class FR_Data_Thread_by_Class(threading.Thread):
    def __init__(self, threadID, seed,index_lists,f_bin_paths, num_class, img_no_per_class, img_height,img_width,task,q):
        threading.Thread.__init__(self)
        super(FR_Data_Thread_by_Class, self).__init__()
        self.threadID = threadID
        self.seed = int(seed)
        self.queue = q
        self._num_class = num_class #number of classes per batch
        self._img_no_per_class = img_no_per_class
        self._batch_size = num_class*img_no_per_class
        self._img_height = img_height
        self._img_width = img_width
        self._channels = 3
        self._thread_stop = False 
        self._index_lists = []
        self._bin_idx = []
        for i, list in enumerate(index_lists):
            self._index_lists += list
            self._bin_idx += [i for a in range(len(list))]

        self._total_classes = len(self._index_lists)
        self._padding = 0.25
        self._task = task
        self._f_bin_paths = f_bin_paths
        self._f_bins = []
       
    
    def _randomize(self, lists, seed):
        permutation = np.random.RandomState(seed=seed).permutation(lists.shape[0])
        shuffled_lists = lists[permutation]
        return shuffled_lists
      
    def get_data(self):
        if len(self._f_bins) == 0:
            return None
		
        res = {"img": np.ndarray(shape=(self._batch_size, self._img_height, self._img_width, self._channels), dtype=np.float32)}
        if self._task == "FR":
            res["label"] =  np.ndarray(shape=(self._batch_size), dtype=np.int32)
        else:
            res["label"] = np.ndarray(shape=(self._batch_size, self._total_classes), dtype=np.float32)

        count = 0
        labels = np.arange(self._total_classes)
        np.random.shuffle(labels)
        labels = labels[:min(self._num_class*2,self._total_classes)]
        ii = 0

        for label in labels:
            if (ii >= self._num_class):
                break
            if self._img_no_per_class < self._index_lists[label].shape[0]:
                indexs = np.arange(self._index_lists[label].shape[0])
                np.random.shuffle(indexs)
            else:
                indexs = np.random.randint(self._index_lists[label].shape[0], size=self._img_no_per_class)

            no_imgs = 0
            
            
            for index in indexs: 
                if (no_imgs >= self._img_no_per_class):
                    break
                read_start = self._index_lists[label][index][0]
                read_end = self._index_lists[label][index][1]
                self._f_bins[self._bin_idx[label]].seek(read_start)
                data = self._f_bins[self._bin_idx[label]].read(read_end - read_start)
                img = Image.open(io.BytesIO(data))
                h,w = img.size
                if (h != self._img_height or w != self._img_width):
                    crop_img = img.resize((self._img_height, self._img_width), Image.ANTIALIAS)
                else:
                    crop_img = img
   
                res["img"][count,:,:,:] = np.array(crop_img)
                if self._task == "FR":
                    res["label"][count] = label
                else: #soft one-hot
                    res["label"][count,:] = (1-0.9)/(self._total_classes-1)
                    res["label"][count,label] = 0.9
				
                no_imgs += 1
                count += 1
 
            ii += 1
        if (count < self._batch_size):
            res["img"] = res["img"][0:count]
            res["label"] = res["label"][0:count]

        return res
       
    
    def Crop_1_face_wiht_lm (self,img, lm , size = 224 , padding = 0.25):
        h,w = img.size
        eye_dist = lm[2] - lm[0]
        extend = 1.5
        left = int(max(lm[0] - eye_dist*extend+0.5 , 0))
        top = int(max(lm[1] - eye_dist*extend+0.5 , 0))
        rihgt = int(min(lm[2] + eye_dist*extend+0.5,w))
        bottom = int(min(lm[3]+ eye_dist + eye_dist*extend +0.5,h))
        dlib_rect = dlib.rectangle(left,top,rihgt,bottom)
        img = np.array(img)
        faces = dlib.full_object_detections()
        
        faces.append(self._sp(img, dlib_rect))
        image = dlib.get_face_chip(img, faces[0], size, padding)
        return image,1
    
    def Crop_1_face_wiht_FD_rect (self,img, rect , size = 224 , padding = 0.25):
        left   = rect[0]
        top    = rect[1]
        rihgt  = rect[2]
        bottom = rect[3]
        dlib_rect = dlib.rectangle(left,top,rihgt,bottom)
        img = np.array(img)
        faces = dlib.full_object_detections()
        faces.append(self._sp(img, dlib_rect))
        image = dlib.get_face_chip(img, faces[0], size, padding)
        return image,1
    
    
    def FD_Crop_1_face (self,img , size = 224 , padding = 0.25):
        img = np.array(img)
        dets = self._detector(img)
        num_face = len(dets)
        index = 0
        if num_face == 0:
            #print ("no_face")
            return None , num_face
        elif num_face > 1:
            distance = 100000000;
            img_center_x = img.shape[0] * 0.5;
            img_center_y = img.shape[1] * 0.5;
            for i,det in enumerate(dets):
                center_x = ( det.left()   + det.right() ) * 0.5;
                center_y = ( det.bottom() + det.top()   ) * 0.5;

                temp_dis = (img_center_x - center_x)**2 + (img_center_y - center_y)**2
                if (temp_dis < distance):
                    distance = temp_dis
                    index = i
        faces = dlib.full_object_detections()
        faces.append(self._sp(img, dets[index]))
        image = dlib.get_face_chip(img, faces[0], size, padding)
        return image, num_face
                
    def run(self):
        self.open_bin()
        while not self._thread_stop:
            datas = self.get_data()
            t1 = time.time()
            try:
                self.queue.put(datas,True,100)
            except:
                print ("get time_out Thread_ID = %d" % self.threadID)

        self.close_bin()
        print ("hread_ID = %d run end" % self.threadID)
    def open_bin(self):
        for path in self._f_bin_paths:
            self._f_bins.append(open(path,"rb"))
    def close_bin(self):
        for f_bin in self._f_bins:
            f_bin.close()
        self._f_bins = []


	
	
	
class Logger(object):

    def __init__(self, wid, log_file, period_secs):
        super(Logger, self).__init__()
        self._wid = wid
        self._log_file = log_file
        self._period_secs = period_secs

        self._last_log_time = -1

        self._lock = threading.Lock()
        self._vals = {}

    def set_vals(self, vals):
        with self._lock:
            for k, v in vals.items():
                if v is not None:
                    self._vals[k] = v
                else:
                    self._vals.pop(k, None)

    def log(self, global_step):
        if not self._vals:
            return   #empty

        self._vals["time"] = "{:02d}-{:02d}-{:02d}".format(time.localtime()[3],time.localtime()[4],time.localtime()[5])
        self._vals["step"] = "{:4d}".format(global_step)
        
        with open(self._log_file, "a") as f:
            jstr = json.dumps(self._vals, sort_keys=True)
            f.write(jstr+"\n")
            self._vals.clear()

            
class Evaluator_FR(object):

    def __init__(self, net, logger, period_secs_train, period_secs_valid , save_path_train,save_path_valid, sess):
        super(Evaluator_FR, self).__init__()
        self._net = net
        self._logger = logger
        self._period_secs_train = period_secs_train
        self._period_secs_valid = period_secs_valid
        self._save_path_train = save_path_train
        self._save_path_valid = save_path_valid
        self._sess = sess
        self._last_eval_time_train = -1
        self._last_eval_time_valid = -1
        
        
        if self._save_path_train != "":
            self._train_max_Gap = -9999999999
            save_dir1 = os.path.dirname(self._save_path_train)
            if not os.path.exists(save_dir1):
                os.makedirs(save_dir1)
                
            self._saver_train = tf.train.Saver(max_to_keep=1)

        if self._save_path_valid != "":
            self._valid_max_Gap = -9999999999
            save_dir2 = os.path.dirname(self._save_path_valid)
            if not os.path.exists(save_dir2):
                os.makedirs(save_dir2)  
                
            self._saver_valid = tf.train.Saver(max_to_keep=1)        
            

    def train_eval(self,input_datas,key):
        if time.time() - self._last_eval_time_train < self._period_secs_train:
            return
        self._last_eval_time_train = time.time()

        test_res = self._net.train_eval(input_datas,key)
        #Same_mean, Same_std, Diff_mean, Diff_std, Gap = self.get_result(test_res)
        
        #print(type(test_res["loss"]))
        raw_log_vals = {"loss": "{0:.3f}".format(test_res["loss"].tolist())}
        raw_log_vals["Same_mean"] = "{:6.3f}".format(test_res["Same_mean"])
        raw_log_vals["Diff_mean"] = "{:6.3f}".format(test_res["Diff_mean"])
        raw_log_vals["Gap"] = "{:6.3f}".format(test_res["Gap"])
        raw_log_vals["step_without_progress"] = "{:7d}".format(test_res["step_without_progress"])
		
        global_step = global_step = self._net.step_cnt()
        l_rate = self._net._lr_rate[key].eval()
        raw_log_vals["l_rate"] = "{:e}".format(l_rate)
		
        print("step: {:7d}, {:<15s}, Cost: {:6.3e}, learning_rate: {:e}, Same_mean {:6.2f}, Diff_mean {:6.2f}, Gap{:6.2f}, step_without_progress{:7d}"
			.format(global_step, test_res["prefix"],test_res["loss"].tolist(),l_rate,test_res["Same_mean"], test_res["Diff_mean"], test_res["Gap"], test_res["step_without_progress"]))
        
        prefix = test_res["prefix"]
        log_vals = {}
        self._logger._vals.clear()
        for k, v in raw_log_vals.items():
            log_vals[prefix + k] = v
        self._logger.set_vals(log_vals)
        
        if self._save_path_train != "":
            if test_res["Gap"] > self._train_max_Gap:
                self._train_max_Gap = test_res["Gap"]
                self._saver_train.save(self._sess, self._save_path_train, global_step=global_step)
                
    def valid_eval(self,input_datas,key):
        if time.time() - self._last_eval_time_valid < self._period_secs_valid:
            return
        
        self._last_eval_time_valid = time.time()
        test_res = self._net.valid_eval(input_datas,key)
        #Same_mean, Same_std, Diff_mean, Diff_std, Gap = self.get_result(test_res)
		
        raw_log_vals = {}
            
       
        raw_log_vals["Same_mean"] = "{:6.3f}".format(test_res["Same_mean"])
        raw_log_vals["Same_std"] = "{:6.3f}".format(test_res["Same_std"])
        raw_log_vals["Diff_mean"] = "{:6.3f}".format(test_res["Diff_mean"])
        raw_log_vals["Diff_std"] = "{:6.3f}".format(test_res["Diff_std"])
        raw_log_vals["Gap"] = "{:6.3f}".format(test_res["Gap"])	
        
        global_step = self._net.step_cnt()
        print("step: {:7d}, {:<15s}, Same_mean {:6.2f}, Same_std:{:6.2f}, Diff_mean {:6.3f}, Diff_std:{:6.3f}, Gap{:6.3f}"
			.format(global_step, test_res["prefix"], test_res["Same_mean"], test_res["Same_std"], test_res["Diff_mean"], test_res["Diff_std"], test_res["Gap"]))
        prefix = test_res["prefix"]
        log_vals = {}
        for k, v in raw_log_vals.items():
            log_vals[prefix + k] = v
        self._logger.set_vals(log_vals)
        
        if self._save_path_valid != "":
            if test_res["Gap"] > self._valid_max_Gap:
                self._valid_max_Gap = test_res["Gap"]
                self._saver_valid.save(self._sess, self._save_path_valid, global_step=global_step)
    

            
class Evaluator_non_FR(object):

    def __init__(self, net, logger, period_secs_train, period_secs_valid , save_path_train,save_path_valid, sess):
        super(Evaluator_non_FR, self).__init__()
        self._net = net
        self._logger = logger
        self._period_secs_train = period_secs_train
        self._period_secs_valid = period_secs_valid
        self._save_path_train = save_path_train
        self._save_path_valid = save_path_valid
        self._sess = sess
        self._last_eval_time_train = -1
        self._last_eval_time_valid = -1

        
        
        if self._save_path_train != "":
            self._train_best_acc = -9999999999
            save_dir1 = os.path.dirname(self._save_path_train)
            if not os.path.exists(save_dir1):
                os.makedirs(save_dir1)
                
            self._saver_train = tf.train.Saver(max_to_keep=1)

        if self._save_path_valid != "":
            self._valid_best_acc = -9999999999
            save_dir2 = os.path.dirname(self._save_path_valid)
            if not os.path.exists(save_dir2):
                os.makedirs(save_dir2)  
                
            self._saver_valid = tf.train.Saver(max_to_keep=1)        
            

    def train_eval(self,input_datas,key):
        if time.time() - self._last_eval_time_train < self._period_secs_train:
            return
        self._last_eval_time_train = time.time()

        test_res = self._net.train_eval(input_datas,key)
        
        predict_M,_,_ = self.Confusion_M(test_res["softmax"],input_datas["label"])
        average_acc = np.average(predict_M.diagonal())
        
        raw_log_vals = {
            "acc_by_class": [ "{:6.2f}".format(predict_M[i,i]) for i in range (predict_M.shape[0]) ],
            "loss": "{0:.3f}".format(test_res["loss"].tolist()),
        }
        global_step = global_step = self._net.step_cnt()
        l_rate = self._net._lr_rate[key].eval()
        print("step: {:7d}, {:<15s}, Cost: {:6.3e}, Average_accuracy: {:6.2f}, learning_rate: {:e}, step_without_progress{:7d}"
              .format(global_step, test_res["prefix"],test_res["loss"].tolist(), average_acc,l_rate, test_res["step_without_progress"]))
        
        prefix = test_res["prefix"]
        log_vals = {}
        self._logger._vals.clear()
        for k, v in raw_log_vals.items():
            log_vals[prefix + k] = v
        self._logger.set_vals(log_vals)
        
        if self._save_path_train != "":
            if average_acc > self._train_best_acc:
                self._train_best_acc = average_acc
                self._saver_train.save(self._sess, self._save_path_train, global_step=global_step)
                
    def valid_eval(self,input_datas,key):
        if time.time() - self._last_eval_time_valid < self._period_secs_valid:
            return
        
        
        self._last_eval_time_valid = time.time()
        test_res = self._net.valid_eval(input_datas,key)
        
        predict_M,_,_ = self.Confusion_M(test_res["softmax"],input_datas["label"])
        average_acc = np.average(predict_M.diagonal())

        
        raw_log_vals = {
            "acc_by_class": [ "{:6.2f}".format(predict_M[i,i]) for i in range (predict_M.shape[0]) ],
        }
        global_step = self._net.step_cnt()
        
        print("step: {:7d}, {:<15s}, Average_accuracy: {:6.2f}".format(global_step, test_res["prefix"], average_acc))
        
        prefix = test_res["prefix"]
        log_vals = {}
        for k, v in raw_log_vals.items():
            log_vals[prefix + k] = v
        self._logger.set_vals(log_vals)
        
        if self._save_path_valid != "":
            if average_acc > self._valid_best_acc:
                self._valid_best_acc = average_acc
                self._saver_valid.save(self._sess, self._save_path_valid, global_step=global_step)
        
        
    def valid_all_eval(self,test_env):
        #for eval.py using
        test_env._i = 0
        total_size = test_env._labels.shape[0]
        batch_size = test_env._batch_size
        loop_num = int(total_size / batch_size)
        residue_num = int (total_size % batch_size)
        
        
        test_res = self._net.valid_eval_input(test_env.get())
        
        if (loop_num == 0):
            for k,v in test_res.items():
                test_res[k] = test_res[k][0:residue_num,:]
        
        for i in range(loop_num):
            res = self._net.valid_eval_input(test_env.get())
            for k,v in res.items():
                if (i == loop_num - 1):
                    test_res[k] = np.append(test_res[k],v[0:residue_num,:],axis = 0)
                else:
                    test_res[k] = np.append(test_res[k],v,axis = 0)
                    
        
        
        predict_M , count_M , mapping_img = self.Confusion_M(test_res["softmax"],test_res["label"])
        average_acc = np.average(predict_M.diagonal())
        
        count_by_class = np.sum(count_M,axis = 1)
        for i in range (predict_M.shape[0]):
            print ("class %d" %i, "{:6.2f}".format(predict_M[i,i]) , "No_of_img %d" % count_by_class[i])
        
        Top_N = 2
        for i in range(test_res["label"].shape[1]):
            index = np.argmax(test_res["label"], 1) == i
            print ("Top_%d class %d" %(Top_N,i)
                   , "{:6.2f}".format(self.Top_N_acc(test_res["softmax"][index,:], test_res["label"][index,:],Top_N))
                   , "No_of_img %d" % count_by_class[i])
        
        print("{},Average_accuracy: {:6.2f}".format(test_res["prefix"], average_acc))
        return mapping_img
        
                  
    def Confusion_M(self, preds, labels):
        preds_1D = np.argmax(preds, 1)
        if len(labels.shape) == 1:
            labels_1D = labels
        else:
            labels_1D = np.argmax(labels, 1)
        confusion_cnt =np.zeros((preds.shape[1],preds.shape[1]), dtype=np.int16)
        mapping_img_path = { "label_"+str(i) : {} for i in range(preds.shape[1])}
        for k,_ in mapping_img_path.items():
            mapping_img_path[k] = { "pred_"+str(i) : [] for i in range(preds.shape[1])}
            #print (k,mapping_img_path[k])
            
        for i in range(labels_1D.shape[0]):
            confusion_cnt[labels_1D[i],preds_1D[i]] += 1
            mapping_img_path["label_"+str(labels_1D[i])]["pred_"+str(preds_1D[i])].append(i)
            
        confusion_acc = confusion_cnt.astype(float) /  \
                    np.expand_dims(np.sum(confusion_cnt,axis = 1),axis = 1) *100
        
        confusion_acc [ confusion_acc != confusion_acc] = 0 # replace nan as 0        
        return confusion_acc , confusion_cnt, mapping_img_path
    
    def Top_N_acc(self, preds, labels,N):
        Top_N_preds = preds.argsort()[:,-N:][:,::-1]
        Top1_labels = np.expand_dims(np.argmax(labels, 1),axis = 1)
        return (100.0 * np.sum(Top_N_preds == Top1_labels)/ preds.shape[0])
 
    
class PeriodicSaver(object):
  
    def __init__(self, save_path, period_secs):
        super(PeriodicSaver, self).__init__()
        self._save_path = save_path
        self._period_secs = period_secs
        self.session = None

        self._saver = tf.train.Saver()

        self._last_save_time = -1

        save_dir = os.path.dirname(self._save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def save(self, global_step):
        if time.time() - self._last_save_time < self._period_secs:
            return
        self._last_save_time = time.time()
        
        self._saver.save(self.session, self._save_path+"/graph.chkp", global_step=global_step)
        
def main():
    
	
    
    Img_W = 112
    Img_H = 112
    tf.reset_default_graph()
    g1 = tf.Graph()
    with g1.as_default() :
        tf.logging.set_verbosity(tf.logging.INFO)
        wid = "{:02d}-{:02d}_{:02d}-{:02d}".format( time.localtime()[1], time.localtime()[2], \
                                      time.localtime()[3],time.localtime()[4])
        #tf.logging.info("work ID: %s", wid)
        exp_dir = "Model/{}".format(wid)
        #exp_dir = "Log\\#41_04-10_12-13"
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        
        log_period_secs = 50
        logger = Logger(wid, exp_dir+"\log.txt", log_period_secs)
        FR_queue = Queue(maxsize=100)
        		
        #train_sets = ["FR_asian_training","FR_west_training"]
        train_sets = ["FR_asian_valid","FR_west_valid"]
        FR_f_bin_path_train = []
        FR_indexs_train = []
        FR_data_dir = "training"
        for name in train_sets:
            FR_file_name = name+".idx"
            FR_f_bin_path_train.append(os.path.join(FR_data_dir,name+".bin"))
            _,indexs_temp,_,_,_ = load_index_lists(FR_data_dir,FR_file_name)
            FR_indexs_train.append(indexs_temp)
		
        valid_sets = ["FR_asian_valid","FR_west_valid"]
        FR_f_bin_path_valid = []
        FR_indexs_valid = []
        FR_data_dir = "training"
        for name in valid_sets:
            FR_file_name = name+".idx"
            FR_f_bin_path_valid.append(os.path.join(FR_data_dir,name+".bin"))
            _,indexs_temp,_,_,_ = load_index_lists(FR_data_dir,FR_file_name)
            FR_indexs_valid.append(indexs_temp)

        train_IDs = 0 
        valid_IDs = 0
        for list in FR_indexs_train:
            train_IDs += len(list)
        for list in FR_indexs_valid:
            valid_IDs += len(list)
		
        print ("train_num: ", train_IDs)
        print ("valid_num: ", valid_IDs)
        print(FR_indexs_train[0][0].shape)
        #return
        FR_thread_num = 16
        data_loader = []
        for i in range(FR_thread_num):
            data_loader.append(FR_Data_Thread_by_Class(i+1,time.time()+i,FR_indexs_train,FR_f_bin_path_train,30,5,Img_H,Img_W,"FR",FR_queue))
            data_loader[i].start()
        
        FR_valid_env = FR_Data_Thread_by_Class(8,time.time() + 8,FR_indexs_valid,FR_f_bin_path_valid,300,2,Img_H,Img_W,"FR",FR_queue)
        FR_valid_env.open_bin()
        FR_valid_data = FR_valid_env.get_data()
        FR_valid_env.close_bin()
		

        #train_batch = {"img":np.ndarray(shape=[10,Img_H,Img_W,3])}
        train_batch = FR_queue.get()
        FR_queue.task_done()
        print("FR_valid ",FR_valid_data["img"].shape)
        net = ArcFace.Res50_Arc_loss(train_batch,train_IDs,g1)
        task = net._task

        del train_batch
        del FR_valid_env
        
        var_init_op = tf.global_variables_initializer()
    
        cp_path = exp_dir+"/checkpoint"
        save_period_secs = 300
        saver = PeriodicSaver(cp_path, save_period_secs)
		
        #saved_file_path = "Model/#41_04-10_12-13/eval_cp"
        saved_file_path= None
        with tf.Session(config=net._config) as sess:
            net.session = sess
            sess.run(var_init_op)       
            _log_trainable_variables()
  
            if saved_file_path == None:
                net.load_pre_trained_ckpt("Model/resnet_v1_50.ckpt")
            else:
                saver_ckpt = tf.train.Saver()  
                ckpt = tf.train.get_checkpoint_state(saved_file_path)
                print (ckpt.model_checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver_ckpt.restore(sess,ckpt.model_checkpoint_path)
            eval_train_secs = 50
            eval_valid_secs = 200
            valid_save_path = exp_dir+"/eval_cp/a"
            train_save_path = ""
            evaluator_FR = Evaluator_FR(net, logger, eval_train_secs,eval_valid_secs, train_save_path,valid_save_path, sess)
            #evaluator_Gender = Evaluator_non_FR(net, logger, eval_train_secs,eval_valid_secs, train_save_path,valid_save_path, sess)
            #evaluator_Age = Evaluator_non_FR(net, logger, eval_train_secs,eval_valid_secs, train_save_path,valid_save_path, sess)
            saver.session = sess
            print ("start_training")
            count = 0
            #sess.run(tf.assign(net._lr_rate[task[0]],1e-5))
            sess.run(tf.assign(net._lr_rate[task[1]],1e-7))
            sess.run(tf.assign(net._lr_rate[task[2]],1e-7))
            training_stop = False
            while (not training_stop):
                if (count%800 == 0):
                    print("FR_queue.size{:3d}".format(FR_queue.qsize()))

                count += 1
                if count%3 == 0 and net._lr_rate[task[0]].eval() >= 1e-6:
                    train_batch = FR_queue.get()
                    FR_queue.task_done()
                    evaluator_FR.train_eval(train_batch,task[0])
                    net.train_one_step(train_batch,task[0])
                    evaluator_FR.valid_eval(FR_valid_data,task[0])
                    logger.log(net.step_cnt())
                    logger._vals.clear()
                """
                elif count%3 == 1 and net._lr_rate[task[1]].eval() >= 1e-6:
                    #print(task[1]+"_L2_loss: ",net._l2_loss[task[1]].eval())
                    train_batch = Gender_queue.get()
                    Gender_queue.task_done()
                    #t1 = time.time()
                    evaluator_Gender.train_eval(train_batch,task[1])
                    net.train_one_step(train_batch,task[1])
                    evaluator_Gender.valid_eval(Gender_valid_data,task[1])
                    #sum_time_2 += time.time() - t1
                    logger.log(net.step_cnt())
                    logger._vals.clear()
				
                elif count%3 == 2 and net._lr_rate[task[2]].eval() >= 1e-6:
                    #print(task[2]+"_L2_loss: ",net._l2_loss[task[2]].eval())
                    train_batch = Age_queue.get()
                    Age_queue.task_done()
                    #t1 = time.time()
                    evaluator_Age.train_eval(train_batch,task[2])
                    #net.train_one_step(train_batch,task[2])
                    evaluator_Age.valid_eval(Age_valid_data,task[2])
                    #sum_time_2 += time.time() - t1
                    logger.log(net.step_cnt())
                    logger._vals.clear()
                else:
                    continue
                """
                saver.save(net.step_cnt())			
                training_stop = True
                for key in task:
                    if net._lr_rate[key].eval() >= 1e-6:
                        training_stop = False        
					
            model_path = exp_dir+"/frozen_model.pb"
            net.freeze_deploy_graph(model_path)
            if valid_save_path != "":
                path = exp_dir+"/eval_cp"
                ckpt = tf.train.get_checkpoint_state(path)
                print (ckpt.model_checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver._saver.restore(sess,ckpt.model_checkpoint_path)
                model_path = exp_dir+"/frozen_model_valid.pb"
                net.freeze_deploy_graph(model_path)
                
            for i in range(FR_thread_num):
                data_loader[i]._thread_stop=True
                data_loader[i].join()
            for i in range(thread_num_Gender):
                Gender_loader[i]._thread_stop=True
                Gender_loader[i].join()				
            for i in range(thread_num_Age):
                Age_loader[i]._thread_stop=True
                Age_loader[i].join()
				
				
            
            while not FR_queue.empty():
                FR_queue.get()
                FR_queue.task_done() 
            FR_queue.join()
			
            while not Gender_queue.empty():
                Gender_queue.get()
                Gender_queue.task_done() 
            Gender_queue.join()
			
            while not Age_queue.empty():
                Age_queue.get()
                Age_queue.task_done() 
            Age_queue.join()

            print ('exiting main thread')
        
            
if __name__ == "__main__":
    main()              