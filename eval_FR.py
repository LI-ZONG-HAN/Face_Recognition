import numpy as np
import tensorflow as tf
from PIL import Image
from Net import triplet_loss as triplet
import time
import threading
from queue import Queue
import os
import dlib
import argparse

def get_input (file_path):
    crop_width = 150
    crop_height = 150
    img = Image.open(file_path)
    width, height = img.size
    img = img.convert('L')
    left = (width - crop_width)/2
    top = (height - crop_height)/2
    right = (width + crop_width)/2
    bottom = (height + crop_height)/2
    cropped_img = img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img).reshape(1,crop_height,crop_width,1)
    cropped_img = (cropped_img - 225./2)/(225./2)
    return cropped_img
	
	
def load_path_lists(data_dir):
    lists = os.listdir(data_dir)
    lists = [os.path.join(data_dir,f) for f in lists]
    lists = [f for f in lists if os.path.isdir(f)]
    results = []
    for f in lists:
        temp_array = np.array([os.path.join(f,path) for path in os.listdir(f)])
        results.append(temp_array)
    
    return results
	
def load_path_lm_lists(data_dir,flie_name):
    with open(os.path.join(data_dir,flie_name),"r") as f:
        lines = f.readlines()
        lines = [f.strip() for f in lines]   
        temp_path = np.array([data_dir+"/"+f.strip().split(" ",1)[0] for f in lines])
        temp_lm_list = np.array([f.strip().split(" ",1)[1].split(" ")[1:] for f in lines],dtype = np.float32)
        label_list = [f.split("/")[3] for f in temp_path]
        hush_table = {}
        path_list = []
        lm_list =[]
        for i,l in enumerate(label_list):
            if l not in hush_table.keys():
                hush_table[l] = [i]
            else:
                hush_table[l] += [i]
        for k,v in hush_table.items():
            path_list.append(temp_path[v])
            lm_list.append(temp_lm_list[v])
        del temp_path
        del temp_lm_list
        del label_list
        del hush_table
    return path_list,lm_list
	
class Data_Thread(threading.Thread):
    def __init__(self, threadID, seed, path_list,lm_list, person_no, img_no_person, img_height,img_width,q,take_all):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.seed = int(seed)
        self.queue = q
        self._person_no = person_no
        self._img_no_person = img_no_person
        self._batch_size = person_no*img_no_person
        self._img_height = img_height
        self._img_width = img_width
        self._channels = 3
        self._thread_stop = False 
        self._detector = dlib.get_frontal_face_detector()
        self._sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
        self._path_list = path_list
        self._lm_list = lm_list
        self._total_id = len(self._path_list)
        self._padding = 0.25
        self._take_all = take_all
       
    
    def _randomize(self, lists, seed):
        permutation = np.random.RandomState(seed=seed).permutation(lists.shape[0])
        shuffled_lists = lists[permutation]
        return shuffled_lists
      
    def get_data(self):
        res = {
            "img": np.ndarray(shape=(self._batch_size, self._img_height, self._img_width, self._channels), dtype=np.float32),
            "label": np.ndarray(shape=(self._batch_size), dtype=np.int32)
        }
        count = 0
        labels = np.arange(self._total_id)
        np.random.shuffle(labels)
        #labels = labels[:min(self._person_no*2,self._total_id)]
        ii = 0
        #print (labels)
        for label in labels:
            #print ("label_test",label)
            if (ii >= self._person_no):
                break

            if (not self._take_all) and self._path_list[label].shape[0] < 2:
                #print ("person_id = %d need more image" %(label))
                continue
            
            #indexs = np.random.randint(self._path_list[label].shape[0], size = self._img_no_person)
            indexs = np.arange(self._path_list[label].shape[0])
            np.random.shuffle(indexs)
            no_imgs = 0
            for index in indexs: 
                if (no_imgs >= self._img_no_person):
                    break
                img = Image.open(self._path_list[label][index])
                img = img.convert('RGB')
                if len(self._lm_list) == 0:
                    crop_img , no_face = self.FD_Crop_1_face(img, size = self._img_height, padding = self._padding)
                else:	
                    lm = self._lm_list[label][index]
                    crop_img, no_face = self.Crop_1_face_wiht_lm(img, lm , size = self._img_height , padding = self._padding)
					
                if not no_face:
                    #print (self._path_list[label][index])
                    continue
                    
                res["img"][count,:,:,:] = crop_img
                res["label"][count] = label
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
        #img = img.crop((left, top, rihgt, bottom))
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
        while not self._thread_stop:
            datas = self.get_data()
            try:
                self.queue.put(datas,True,100)
            except:
                print ("get time_out Thread_ID = %d" % self.threadID)
                
        print ("hread_ID = %d run end" % self.threadID)


def load_graph(frozen_graph_path):
    graph = tf.Graph()
    with tf.gfile.GFile(frozen_graph_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it 
    with graph.as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        
    return graph


def inference_img(graph,input_batch,dim):
    imgs = (input_batch["img"]- 255.0/2) / (255.0/2)
	
    #imgs = np.ndarray(input_batch["img"].shape,dtype = np.float32)
    #imgs[:,:,:,0] = (input_batch["img"][:,:,:,0] - 122.782)/256
    #imgs[:,:,:,1] = (input_batch["img"][:,:,:,1] - 117.001)/256
    #imgs[:,:,:,2] = (input_batch["img"][:,:,:,2] - 104.298)/256
    with graph.as_default():
        x = graph.get_tensor_by_name('input:0')
        embeddings = graph.get_tensor_by_name('embedding:0')
        #embeddings = graph.get_tensor_by_name('l2_normalize:0')
		
        #embeddings_norm = tf.nn.l2_normalize(embeddings, axis=1)
        emb_input = tf.placeholder(name="emb_input", shape=[None,dim], dtype=tf.float32)
        
        pairwise_dist = triplet._pairwise_distances(emb_input)
        Same = Same_person_eval(input_batch["label"],pairwise_dist)
        Diff = Diff_person_eval(input_batch["label"],pairwise_dist)
        run_ops = {"Same":Same, "Diff":Diff}
        cut_interval = 20
        
        with tf.Session(graph = graph) as sess:
            total_num = imgs.shape[0]
            emb_np = np.ndarray(shape=[total_num,dim], dtype=np.float32)
            cut_ind = np.arange(0,total_num,cut_interval)
            if cut_ind[-1] != total_num:
                cut_ind = np.append(cut_ind,total_num)
            
            for i in range (cut_ind.shape[0]-1):
                start = cut_ind[i]
                end = cut_ind[i+1]
                temp = sess.run(embeddings,feed_dict = {x:imgs[start:end]})
                #print (temp.shape)
                emb_np[start:end] = temp
            
            return sess.run(run_ops,feed_dict = {emb_input:emb_np}) , emb_np
			
def inference_img_dlib(graph,input_batch, dim):
    imgs = np.ndarray(input_batch["img"].shape,dtype = np.float32)
    imgs[:,:,:,0] = (input_batch["img"][:,:,:,0] - 122.782)/256
    imgs[:,:,:,1] = (input_batch["img"][:,:,:,1] - 117.001)/256
    imgs[:,:,:,2] = (input_batch["img"][:,:,:,2] - 104.298)/256
    with graph.as_default():
        x = graph.get_tensor_by_name('input:0')
        embeddings = graph.get_tensor_by_name('fc/BiasAdd:0')
        #embeddings_norm = tf.nn.l2_normalize(embeddings, axis=1)
        emb_input = tf.placeholder(name="emb_input", shape=[None,dim], dtype=tf.float32)
        
        pairwise_dist = triplet._pairwise_distances(emb_input)
        Same = Same_person_eval(input_batch["label"],pairwise_dist)
        Diff = Diff_person_eval(input_batch["label"],pairwise_dist)
        run_ops = {"Same":Same, "Diff":Diff}
        cut_interval = 20
        with tf.Session(graph = graph) as sess:
            total_num = imgs.shape[0]
            emb_np = np.ndarray(shape=[total_num,dim], dtype=np.float32)
            cut_ind = np.arange(0,total_num,cut_interval)
            if cut_ind[-1] != total_num:
                cut_ind = np.append(cut_ind,total_num)
            for i in range (cut_ind.shape[0]-1):
                start = cut_ind[i]
                end = cut_ind[i+1]
                temp = sess.run(embeddings,feed_dict = {x:imgs[start:end]})
                #print (temp.shape)
                emb_np[start:end] = temp
            return sess.run(run_ops,feed_dict = {emb_input:emb_np}) , emb_np
			
			
def Same_person_eval(labels, pairwise_dist):
    
    mask_anchor_positive = triplet._get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
    return anchor_positive_dist

def Diff_person_eval(labels, pairwise_dist):
    
    mask_anchor_negative = triplet._get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_negative_dist = tf.multiply(mask_anchor_negative, pairwise_dist)
    return anchor_negative_dist

def main():
    
    parser = argparse.ArgumentParser(description = 'Eval FR model')
    parser.add_argument('eval_datasets', type=int, help='pls chose one of 6 datasets, 0:LFW, 1:asian_train, 2:asian_valid, 3:west_train, 4:west_valid, 5:Geo_test')
    parser.add_argument('-model', required=True, type=str, help='Path to trained FR model')
    parser.add_argument('-w', '--imgage_width', type=int, default = 112, help='(optional) imgage_width Default: 112')
    parser.add_argument('-dim', '--Embedding_Dims', type=int, default = 512, help='(optional) imgage_width Default: 512')
    args = parser.parse_args()
    task = ["LFW","asian_valid","asian_asianing", "West_valid", "West_training", "Geo_test"]
    _info = {}
    _info[task[0]] = {"path":"Data/LFW","load_type":0, "IDs_C":1500, "C_per_ID":2,"take_all":0}
    _info[task[1]] = {"dir":"training/FR_original_data","file":"asian_valid","load_type":1, "IDs_C":500, "C_per_ID":2,"take_all":0}
    _info[task[2]] = {"dir":"training/FR_original_data","file":"asian_training","load_type":1, "IDs_C":1000, "C_per_ID":2,"take_all":0}
    _info[task[3]] = {"dir":"training/FR_original_data","file":"West_valid","load_type":1, "IDs_C":500, "C_per_ID":2,"take_all":0}
    _info[task[4]] = {"dir":"training/FR_original_data","file":"West_training","load_type":1, "IDs_C":1000, "C_per_ID":2,"take_all":0}
    _info[task[5]] = {"path":"Data/Geo_test_set/Geo_Test_set","load_type":0, "IDs_C":900, "C_per_ID":30,"take_all":1}
    if args.eval_datasets < 0 or args.eval_datasets > 5 :
        print("eval_datasets pls chose 0~5")
        return
    
    eval_item = task[args.eval_datasets]
    model_path = args.model
    Img_W = args.imgage_width
    Img_H = Img_W
    emb_dim = args.Embedding_Dims
    print("{:15}{}".format("eval_item",eval_item))
    print("{:15}{}".format("model_path",model_path))
    print("{:15}{}".format("image_width",Img_W))
    print("{:15}{}".format("emb_dim",emb_dim))
    for k,v in _info[eval_item].items():
        print("{:15}{}".format(k,v))

    tf.reset_default_graph()
    my_queue = Queue(maxsize=100)
    g2 = load_graph(model_path)
    data_thread1 = None
    if (_info[eval_item]["load_type"] == 0):
        path_lists = load_path_lists(_info[eval_item]["path"])
        data_thread1 = Data_Thread(1,time.time(), path_lists, [] ,_info[eval_item]["IDs_C"],_info[eval_item]["C_per_ID"],Img_H,Img_W,my_queue,_info[eval_item]["take_all"])
    else:
        valid_path,valid_lm = load_path_lm_lists(_info[eval_item]["dir"],_info[eval_item]["file"])
        data_thread1 = Data_Thread(1,time.time(), valid_path, valid_lm ,_info[eval_item]["IDs_C"],_info[eval_item]["C_per_ID"],Img_H,Img_W,my_queue,_info[eval_item]["take_all"])
	
    print ("load list finished")
    loop = 5
    if eval_item == "Geo_test":
        loop = 1
    for i in range(loop):
        input_batch = data_thread1.get_data()
        print ("batch load finished")	
        print (input_batch["img"].shape)
        res, emb_np = inference_img(g2,input_batch, emb_dim) 
        #print(emb_np)
        #print (np.mean(np.linalg.norm(emb_np,axis = 1)))
        Same = np.triu(res["Same"],1).flatten()
        Diff = np.triu(res["Diff"],1).flatten()
        Same = Same[Same>0]#[0:1500]
        Diff = Diff[Diff>0]
        print (Diff[Diff>2.0])
        Total_Same_count = Same.shape[0]
        Same_mean = np.mean(Same)
        Same_std = np.std(Same,ddof=1)
		
        Total_Diff_count = Diff.shape[0]
        Diff_mean = np.mean(Diff)
        Diff_std = np.std(Diff,ddof=1)
        thresholds = np.arange(0.4,1.8,0.01)
        acc_max = -1
        thres_max = -1
        t1 = time.time()
        for thres in thresholds:
            Same_right_count = Same[Same<thres.tolist()].shape[0]
            Diff_right_count = Diff[Diff>=thres.tolist()].shape[0]
            acc_T = Same_right_count/Total_Same_count*100
            acc_N = Diff_right_count/Total_Diff_count*100
            acc = (acc_T+acc_N)*0.5
            #print(acc,thres.tolist())
            if acc_max < acc:
                acc_max = acc
                thres_max = thres.tolist()
        print("time= ",time.time() - t1)
        #threshold = 1.22
        Same_right_count = Same[Same<thres_max].shape[0]
        Diff_right_count = Diff[Diff>=thres_max].shape[0]
        print ("Total_same_count:{:4d}, mean_dist{:6.3f}, std{:6.3f}".format(Total_Same_count,Same_mean,Same_std))
        print ("Total_Diff_count:{:4d}, mean_dist{:6.3f}, std{:6.3f}".format(Total_Diff_count,Diff_mean,Diff_std))
        print ("Gap:{:6.3f}".format((Diff_mean-Same_mean)/Diff_std))
        print ("Same_accuracy:{:6.2f}".format( Same_right_count/Total_Same_count*100 ))
        print ("Diff_accuracy:{:6.2f}".format( Diff_right_count/Total_Diff_count*100 ))
        print ("threshold:{:6.2f}".format(thres_max))
	
    
	
	
	
	
    
    
if __name__ == "__main__":
    main()