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

def pre_process(input_batch):
    imgs = (input_batch - 255.0/2) / (255.0/2)
    return imgs
	
def load_path_lists(data_dir):
    lists = os.listdir(data_dir)
    lists = [os.path.join(data_dir,f) for f in lists]
    lists = [f for f in lists if os.path.isdir(f)]
    results = []
    for f in lists:
        temp_array = [os.path.join(f,path) for path in os.listdir(f)]
        results.append(temp_array)
    
    return results
	
def load_path_lm_lists(data_dir,flie_name):
    with open(os.path.join(data_dir,flie_name),"r") as f:
        lines = f.readlines()
        lines = [f.strip() for f in lines]   
        temp_path = np.array([data_dir+"/"+f.strip().split(" ",1)[0] for f in lines])
        temp_lm_list = np.array([f.strip().split(" ",1)[1].split(" ")[1:] for f in lines],dtype = np.float32)
        label_list = [f.split("/")[2] for f in temp_path]
        hush_table = {}
        path_list = []
        lm_list =[]
        keys_list_temp=[]
        for i,l in enumerate(label_list):
            if l not in hush_table.keys():
                hush_table[l] = [i]
                keys_list_temp.append(l)
            else:
                hush_table[l] += [i]
        #for k,v in hush_table.items():
        #    path_list.append(temp_path[v])
        #    lm_list.append(temp_lm_list[v])
        for i,l in enumerate(keys_list_temp):
            path_list.append(temp_path[hush_table[l]])
            lm_list.append(temp_lm_list[hush_table[l]])
        del temp_path
        del temp_lm_list
        del label_list
        del hush_table
    return path_list,lm_list
	
class Data_Thread(threading.Thread):
    def __init__(self, threadID,batch_size, img_height,img_width, jitter_count,q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.queue = q
        self._batch_size = batch_size
        self._img_height = img_height
        self._img_width = img_width
        self._channels = 3
        self._thread_stop = False 
        self._detector = dlib.get_frontal_face_detector()
        self._sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
        self._padding = 0.25
        self.start_index = 0
        self._jitter_count = jitter_count
      
    def get_data(self):
        global index
        global path_list
        global label_list
        global g_Lock
        global detect_c
        global Img_Error_list_path
        with g_Lock:
            m_index = index
            global_list_len = len(path_list)
            end = min(m_index + self._batch_size, global_list_len)
            if (m_index == end):
                return None
            m_path_list = path_list[m_index:end]
            m_label_list = label_list[m_index:end]
            index = end
        
        res = {
            "img": np.ndarray(shape=(self._batch_size, self._img_height, self._img_width, self._channels), dtype=np.float32),
            "label":np.ndarray(shape=(self._batch_size), dtype=np.int32),
            "path_list": [],
            "last_batch": False
        }
        if end == global_list_len:
            res["last_batch"] = True
        
        count = 0
        for i,path in enumerate(m_path_list):

            try :
                img = Image.open(path)
                img = img.convert('RGB')
                h,w = img.size
                if (h == 150 and w == 150):
                    crop_img = img.resize((self._img_height, self._img_width), Image.ANTIALIAS)
                    crop_img = np.array(crop_img)
                else:
                    crop_img, _ = self.Crop_1_face_no_FD(img, self._img_height , self._padding)

                res["img"][count,:,:,:] = crop_img
                res["label"][count] = m_label_list[i]
                res["path_list"].append(path)
                count += 1
                
            except:
                print("index= ",i+ m_index)
                print("load_img_error: ",path)
                with open(Img_Error_list_path,"a") as f:
                    f.write(path+"\n")

        if len(res["path_list"]) == 0:
            return res
        elif (len(res["path_list"]) < self._batch_size):
            res["img"] = res["img"][0:len(res["path_list"])]
            res["label"] = res["label"][0:len(res["path_list"])]
            

			
        detect_c += res["img"].shape[0]

        if self._jitter_count:
            list_imgs =[]
            for i in range(res["img"].shape[0]):
                list_imgs += dlib.jitter_image(np.uint8(res["img"][i]), num_jitters=self._jitter_count, disturb_colors=True)
            res["img"] = np.array(list_imgs,dtype = np.float32)

        
        #res["img"] = pre_process(res["img"])

        return res
       
    
    def Crop_1_face_no_FD (self,img, size = 224 , padding = 0.25):
        h,w = img.size
        #eye_dist = lm[2] - lm[0]
        #extend = 1.5
        left = 0
        top = 0
        rihgt = w
        bottom = h
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
            return None , num_face, None, None
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
        return image, num_face, dets[index], faces[0]
                
    def run(self):
        global index
        global path_list
        while not self._thread_stop:
            if index > len(path_list):
                self._thread_stop = True
                break
            datas = self.get_data()
            if datas != None:
                if len(datas["path_list"]) == 0:
                    continue
                self.queue.put(datas)
            else:
                self._thread_stop = True
                break
            #try:
            #self.queue.put(datas,True,100)
            #except:
            #    print ("get time_out Thread_ID = %d" % self.threadID)
                
        print ("Load_Thread_ID = %d run end" % self.threadID)

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

    with graph.as_default():
        x = graph.get_tensor_by_name('input:0')
        Gender = graph.get_tensor_by_name('Gender:0')

        cut_interval = 20
        with tf.Session(graph = graph) as sess:
            total_num = imgs.shape[0]
            sofemax_np = np.ndarray(shape=[total_num,dim], dtype=np.float32)
            cut_ind = np.arange(0,total_num,cut_interval)
            if cut_ind[-1] != total_num:
                cut_ind = np.append(cut_ind,total_num)
            
            for i in range (cut_ind.shape[0]-1):
                start = cut_ind[i]
                end = cut_ind[i+1]
                temp = sess.run(Gender,feed_dict = {x:imgs[start:end]})
                #print (temp.shape)
                sofemax_np[start:end] = temp
            
            return sofemax_np		

def Confusion_M(preds, labels):
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
			


parser = argparse.ArgumentParser(description = 'eval gender Age with label')
parser.add_argument('Gender_or_Age', type=str, help='pls type Gender or Age"')
parser.add_argument('-dir', required=True, type=str, help='Path to floder of eval dataset')
parser.add_argument('-model', '--load_model_path', required=True, type=str, help='Path to trained FR model, type is tensorflow pb file')
parser.add_argument('-img_w', '--imgage_width', type=int, default = 112, help='(optional) imgage_width Default: 112')
parser.add_argument('-fr_dim', '--FR_Emb_Dim', type=int, default = 512, help='(optional) FR_Embedding_Dims Default: 512')
args = parser.parse_args()	

task = ["Gender","Age"]
eval_item = task.index(args.Gender_or_Age) if (args.Gender_or_Age in task) else -1
emb_dim = args.FR_Emb_Dim
data_floder = args.dir
model_path = args.load_model_path
img_W = args.imgage_width
img_H = img_W
print("{:15}{}".format("eval_item",task[eval_item]))
print("{:15}{}".format("data_floder",data_floder))
print("{:15}{}".format("model_path",model_path))
print("{:15}{}".format("image_width",img_W))
print("{:15}{}".format("emb_dim",emb_dim))

if (eval_item < 0):
    print("training_item not found in task, task = [\"Gender\",\"Age\"]")
else:
	tf.reset_default_graph()
	#emb_dim = 512
	index = 0
	FD_Lost_c = 0
	detect_c = 0

	g_Lock = threading.Lock()


	if eval_item:
        #Age_Dataset_dir = "training/age_data/__age_valid_2"
	    path_lists_temp = load_path_lists(data_floder)
	    print(len(path_lists_temp))
	else:
	    #Gender_Dataset_dir = "training/gender_data/__gender_valid"
	    path_lists_temp = load_path_lists(data_floder)
	    path_lists_temp = path_lists_temp[0:2]


	Num_Classes = len(path_lists_temp)
	path_list = []
	label_list = []
	for i in range(len(path_lists_temp)):
		label_list += [i for l in path_lists_temp[i]]	
		path_list += path_lists_temp[i]

	print(len(label_list))
	print(len(path_list))


	my_queue = Queue(maxsize=100)
	batch_size = 20
	thread_num = 5
	
	jitter_count = 0


	g2 = load_graph(model_path)
	data_loader = []
	for i in range(thread_num):
		data_loader.append(Data_Thread(i+1,batch_size, img_H,img_W, jitter_count, my_queue))
		data_loader[i].start()

		
		
		

	"""
	jitter_count = 0
	run_count = 0
	last_batch = False
	while(1):
		if run_count%50==0:
			print ("batch_run= ",run_count," Index= ", index)
					
		run_count += 1

		if last_batch and my_queue.empty():
			test_bool = True
			for i in range(thread_num):
				test_bool = (test_bool and data_loader[i]._thread_stop)
			if test_bool:
				break

		
		test_batch = my_queue.get()
		my_queue.task_done()
		if not last_batch:
			last_batch = test_batch["last_batch"]
			
		count = 0
		if jitter_count:
			list_imgs =[]
			for i in range(test_batch["img"].shape[0]):
				print(count)
				count += 1
				list_imgs += dlib.jitter_image(np.uint8(test_batch["img"][i]), num_jitters=jitter_count, disturb_colors=True)
			imgs = np.array(list_imgs,dtype = np.float32)
		else:
			imgs = test_batch["img"]
			
		floder = "landmark_check"
		if jitter_count:
			for i,path in enumerate(test_batch["path_list"]):
				file_name, ext = path.rsplit("\\",1)[-1].rsplit(".",1)
				for j in range(jitter_count):
					img_s= Image.fromarray(np.uint8(imgs[i*jitter_count + j]))
					new_path = os.path.join(floder,file_name+"_"+str(j)+"."+ext)
					print(new_path)
					img_s.save(new_path)
		else:
			for i,path in enumerate(test_batch["path_list"]):
				img_s= Image.fromarray(np.uint8(imgs[i]))
				img_s.save(os.path.join(floder,path.rsplit("\\",1)[-1]))
		
	"""


	print ("total_img_no = ",len(path_list))
	last_batch = False
	tt1 = tt2 = tt3 = tt4 = tt5 = tts = 0
	t1 = time.time()
	Total_Pred = np.ndarray(shape=[len(path_list),Num_Classes],dtype = np.float32)
	Total_label = np.ndarray(shape=[len(path_list)],dtype = np.int32)
	start = 0
	with g2.as_default() :
		x = g2.get_tensor_by_name('input:0')
		Gender_out = g2.get_tensor_by_name('Gender:0')
		Age_output = g2.get_tensor_by_name('Age:0')
		if eval_item:
			output = Age_output
		else:
			output = Gender_out
		with tf.Session(graph = g2) as sess:
			print ("start_inference")
			run_count = 0
			#tt1 = tt2 = tt3 = tt4 = tt5 = tts = 0
			while(1):
				if run_count%50==0:
					print ("batch_run= ",run_count," Index= ", index," queue_size= ",my_queue.qsize(),"  ",tt1,tt3,tt4)
					#print ("my_queue.size= ",my_queue.qsize())
					#print (tt1,tt2,tt3,tt4)
					tt1 = tt2 = tt3 = tt4 = tt5 = tts = 0
				run_count += 1

				if last_batch and my_queue.empty():
					test_bool = True
					for i in range(thread_num):
						test_bool = (test_bool and data_loader[i]._thread_stop)
					if test_bool:
						break

				tt_s = time.time()
				test_batch = my_queue.get()
				my_queue.task_done()
				tt1  += time.time() - tt_s
				tt_s = time.time()


				if not last_batch:
					last_batch = test_batch["last_batch"]
				
				
				#ttt_s = time.time()
				#for i in range(50):
				#    pre_process_dlib(test_batch["img"])
				#print(time.time() - ttt_s)
				#print(test_batch["img"].shape, test_batch["img"].dtype)
				#print(test_batch["img_test"].shape, test_batch["img_test"].dtype)
				

				
				
				#Num_img = len(test_batch["path_list"])
				#test_batch["img"] = np.concatenate((test_batch["img"],test_batch["img"][:,:,::-1,:]),axis = 0) #img flip
				imgs = pre_process(test_batch["img"])

			   
				
				
				if jitter_count:
					N = len(test_batch["path_list"])
					emb_np = np.ndarray(shape=[N,emb_dim],dtype = np.float32)
					emb_np_temp = sess.run(output,feed_dict = {x:imgs})
					for i in range(N):
						emb_np[i] = np.mean(emb_np_temp[i*jitter_count: (i+1)*jitter_count],axis = 0)
					
				else:
					emb_np = sess.run(output,feed_dict = {x:imgs})


				Total_Pred[start:start+emb_np.shape[0]] = emb_np
				Total_label[start:start+emb_np.shape[0]] = test_batch["label"]
				start += emb_np.shape[0]
				tt3 += time.time() - tt_s
				tt_s = time.time()
				
			   
				#emb_add = emb_np[0:Num_img] + emb_np[Num_img:]
				#emb_add = emb_add / np.linalg.norm(emb_add,axis = 1).reshape(-1,1)
				
				
							

				tt4 += time.time() - tt_s
				tt_s = time.time()
				
				

	print ("total_time= ", time.time()-t1)


		
	for i in range(thread_num):
		data_loader[i]._thread_stop=True
		data_loader[i].join()	
		
	acc,cnt,_ = Confusion_M(Total_Pred,Total_label)
	print(Total_Pred.shape)
	print(Total_label.shape)
	average_acc = np.average(acc.diagonal())
	print("average_accuracy: ",average_acc)
	for a in cnt:
		print(a)

	for a in acc:
		print(a)
	
	
	
    
"""
if __name__ == "__main__":
    main()
"""