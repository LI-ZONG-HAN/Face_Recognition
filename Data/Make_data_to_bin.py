import numpy as np
import tensorflow as tf
from PIL import Image
#from Net import triplet_loss as triplet
import time
import threading
from queue import Queue
import os
import dlib

from sys import getsizeof
import io
import argparse

"""
def write_to_bin(f_bin,f_inx,img_path):
    with open(img_path,"rb") as f_img:
        start_index = f_bin.tell()
        f_bin.write(f_img.read())
        end_index = f_bin.tell()
    f_inx.write("{} {:d} {:d}\n".format(img_path,start_index,end_index))
"""
def write_to_bin(f_bin,f_inx,img_path,img_label,img_bytes):
    start_index = f_bin.tell()
    f_bin.write(img_bytes)
    end_index = f_bin.tell()
    f_inx.write("{}\t{:d}\t{:d}\t{:d}\n".format(img_path,img_label,start_index,end_index))
	
def load_path_lists(data_dir):
    lists = os.listdir(data_dir)
    lists = [os.path.join(data_dir,f) for f in lists]
    #print(lists)
    lists = [f for f in lists if os.path.isdir(f)]
    results = []
    labels = []
    for i, f in enumerate(lists):
        temp_array = [os.path.join(f,path).replace("\\","/") for path in os.listdir(f)]
        #results.append(temp_array)
        results += temp_array
        labels += [ i for f in range(len(temp_array))]
    
    return np.array(results), np.array(labels, dtype = np.int32)
	
def load_path_lists_FR(data_dir,flie_name):
    with open(os.path.join(data_dir,flie_name).replace("\\","/"),"r") as f:
        lines = f.readlines()
        lines = [f.strip() for f in lines]
        label_list = [f.strip().split(" ",1)[0].split("/")[1] for f in lines]
        path_list = np.array([data_dir+"/"+f.strip().split(" ",1)[0].replace("\\","/") for f in lines])
        lm_list = np.array([f.strip().split(" ",1)[1].split(" ")[1:] for f in lines],dtype = np.float32)
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
	
def load_path_lists_Gender_Age(data_dir,flie_name,gender = True):
    with open(os.path.join(data_dir,flie_name).replace("\\","/"),"r") as f:
        lines = f.readlines()
        lines = [f.strip() for f in lines]
        label_list = [f.strip().split(" ",1)[0].split("/")[1] for f in lines]
        if gender:
            labels = np.array([f.strip().split(" ")[2] for f in lines],dtype = np.uint64)
        else:
            labels = np.array([f.strip().split(" ")[3] for f in lines],dtype = np.uint64)
            labels[labels>6] = 6
        path_list = np.array([data_dir+"/"+f.strip().split(" ",1)[0].replace("\\","/") for f in lines])
        lm_list = np.array([f.strip().split(" ",1)[1].split(" ")[3:] for f in lines],dtype = np.float32)
		
        hush_table = {}
        for i in range(np.amax(labels).tolist()+1):
            hush_table[i] = []

        path_list_by_id = []
        lm_list_by_id =[]
        for i,l in enumerate(labels):
            hush_table[l] += [i]

        for k in range(np.amax(labels).tolist()+1):
            path_list_by_id.append(path_list[hush_table[k]])
            lm_list_by_id.append(lm_list[hush_table[k]])
		
        del hush_table
    return path_list_by_id, lm_list_by_id, path_list, lm_list, labels
	
class Data_Thread(threading.Thread):
    def __init__(self, threadID,batch_size, img_height,img_width,padding,q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.queue = q
        self._batch_size = batch_size
        self._img_height = img_height
        self._img_width = img_width
        self._channels = 3
        self._thread_stop = False 
        self._detector = dlib.get_frontal_face_detector()
        self._sp = dlib.shape_predictor("../shape_predictor_5_face_landmarks.dat")
        self._padding = padding
        self.start_index = 0
        #self._jitter_count = jitter_count
      
    def get_data(self):
        global index
        global train_paths
        global train_lms
        global train_labels
        global g_Lock
        global detect_c
        with g_Lock:
            m_index = index
            global_list_len = len(train_paths)
            end = min(m_index + self._batch_size, global_list_len)
            if (m_index == end):
                return None
            m_path_list = train_paths[m_index:end]
            if train_lms.shape[0] == 0:
                m_train_lm = np.array([None])
            else:
                m_train_lm = train_lms[m_index:end]
            m_labels = train_labels[m_index:end]
            index = end
        
        res = {
            "img": [],
            "path_list": [],
            "label" :[],
            "last_batch": False
        }
        if end == global_list_len:
            res["last_batch"] = True
        
        count = 0
        for i,path in enumerate(m_path_list):

            #try :
            img = Image.open(path)
            img = img.convert('RGB')
            if m_train_lm.any() == None:
                crop_img, _ = self.Crop_1_face_no_FD (img, self._img_height , self._padding)
            else:
                crop_img, _ = self.Crop_1_face_wiht_lm(img, m_train_lm[i] , self._img_height , self._padding)
            crop_img = Image.fromarray(np.uint8(crop_img))
            #h,w = img.size
            #if (h != 150 or w != 150):
            #    crop_img = img.resize((self._img_height, self._img_width), Image.ANTIALIAS)
            #else:
            #    crop_img = img
                
            byteIO = io.BytesIO()
            crop_img.save(byteIO, format='JPEG')
            res["img"].append(byteIO.getvalue())
            res["path_list"].append(path)
            res["label"].append(m_labels[i])
            count += 1
                
            #except:
            #    print("index= ",i+ m_index)
            #    print("load_img_error: ",path)


        if len(res["path_list"]) == 0:
            return res
        elif (len(res["path_list"]) < self._batch_size):
            res["img"] = res["img"][0:len(res["path_list"])]
            

			
        detect_c += len(res["img"])

        #if self._jitter_count:
        #    list_imgs =[]
        #    for i in range(res["img"].shape[0]):
        #        list_imgs += dlib.jitter_image(np.uint8(res["img"][i]), num_jitters=self._jitter_count, disturb_colors=True)
        #    res["img"] = np.array(list_imgs,dtype = np.float32)

        
        #res["img"] = pre_process(res["img"])

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
        global train_paths
        while not self._thread_stop:
            if index > len(train_paths):
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

		
tf.reset_default_graph()

parser = argparse.ArgumentParser(description = 'Make img data to binary')
parser.add_argument('is_table_use', type=str, help='is_table_use?  True:Table_use False: read floder direct')
parser.add_argument('is_FR_Gender_Age_dataset', type=str, help='is FR is_FR_Gender_Age_dataset?  FR or Gender or Age')
parser.add_argument('-dir', required=True, type=str, help='Path to root floder')
parser.add_argument('-f', '--file_name', required=True, type=str, help='Path to image floder or table file')
parser.add_argument('-out_name', '--output_name', type=str, default = "images", help='(optional) output_name Default: images')
parser.add_argument('-w', '--imgage_width', type=int, default = 224, help='(optional) imgage_width Default: 224')
parser.add_argument('-p', '--padding_ratio', type=float, default = 0.25, help='(optional) padding_ratio Default: 0.25')
args = parser.parse_args()




bool_list = {"True": True, "False": False}
type_list ={"FR":0, "Gender":1, "Age":2}
if not args.is_table_use in bool_list.keys():
    print("is_table_use typeing error, pls type True or False")
elif not args.is_FR_Gender_Age_dataset in type_list.keys():
	print("is_FR_Gender_Age_dataset typeing error, pls type FR or Gender or Age")

else:
	is_table_use = bool_list[args.is_table_use]
	data_type = type_list[args.is_FR_Gender_Age_dataset]
	data_dir = args.dir.replace("\\","/")
	file_path = args.file_name
	img_W = args.imgage_width
	img_H = img_W
	pad_ratio = args.padding_ratio
	out_name = args.output_name
	print("{:15}{}".format("is_table_use",is_table_use))
	print("{:15}{}".format("which_dataset",args.is_FR_Gender_Age_dataset))
	print("{:15}{}".format("data_dir",data_dir))
	print("{:15}{}".format("file_path",file_path))
	print("{:15}{}".format("img_W",img_W))
	print("{:15}{}".format("pad_ratio",pad_ratio))
	print("{:15}{}".format("out_name",out_name))
	
	train_paths = None
	train_lms = None
	train_labels = None
	if is_table_use:
		#data_dir = "training/FR_original_data"
		#FR_file_name = "West_training"
		if data_type == 0:
			_,_,train_paths,train_lms,train_labels = load_path_lists_FR(data_dir,file_path)
		else:
			gender = True if data_type==1 else False
			_,_,train_paths,train_lms,train_labels = load_path_lists_Gender_Age(data_dir,file_path,gender)
	else:
		#data_dir = "training/age_data"
		#floder = "__age_valid_2"
		train_paths, train_labels = load_path_lists(data_dir+"/"+file_path)
		train_lms = np.array([])
		
		
	#data_dir = "training/age_data"
	#floder = "__age_valid_2"
	#train_paths, train_labels = load_path_lists(data_dir+"/"+floder)
	#train_lms = np.array([])




	#data_dir = "training/FR_original_data"
	#FR_file_name = "West_training"
	#_,_,train_paths,train_lms,train_labels = load_path_lists_FR(data_dir,FR_file_name)







	index = 0
	FD_Lost_c = 0
	detect_c = 0
	g_Lock = threading.Lock()

	#print(train_paths[0])
	print(train_paths.shape)
	print(train_lms.shape)
	print(train_labels.shape)




	my_queue = Queue(maxsize=100)
	batch_size = 20
	thread_num = 1
	jitter_count = 0


	data_loader = []
	for i in range(thread_num):
		data_loader.append(Data_Thread(i+1,batch_size, img_H, img_W, pad_ratio, my_queue))
		data_loader[i].start()

		
		
		


	jitter_count = 0
	run_count = 0
	last_batch = False
	target_floder = data_dir
	bin_path = os.path.join(target_floder,out_name+".bin")#"FR_west_training_pad_15.bin")
	idx_path = os.path.join(target_floder,out_name+".idx")#"FR_west_training_pad_15.idx")


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
		"""
		if jitter_count:
			list_imgs =[]
			for i in range(test_batch["img"].shape[0]):
				print(count)
				count += 1
				list_imgs += dlib.jitter_image(np.uint8(test_batch["img"][i]), num_jitters=jitter_count, disturb_colors=True)
			imgs = np.array(list_imgs,dtype = np.float32)
		else:
			imgs = test_batch["img"]
		"""    
		imgs = test_batch["img"]    
		"""
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
		f_bin = open(bin_path,"ab")
		f_inx = open(idx_path,"a")
		for i,path in enumerate(test_batch["path_list"]):
			#save_path = os.path.join(target,path.split("/",2)[-1])
			save_path = path.split(data_dir)[-1].split("/",1)[-1]
			#print(path.split(data_dir)[-1] , data_dir)
			label = test_batch["label"][i]
			#print(save_path)
			write_to_bin(f_bin,f_inx,save_path,label,imgs[i])

			#if not os.path.exists(save_path.rsplit("/",1)[0]):
			#    os.makedirs(save_path.rsplit("/",1)[0])
			#try:
			#    img_s= Image.fromarray(np.uint8(imgs[i]))
			#    img_s.save(save_path)
			#except:
			#    print("Error: ",save_path)
			
		f_bin.close()
		f_inx.close()
				
				



		
	for i in range(thread_num):
		data_loader[i]._thread_stop=True
		data_loader[i].join()	
    

	
	
    
"""
if __name__ == "__main__":
    main()
"""