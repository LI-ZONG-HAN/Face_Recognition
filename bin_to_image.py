from PIL import Image
import os
import numpy as np
import io



	


def load_index_lists(data_dir,flie_name):
    with open(os.path.join(data_dir,flie_name),"r") as f:
        lines = f.readlines()
        lines = [f.strip() for f in lines]
        labels = np.array([f.strip().split("\t")[1] for f in lines],dtype = np.uint64)
        path_list = np.array([f.strip().split("\t",1)[0] for f in lines])
        index_list = np.array([f.strip().split("\t")[2:] for f in lines],dtype = np.uint64)
        hush_table = {}
        path_list_by_id = []
        index_list_by_id =[]
        keys_list_temp=[]
        for i,l in enumerate(labels):
            if l not in hush_table.keys():
                hush_table[l] = [i]
                keys_list_temp.append(l)
            else:
                hush_table[l] += [i]

        for k in keys_list_temp:
            path_list_by_id.append(path_list[hush_table[k]])
            index_list_by_id.append(index_list[hush_table[k]])
        
        del hush_table
    return path_list_by_id, index_list_by_id, path_list, index_list, labels


    
        
def main():
    
    train_sets = ["FR_asian_valid","FR_west_valid"]
    label_shift = 0
    FR_f_bin_path_train = []
    FR_indexs_train = []
    FR_paths = []
    FR_data_dir = "training"
    for name in train_sets:
        FR_file_name = name+".idx"
        print(name)
        FR_f_bin_path_train.append(os.path.join(FR_data_dir,name+".bin"))
        _,_,paths_temp,indexs_temp,_ = load_index_lists(FR_data_dir,FR_file_name)
        FR_indexs_train.append(indexs_temp)
        FR_paths.append(paths_temp)
		
    target_dir = "training/temp"
    count = 0
    for i, paths in enumerate(FR_paths):
        f_bin = open(FR_f_bin_path_train[i],"rb")
        for j, path in enumerate(paths):
            count += 1 
            if count%1000 == 0:
                print(count)
            floder = os.path.join(target_dir,path.rsplit("/",1)[0])
            if not os.path.exists(floder):
                print(floder)
                os.makedirs(floder)
            FR_indexs_train[i][j]
            read_start = FR_indexs_train[i][j][0]
            read_end = FR_indexs_train[i][j][1]
            f_bin.seek(read_start)
            data = f_bin.read(read_end - read_start)
            #img = Image.open(io.BytesIO(data))
            _io = io.BytesIO(data)
            with open(os.path.join(target_dir,path),'wb') as out: ## Open temporary file as bytes
                out.write(data)
        f_bin.close()
     
if __name__ == "__main__":
    main()              
