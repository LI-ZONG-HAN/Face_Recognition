# Make images data to binary files for training
## Why
The older way i used is that load separate images in HD in each batch.It takes a long time at open each image files. Therefor i packed all image files in one big binary file and the position for each image in the bin file is recorded in the idx file. 
## How
### step 1
The program will expect to be given a directory structured as follows:


Case1 :Only for this FR case, with images files list table
```
root floder/
   image_floder/
           label1/            
               image1.jpg
               image2.jpg            
               image3.jpg        
           label2/            
               image4.jpg            
               image5.jpg            
               image6.jpg        
           label3/            
               image7.jpg            
               image8.jpg            
               image9.jpg
           
   FR_images_list_table
```

Case2 : Without images files list table (Non_FR data)
```
root floder/
   image_floder/
           label1/            
               image1.jpg
               image2.jpg            
               image3.jpg        
           label2/            
               image4.jpg            
               image5.jpg            
               image6.jpg        
           label3/            
               image7.jpg            
               image8.jpg            
               image9.jpg
```
### step 2 : Run Make_data_to_bin.py
pls type
```
python Make_data_to_bin.py True -dir root_floder_path -f image_floder or FR_images_list_table -out_name bin_file_name
```
First argument True mean case1 and False mean case2

After program complete, two files bin and idx will be created in root_floder

bin : image data

idx : There four columns, original_file_path/label/start position in bin/end position in bin 

### recovery bin to images

pls type
```
python bin_to_image.py -in_dir bin_floder_path -bin_name bin_file_name -out_dir floder_to_save_images
```


