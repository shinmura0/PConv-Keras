import os
import pathlib
import cv2
import numpy as np
import json

src_folder=[
    'D:/food-dataset-src/dataset100/UECFOOD100',
    'D:/food-dataset-src/dataset256/UECFOOD256',
    'D:/food-dataset-src/Food-11',
    'D:/food-dataset-src/food-101/images',
    'D:/food-dataset-src/pixfood20/release/images'
]
tgt_training_folder=[
    'D:/food-dataset/training/dataset100',
    'D:/food-dataset/training/dataset256',
    'D:/food-dataset/training/Food-11',
    'D:/food-dataset/training/food-101',
    'D:/food-dataset/training/pixfood20'
]
tgt_validation_folder= 'D:/food-dataset/validation/pixfood20'
tgt_evalutation_folder= 'D:/food-dataset/evaluation/pixfood20'

max_height = 256
max_width = 256
img_cnt=0

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

for i in range(len(src_folder)):
    if i==4: #pixfood, special sepration
        json_path='D:/food-dataset-src/pixfood20/release/images.json'
        img_folder_prefix='D:/food-dataset-src/pixfood20/release/images'
        lines = [line.rstrip('\n') for line in open(json_path,mode="r", encoding="utf-8")]
        for line in lines:
            tmp =  json.loads(line)
            if('食物' in tmp['image_path']):
                img_f_path=img_folder_prefix+'/'+tmp['image_path']
                #print(img_f_path)
                img = cv_imread(img_f_path)
                height, width, channels = img.shape
                # print(height, width, channels)
                if (height < max_height or width < max_width):
                    print('skip this image:' + str(img_f_path))
                elif max_height < height or max_width < width:
                    # get scaling factor
                    scaling_factor = max_height / float(height)
                    if max_width / float(width) > scaling_factor:
                        scaling_factor = max_width / float(width)
                    # resize image
                    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor,
                                     interpolation=cv2.INTER_AREA)
                    sh_height, sh_width, sh_channels = img.shape
                    if (sh_height > 256):
                        crop_img = img[int(sh_height / 2) - 128:int(sh_height / 2) + 128, 0:256]
                    elif (sh_width > 256):
                        crop_img = img[0:256, int(sh_height / 2) - 128:int(sh_height / 2) + 128]
                    else:
                        crop_img = img
                    params = list()
                    params.append(cv2.IMWRITE_PNG_COMPRESSION)
                    params.append(8)
                    img_cnt += 1
                    img_file_name = tgt_training_folder[i] + "/img_" + "{0:07d}".format(img_cnt) + ".bmp"
                    cv2.imwrite(img_file_name, crop_img, params)
        pass
    elif i<=3:
        print(src_folder[i])
        d = src_folder[i]
        d_child=[os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
        for j in range(len(d_child)):
            #print(d_child[j])
            #if(j==0):
                onlyfiles = [filepath.absolute() for filepath in pathlib.Path(d_child[j]).glob('**/*')]
                for k in range(len(onlyfiles)):
                    #if(k==0):
                        #print(onlyfiles[k])
                        if(str(onlyfiles[k]).endswith('.jpg') or str(onlyfiles[k]).endswith('.jpeg') or
                          str(onlyfiles[k]).endswith('.png') or str(onlyfiles[k]).endswith('.bmp')):
                            pass
                        else:
                            print("skip this file:" + str(onlyfiles[k]))
                            continue

                        img = cv2.imread(str(onlyfiles[k]).replace('\\', '/'))
                        height, width, channels = img.shape
                        #print(height, width, channels)
                        if(height<max_height or width < max_width):
                            print('skip this image:'+str(onlyfiles[k]))
                        elif max_height < height or max_width < width:
                            # get scaling factor
                            scaling_factor = max_height / float(height)
                            if max_width / float(width) > scaling_factor:
                                scaling_factor = max_width / float(width)
                            # resize image
                            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor,
                                             interpolation=cv2.INTER_AREA)
                            sh_height, sh_width, sh_channels = img.shape
                            if(sh_height>256):
                                crop_img = img[int(sh_height/2)-128:int(sh_height/2)+128, 0:256]
                            elif(sh_width>256):
                                crop_img = img[0:256,int(sh_height/2)-128:int(sh_height/2)+128]
                            else:
                                crop_img = img
                            params = list()
                            params.append(cv2.IMWRITE_PNG_COMPRESSION)
                            params.append(8)
                            img_cnt+=1
                            img_file_name=tgt_training_folder[i]+"/img_"+"{0:07d}".format(img_cnt)+".bmp"
                            cv2.imwrite(img_file_name, crop_img, params)


