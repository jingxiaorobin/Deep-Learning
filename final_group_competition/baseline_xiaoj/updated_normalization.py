import os
import cv2
import numpy as np
 
 

def compute(path):
    file_names = os.listdir(path)
    #per_image_Rmean = []
    #per_image_Gmean = []
    #per_image_Bmean = []
    #m_list, s_list = [], []
    
    for file_name in file_names:
        classify = os.listdir(path + '/' +file_name)
        for i in classify:
            img = cv2.imread(os.path.join(path,file_name ,i))
            img = img / 255.0
            #print(img,os.path.join(path,i))
            m, s = cv2.meanStdDev(img)
            m_list.append(m.reshape((3,)))
            s_list.append(s.reshape((3,)))
    return m_list, s_list

            
 
if __name__ == '__main__':
    m_list, s_list = [], []
    path = ['/Users/xiaojing/Documents/Jingxiao/DL/final_project/ssl_data_96/supervised/train',
            '/Users/xiaojing/Documents/Jingxiao/DL/final_project/ssl_data_96/supervised/val',
            '/Users/xiaojing/Documents/Jingxiao/DL/final_project/ssl_data_96/unsupervised']
    for  p in path:
        m_list, s_list = compute(p)
        print(p)
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print(m[0][::-1])
    print(s[0][::-1])