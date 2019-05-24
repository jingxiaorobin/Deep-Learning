import os,sys
import random
import shutil
 
 
def copyFile(fileDir):
    pathDir = os.listdir(fileDir)
    sample = random.sample(pathDir, 200)
    #print(sample)
    for name in sample:
        shutil.move(fileDir + name, tarDir + name)
 
 
if __name__ == '__main__':
    # open /textiles
    path = "/home/fairy/workspace/dataset/textiles/"
    dirs = os.listdir(path)
    i = 0
    # output all folds
    for file in dirs:
        print(file)
        i = i+1
        filename = "/home/fairy/workspace/dataset/Fabric" + str(i)
        os.mkdir(filename)
        fileDir = path + "Fabric"+str(i) + "/"
        tarDir = filename + "/"
        copyFile(fileDir)
