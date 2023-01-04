import matplotlib.pyplot as plt
import numpy as np
import math as mp
from PIL import Image


# paramètres : path vers l'image à traiter : path_image : (String)

IMG_PATH_1 = 'pns_original.jpg'
IMG_PATH_2 = 'the-legend.jpg'

def checkExtension(path_image):
    
    path = path_image.split("/")
    name_image = path[-1].split(".")[0]
    extension = path[-1].split(".")[-1]
    path.pop(-1)
    path_folder = "/".join(path)
    

    return extension , path_folder, name_image


def getImage(path_image):
    
    res = checkExtension(path_image)
    print(res)
    print(res[1]+"/"+res[2]+".jpg")
    if(res[0] != "jpg"  or res[0] != "JPG"):
        im_bad_ext = Image.open(path_image)
        im_bad_ext.convert('RGB')
        good_img = im_bad_ext.save(res[2]+'.jpg')
    return plt.imread(res[2]+'.jpg')



def tronquerImage(path_image):
    
    
    img_matrix = getImage(path_image)
    dim = np.shape(img_matrix)
    width = dim[0]
    height = dim[1]
    
    new_width = width - (width%8)
    new_height = height - (height%8)
    
    for i in range(width-1, new_width-1,-1):
        img_matrix = np.delete(img_matrix,i,0)
    
    for j in range(height-1, new_height-1,-1):
        img_matrix = np.delete(img_matrix,j,1)
        
    
    new_matrix_2D = [[0]*new_height]*new_width
    for i in range(0,new_width):
        for j in range(0,new_height):

            new_matrix_2D[i][j] = img_matrix[i][j][0]
         
        
    return new_matrix_2D
    

test = tronquerImage(IMG_PATH_2)

def centrageSurOctet(matrix_img):
    dim = np.shape(matrix_img)
    new_matrix_2D = [[0]*dim[1]]*dim[0]
    for i in range(0,dim[0]):
        for j in range(0,dim[1]):
          
            new_matrix_2D[i][j] = matrix_img[i][j] - 128
    
    return new_matrix_2D

         
f = centrageSurOctet(test)
print(f)













