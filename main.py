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
         
    print(np.shape(new_matrix_2D))   
    return new_matrix_2D
    

def centrageSurOctet(matrix_img):
    dim = np.shape(matrix_img)
    new_matrix_2D = [[0]*dim[1]]*dim[0]
    for i in range(0,dim[0]):
        for j in range(0,dim[1]):
          
            new_matrix_2D[i][j] = matrix_img[i][j] - 128
    
    return new_matrix_2D


def matrixPassageOrtho(n):
    P = [[0]*n for i in range(n)]
    C0 = 1/(mp.sqrt(2))
    for i in range(0,n):
        for j in range(0,n):
            if (i == 0 ):
                P[i][j] = (1/2) * C0 * np.cos((2*j + 1) * i * mp.pi/16)
            else:
                P[i][j] = (1/2) * mp.cos((2*j + 1) * i * mp.pi/16)
    return np.array(P)


def extraireBloc(i,j, matrix_image):
    
    u = 0
    v = 0
    bloc = np.zeros((8,8))
    for k in range(i,i+8):
        for p in range (j,j+8):
            bloc[u,v] = matrix_image[k,p]
            v = v + 1
        u = u + 1
        v = 0
    return np.array(bloc)


def chgtBaseBloc(bloc,P, Pt):
    return np.dot(np.dot(P,bloc),Pt)

def ecrireBlocToMatrix(matrix_image_2D,bloc,i,j):
    u = 0
    v = 0
    for k in range(i,i+8):
        for p in range(j,j+8):
            matrix_image_2D[k][p] = bloc[u][v]
            v = v + 1
        u = u + 1
        v = 0
    

def DiscretCosineTransform(matrix_image_2D,P,Pt):
    
    dim = np.shape(matrix_image_2D)
    width = dim[0]
    height = dim[1]
    
    new_matrix_dct = [[0]*height]*width

    for i in range(0,16,8):
        for j in range(0,16,8):
            bloc = extraireBloc(i,j,matrix_image_2D)
            bloc_dct = chgtBaseBloc(bloc,P,Pt)
            ecrireBlocToMatrix(new_matrix_dct,bloc_dct,i,j)
        
    return np.array(new_matrix_dct)
      
      

img1 = tronquerImage(IMG_PATH_1) 
img2 = centrageSurOctet(img1)

test = np.zeros((600,1200))
for i in range(0,600):
    for j in range(0,1200):
        test[i][j] = img2[i][j]

P = matrixPassageOrtho(8)
Pt = P.transpose()

test1 = DiscretCosineTransform(test,P,Pt)
print(test1.astype(int))

#print(t.astype(int))
plt.imsave('./test.jpg',np.array(test1))
# test = np.zeros((16,16))
# bloc = 2*np.ones((8,8))
# ecrireBlocToMatrix(test,bloc,0,0)
# ecrireBlocToMatrix(test,bloc,0,8)
# ecrireBlocToMatrix(test,bloc,8,8)
# ecrireBlocToMatrix(test,bloc,8,0)
# test1 = DiscretCosineTransform(test,bloc,bloc)
# print(np.array(test1))


        
    
    
    
    




                
            
            













