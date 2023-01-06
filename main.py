import matplotlib.pyplot as plt
import numpy as np
import math as mp
from PIL import Image
import time



IMG_PATH_1 = 'pns_original.jpg'
IMG_PATH_2 = 'the-legend.jpg'
IMG_PATH_3 = '491050.jpg'
MATRIX_Q = np.array([[16.0,11.0,10.0,16.0,24.0,40.0,51.0,61.0],
                    [12.0,12.0,13.0,19.0,26.0,58.0,60.0,55.0],
                    [14.0,13.0,16.0,24.0,40.0,57.0,69.0,56.0],
                    [14.0,17.0,22.0,29.0,51.0,87.0,80.0,62.0],
                    [18.0,22.0,37.0,56.0,68.0,109.0,103.0,77.0],
                    [24.0,35.0,55.0,64.0,81.0,104.0,113.0,92.0],
                    [49.0,64.0,78.0,87.0,103.0,121.0,120.0,101.0],
                    [72.0,92.0,95.0,98.0,112.0,100.0,103.0,99.0]])






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
    return plt.imread(res[2]+'.jpg') # renvoie un tableau de tableau de pixels




#####################################################################
#                                                                   #
#                             TRONCAGE                              #
#                                                                   #
#####################################################################





def tronquerImage(path_image, canal):
    
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
        
    
    new_matrix_2D = np.array([[0.0]*new_height]*new_width)
    for i in range(0,new_width):
        for j in range(0,new_height):

            new_matrix_2D[i,j] = img_matrix[i,j][canal]
         
    return new_matrix_2D
    



#####################################################################
#                                                                   #
#                        CENTRAGE SUR OCTET                         #
#                                                                   #
#####################################################################



def centrageSurOctet(path_image,canal):
    
    matrix_img = tronquerImage(path_image,canal)
    
    dim = np.shape(matrix_img)
    
    new_matrix_2D = np.array([[0.0]*dim[1]]*dim[0])
    
    for i in range(0,dim[0]):
        
        for j in range(0,dim[1]):
          
            new_matrix_2D[i,j] = matrix_img[i,j] - 128.0
    
    return np.array(new_matrix_2D)



#####################################################################
#                                                                   #
#                        MATRICE ORTHOGONALE                        #
#                                                                   #
#####################################################################




def matrixPassageOrtho(n):
    
    P = np.array([[0.0]*n]*n)
    C0 = 1/(mp.sqrt(2))
    for i in range(0,n):
        for j in range(0,n):
            if (i == 0 ):
                P[i,j] = (1.0/2.0) * C0 * np.cos((2.0*float(j) + 1.0) * float(i) * mp.pi/16.0)
            else:
                P[i,j] = (1.0/2.0) * mp.cos((2.0*float(j) + 1.0) * float(i) * mp.pi/16.0)
    return P



def extraireBloc(i,j, matrix_image):
    return matrix_image[i:i+8,j:j+8]


def chgtBaseBloc(bloc,P, Pt):
    return (np.dot(np.dot(P,bloc),Pt))


def ecrireBloc(matrix_image_2D,bloc,i,j):
    matrix_image_2D[i:i+8,j:j+8] = bloc


#####################################################################
#                                                                   #
#                         DCTII                                     #
#                                                                   #
#####################################################################





def DCTII(array_img,P,Pt):
    
    dim = np.shape(array_img)
    width = dim[0]
    height = dim[1]
    
    img_DCT_II = np.array([[0.0]*height]*width)

    for i in range(0,width,8):
        
        for j in range(0,height,8):
            
            M = extraireBloc(i,j,array_img)
            D = chgtBaseBloc(M,P,Pt)
            ecrireBloc(img_DCT_II,D,i,j)
        
    return img_DCT_II
      
      
      
      
#####################################################################
#                                                                   #
#                                 FILTRE                            #
#                                                                   #
#####################################################################



def Filtre(bloc, filtre):
    
    new_bloc = np.array([[0.0]*8]*8)
    
    for i in range(8):
        for j in range(8):
            if((i+j) >= filtre):
                new_bloc[i,j] = 0.0
            else:
                new_bloc[i,j] = bloc[i,j]
                
    return new_bloc





#####################################################################
#                                                                   #
#                         COMPRESSION                               #
#                                                                   #
#####################################################################



def Compression(matrix_image_2D_1,filtre,done):
    
    dim = np.shape(matrix_image_2D_1)
    
    width = dim[0]
    
    height = dim[1]
    
    new_matrix_dct = np.array([[0.0]*height]*width)

    for i in range(0,width,8):
        
        for j in range(0,height,8):
            
            if(done == True):
                
                bloc = extraireBloc(i,j,matrix_image_2D_1)
                bloc_divide = np.floor(np.divide(bloc,MATRIX_Q))
                ecrireBloc(new_matrix_dct,bloc_compress,i,j)
                
            else:
                
                bloc = extraireBloc(i,j,matrix_image_2D_1)
                bloc_divide = np.floor(np.divide(bloc,MATRIX_Q))
                bloc_compress = Filtre(bloc_divide,filtre)
                ecrireBloc(new_matrix_dct,bloc_compress,i,j)
        
    return new_matrix_dct





#####################################################################
#                                                                   #
#                         DECOMPRESSION                             #
#                                                                   #
#####################################################################



def Decompression(matrix_image_2D_2,P ,Pt):
    
    dim = np.shape(matrix_image_2D_2)
    width = dim[0]
    height = dim[1]
    
    new_matrix_dct = np.array([[0.0]*height]*width)
    
    for i in range(0,width,8):
        for j in range(0,height,8):
            
            bloc = extraireBloc(i,j,matrix_image_2D_2)
            bloc_Q = np.multiply(bloc,MATRIX_Q)
            bloc_recup = np.dot(np.dot(Pt,bloc_Q),P) + 128.0
            ecrireBloc(new_matrix_dct,bloc_recup,i,j)

    return new_matrix_dct


            
#####################################################################
#                                                                   #
#                   MAIN LANCEMENT DU PROGRAMME                     #
#                                                                   #
#####################################################################







img = getImage(IMG_PATH_3)
img1 = centrageSurOctet(IMG_PATH_3,0) #canal R
img2 = centrageSurOctet(IMG_PATH_3,1) #canal G
img3 = centrageSurOctet(IMG_PATH_3,2) #canal B

P = matrixPassageOrtho(8)
Pt = P.transpose()

start_time = time.time()


res0 = DCTII(img1,P,Pt)
res1 = Compression(res0,6,False)
res2 = Decompression(res1,P,Pt)

res3 = DCTII(img2,P,Pt)
res4 = Compression(res3,6,False)
res5 = Decompression(res4,P,Pt)

res6 = DCTII(img3,P,Pt)
res7 = Compression(res6,6,False)
res8 = Decompression(res7,P,Pt)






#####################################################################
#                                                                   #
#        ECRITURE ET AFFCIHAGE DE L'IMAGE APRES COMPRESSION         #
#                                                                   #
#####################################################################



image = np.empty((2400,3840,3)).astype(int)


for i in range(2400):
    for j in range(3840):
        image[i,j,0] = res2[i,j]
        image[i,j,1] = res5[i,j]
        image[i,j,2] = res8[i,j]

im = plt.imshow(image)



print("--- %s seconds ---" % (time.time() - start_time))
plt.show()








        
    
    
    
    




                
            
            













