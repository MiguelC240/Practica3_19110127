import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import cv2
import math

import numpy as np
import skimage
from skimage import io



Img1 = cv2.imread('Imagen_Dia.jpg')
Img2 = cv2.imread('Imagen_Noche.jpg')
Img3 = cv2.imread('Imagen_Noche.jpg',0)


res1 = cv2.resize(Img1, dsize=(280, 280))
res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2RGB)

res2 = cv2.resize(Img2, dsize=(280, 280))
res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2RGB)
Img_Negro = cv2.resize(Img3, dsize=(380, 380))


color = ('b','g','r')

#Figura_Completa, ax = plt.subplots(4, 3)
#Figura_Completa.set_size_inches(12, 42)

####################################################################
#                               SUMA 1                             #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')


#Suma
Suma = res1 + res2



#Imagen SUMA
ax[0, 1].imshow(Suma)
ax[0, 1].set_title('Suma')
ax[0, 1].axis('off')


#Histograma Imagen SUMA sin ecualizar

for i, c in enumerate(color):
    hist_suma = cv2.calcHist([Suma], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_suma, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Suma')
ax[1, 1].axis('off')


#Ecualizacion imagen SUMA
Ecua_suma = cv2.cvtColor(Suma,cv2.COLOR_BGR2YUV)
Ecua_suma[:,:,0] = cv2.equalizeHist(Ecua_suma[:,:,0])
Ecualizacion_suma = cv2.cvtColor(Ecua_suma,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_suma)
ax[2, 1].set_title('Imagen suma Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen SUMA ecualizada

for i, c in enumerate(color):
    hist_ecua_suma = cv2.calcHist([Ecualizacion_suma], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_suma, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Suma Ecua')
ax[3, 1].axis('off')


plt.show()



####################################################################
#                               SUMA 2                             #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#Suma 2
Adicion = cv2.add(res1,res2)



#Imagen SUMA
ax[0, 1].imshow(Adicion)
ax[0, 1].set_title('Adicion')
ax[0, 1].axis('off')


#Histograma Imagen SUMA sin ecualizar

for i, c in enumerate(color):
    hist_Adicion = cv2.calcHist([Adicion], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Adicion, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Suma')
ax[1, 1].axis('off')


#Ecualizacion imagen SUMA
Ecua_Adicion = cv2.cvtColor(Adicion,cv2.COLOR_BGR2YUV)
Ecua_Adicion[:,:,0] = cv2.equalizeHist(Ecua_Adicion[:,:,0])
Ecualizacion_Adicion = cv2.cvtColor(Ecua_Adicion,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Adicion)
ax[2, 1].set_title('Imagen suma Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen SUMA ecualizada

for i, c in enumerate(color):
    hist_ecua_Adicion = cv2.calcHist([Ecualizacion_Adicion], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Adicion, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Suma Ecua')
ax[3, 1].axis('off')




plt.show()


####################################################################
#                               SUMA 3                             #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')


#Suma 3
Adi = cv2.addWeighted(res1,0.5,res2,0.5,0)



#Imagen SUMA
ax[0, 1].imshow(Adi)
ax[0, 1].set_title('Adicion')
ax[0, 1].axis('off')


#Histograma Imagen SUMA sin ecualizar

for i, c in enumerate(color):
    hist_Adi = cv2.calcHist([Adi], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Adi, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Suma')
ax[1, 1].axis('off')


#Ecualizacion imagen SUMA
Ecua_Adi = cv2.cvtColor(Adi,cv2.COLOR_BGR2YUV)
Ecua_Adi[:,:,0] = cv2.equalizeHist(Ecua_Adi[:,:,0])
Ecualizacion_Adi = cv2.cvtColor(Ecua_Adi,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Adi)
ax[2, 1].set_title('Imagen suma Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen SUMA ecualizada

for i, c in enumerate(color):
    hist_ecua_Adi = cv2.calcHist([Ecualizacion_Adi], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Adi, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Suma Ecua')
ax[3, 1].axis('off')




plt.show()



####################################################################
#                               RESTA 1                            #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')


#Resta 1
Resta = res1 - res2


#Imagen RESTA
ax[0, 1].imshow(Resta)
ax[0, 1].set_title('Resta')
ax[0, 1].axis('off')


#Histograma Imagen RESTA sin ecualizar

for i, c in enumerate(color):
    hist_Resta = cv2.calcHist([Resta], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Resta, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Resta')
ax[1, 1].axis('off')


#Ecualizacion imagen RESTA
Ecua_Resta = cv2.cvtColor(Resta,cv2.COLOR_BGR2YUV)
Ecua_Resta[:,:,0] = cv2.equalizeHist(Ecua_Resta[:,:,0])
Ecualizacion_Resta = cv2.cvtColor(Ecua_Resta,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Resta)
ax[2, 1].set_title('Imagen Resta Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen RESTA ecualizada

for i, c in enumerate(color):
    hist_ecua_Resta = cv2.calcHist([Ecualizacion_Resta], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Resta, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Resta Ecua')
ax[3, 1].axis('off')



plt.show()





####################################################################
#                               RESTA 2                            #
####################################################################


Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#Resta 2
Sustraccion = cv2.subtract(res1,res2)



#Imagen RESTA
ax[0, 1].imshow(Sustraccion)
ax[0, 1].set_title('Sustraccion')
ax[0, 1].axis('off')


#Histograma Imagen RESTA sin ecualizar

for i, c in enumerate(color):
    hist_Sustraccion = cv2.calcHist([Sustraccion], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Sustraccion, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Resta')
ax[1, 1].axis('off')


#Ecualizacion imagen RESTA
Ecua_Sustraccion = cv2.cvtColor(Sustraccion,cv2.COLOR_BGR2YUV)
Ecua_Sustraccion[:,:,0] = cv2.equalizeHist(Ecua_Sustraccion[:,:,0])
Ecualizacion_Sustraccion = cv2.cvtColor(Ecua_Sustraccion,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Sustraccion)
ax[2, 1].set_title('Imagen Resta Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen RESTA ecualizada

for i, c in enumerate(color):
    hist_ecua_Sustraccion = cv2.calcHist([Ecualizacion_Sustraccion], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Sustraccion, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Resta Ecua')
ax[3, 1].axis('off')


plt.show()





####################################################################
#                               RESTA 3                            #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#Resta 3
Absdiff = cv2.absdiff(res1,res2)



#Imagen RESTA
ax[0, 1].imshow(Absdiff)
ax[0, 1].set_title('Sustraccion')
ax[0, 1].axis('off')


#Histograma Imagen RESTA sin ecualizar

for i, c in enumerate(color):
    hist_Absdiff = cv2.calcHist([Absdiff], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Absdiff, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Resta')
ax[1, 1].axis('off')


#Ecualizacion imagen RESTA
Ecua_Absdiff = cv2.cvtColor(Absdiff,cv2.COLOR_BGR2YUV)
Ecua_Absdiff[:,:,0] = cv2.equalizeHist(Ecua_Absdiff[:,:,0])
Ecualizacion_Absdiff = cv2.cvtColor(Ecua_Absdiff,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Absdiff)
ax[2, 1].set_title('Imagen Resta Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen RESTA ecualizada

for i, c in enumerate(color):
    hist_ecua_Absdiff = cv2.calcHist([Ecualizacion_Absdiff], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Absdiff, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Resta Ecua')
ax[3, 1].axis('off')




plt.show()






####################################################################
#                      MULTIPLICACION 1                            #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#Multiplicaicion 1
Multiplicacion = res1 * res2



#Imagen MULTIPLICACION
ax[0, 1].imshow(Multiplicacion)
ax[0, 1].set_title('Multiplicacion')
ax[0, 1].axis('off')


#Histograma Imagen MULTIPLICACION sin ecualizar

for i, c in enumerate(color):
    hist_Multiplicacion = cv2.calcHist([Multiplicacion], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Multiplicacion, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Multiplicacion')
ax[1, 1].axis('off')


#Ecualizacion imagen MULTIPLICACION
Ecua_Multiplicacion = cv2.cvtColor(Multiplicacion,cv2.COLOR_BGR2YUV)
Ecua_Multiplicacion[:,:,0] = cv2.equalizeHist(Ecua_Multiplicacion[:,:,0])
Ecualizacion_Multiplicacion = cv2.cvtColor(Ecua_Multiplicacion,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Multiplicacion)
ax[2, 1].set_title('Imagen Multiplicacion Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen MULTIPLICACION ecualizada

for i, c in enumerate(color):
    hist_ecua_Multiplicacion = cv2.calcHist([Ecualizacion_Multiplicacion], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Multiplicacion, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Multiplicacion Ecua')
ax[3, 1].axis('off')



plt.show()



####################################################################
#                      MULTIPLICACION 2                            #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#Multiplicaicion 2
Multiply = cv2.multiply(res1,res2)


#Imagen MULTIPLICACION
ax[0, 1].imshow(Multiply)
ax[0, 1].set_title('Multiply')
ax[0, 1].axis('off')


#Histograma Imagen MULTIPLICACION sin ecualizar

for i, c in enumerate(color):
    hist_Multiply = cv2.calcHist([Multiply], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Multiply, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Multiply')
ax[1, 1].axis('off')


#Ecualizacion imagen MULTIPLICACION
Ecua_Multiply = cv2.cvtColor(Multiply,cv2.COLOR_BGR2YUV)
Ecua_Multiply[:,:,0] = cv2.equalizeHist(Ecua_Multiply[:,:,0])
Ecualizacion_Multiply = cv2.cvtColor(Ecua_Multiply,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Multiply)
ax[2, 1].set_title('Imagen Multiply Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen MULTIPLICACION ecualizada

for i, c in enumerate(color):
    hist_ecua_Multiply = cv2.calcHist([Ecualizacion_Multiply], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Multiply, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Multiply Ecua')
ax[3, 1].axis('off')




plt.show()




####################################################################
#                            DIVISION 1                            #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#DIVISION 1
Division = cv2.divide(res1,res2)


#Imagen DIVISION
ax[0, 1].imshow(Division)
ax[0, 1].set_title('Division')
ax[0, 1].axis('off')


#Histograma Imagen DIVISION sin ecualizar

for i, c in enumerate(color):
    hist_Division = cv2.calcHist([Division], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Division, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Division')
ax[1, 1].axis('off')


#Ecualizacion imagen DIVISION
Ecua_Division = cv2.cvtColor(Division,cv2.COLOR_BGR2YUV)
Ecua_Division[:,:,0] = cv2.equalizeHist(Ecua_Division[:,:,0])
Ecualizacion_Division= cv2.cvtColor(Ecua_Division,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Division)
ax[2, 1].set_title('Imagen Division Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen DIVISION ecualizada

for i, c in enumerate(color):
    hist_ecua_Division = cv2.calcHist([Ecualizacion_Division], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Division, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Division Ecua')
ax[3, 1].axis('off')


plt.show()



####################################################################
#                            LOGARITMO 1                           #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#LOGARITMO 1
Logaritmo = np.zeros(res1.shape, res1.dtype)
c = 1
Logaritmo = c * np.log(1+res1)
maxi = np.amax(Logaritmo)
Logaritmo = np.uint8(Logaritmo / maxi *255)


#Imagen LOGARITMO
ax[0, 1].imshow(Logaritmo)
ax[0, 1].set_title('Logaritmo')
ax[0, 1].axis('off')


#Histograma Imagen LOGARITMO sin ecualizar

for i, c in enumerate(color):
    hist_Logaritmo = cv2.calcHist([Logaritmo], [1], None, [256], [0, 256])
    ax[1, 1].plot(hist_Logaritmo, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Logaritmo')
ax[1, 1].axis('off')


#Ecualizacion imagen LOGARITMO
Ecua_Logaritmo = cv2.cvtColor(Logaritmo,cv2.COLOR_BGR2YUV)
Ecua_Logaritmo[:,:,0] = cv2.equalizeHist(Ecua_Logaritmo[:,:,0])
Ecualizacion_Logaritmo = cv2.cvtColor(Ecua_Logaritmo,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Logaritmo)
ax[2, 1].set_title('Imagen Logaritmo Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen LOGARITMO ecualizada

for i, c in enumerate(color):
    hist_ecua_Logaritmo = cv2.calcHist([Ecualizacion_Logaritmo], [1], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Logaritmo, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Logaritmo Ecua')
ax[3, 1].axis('off')


plt.show()



####################################################################
#                            LOGARITMO 2                           #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#LOGARITMO 2
Logaritmo = np.zeros(res2.shape, res2.dtype)
c = 1
Logaritmo = c * np.log(1+res2)
maxi = np.amax(Logaritmo)
Logaritmo = np.uint8(Logaritmo / maxi *255)


#Imagen LOGARITMO
ax[0, 1].imshow(Logaritmo)
ax[0, 1].set_title('Logaritmo')
ax[0, 1].axis('off')


#Histograma Imagen LOGARITMO sin ecualizar

for i, c in enumerate(color):
    hist_Logaritmo = cv2.calcHist([Logaritmo], [1], None, [256], [0, 256])
    ax[1, 1].plot(hist_Logaritmo, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Logaritmo')
ax[1, 1].axis('off')


#Ecualizacion imagen LOGARITMO
Ecua_Logaritmo = cv2.cvtColor(Logaritmo,cv2.COLOR_BGR2YUV)
Ecua_Logaritmo[:,:,0] = cv2.equalizeHist(Ecua_Logaritmo[:,:,0])
Ecualizacion_Logaritmo = cv2.cvtColor(Ecua_Logaritmo,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Logaritmo)
ax[2, 1].set_title('Imagen Logaritmo Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen LOGARITMO ecualizada

for i, c in enumerate(color):
    hist_ecua_Logaritmo = cv2.calcHist([Ecualizacion_Logaritmo], [1], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Logaritmo, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Logaritmo Ecua')
ax[3, 1].axis('off')

plt.show()




####################################################################
#                               RAIZ                               #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')




#RAIZ
Raiz = (res1**(0.5))
Raiz_m = np.float32(Raiz)
cv2.imwrite('raiz.jpg',Raiz)
Raiz_g = cv2.imread('raiz.jpg',0)

#Imagen RAIZ
ax[0, 1].imshow(Raiz)
ax[0, 1].set_title('Raiz')
ax[0, 1].axis('off')


#Histograma Imagen RAIZ sin ecualizar

for i, c in enumerate(color):
    hist_Raiz = cv2.calcHist([Raiz_m], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Raiz, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Raiz')
ax[1, 1].axis('off')


#Ecualizacion imagen RAIZ
Ecua_Raiz = cv2.equalizeHist(Raiz_g)


ax[2, 1].imshow(Ecua_Raiz)
ax[2, 1].set_title('Imagen Raiz Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen RAIZ ecualizada

for i, c in enumerate(color):
    hist_ecua_Raiz = cv2.calcHist([Ecua_Raiz], [0], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Raiz, color = 'gray')
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Raiz Ecua')
ax[3, 1].axis('off')


plt.show()



####################################################################
#                           DERIVADA                               #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')


#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')




#DERIVADA
Derivada = cv2.Laplacian(res1,cv2.CV_32F)
Derivada_m = np.float32(Derivada)
cv2.imwrite('Derivada.jpg',Derivada)
Derivada_g = cv2.imread('Derivada.jpg',0)

#Imagen DERIVADA
ax[0, 1].imshow(Derivada)
ax[0, 1].set_title('Derivada')
ax[0, 1].axis('off')


#Histograma Imagen DERIVADA sin ecualizar

for i, c in enumerate(color):
    hist_Derivada = cv2.calcHist([Derivada_m], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Derivada, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Derivada')
ax[1, 1].axis('off')


#Ecualizacion imagen DERIVADA
Ecua_Derivada = cv2.equalizeHist(Derivada_g)
#cv2.imshow('Derivada',Derivada_g)
cv2.imshow('Derivada2',Ecua_Derivada)

#ax[2, 1].imshow(Derivada_g)
ax[2, 1].imshow(Ecua_Derivada)
ax[2, 1].set_title('Imagen Derivada Ecua')
ax[2, 1].axis('off')
cv2.waitKey(0)


#Histograma Imagen DERIVADA ecualizada

for i, c in enumerate(color):
    hist_ecua_Derivada = cv2.calcHist([Ecua_Derivada], [0], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Derivada, color = 'gray')
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Derivada Ecua')
ax[3, 1].axis('off')


plt.show()






####################################################################
#                             POTENCIA 1                           #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#POTENCIA
Potencia = (res1**2)

#Imagen POTENCIA
ax[0, 1].imshow(Potencia)
ax[0, 1].set_title('Potencia')
ax[0, 1].axis('off')


#Histograma Imagen POTENCIA sin ecualizar

for i, c in enumerate(color):
    hist_Potencia = cv2.calcHist([Potencia], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Potencia, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Potencia')
ax[1, 1].axis('off')


#Ecualizacion imagen POTENCIA
Ecua_Potencia= cv2.cvtColor(Potencia,cv2.COLOR_BGR2YUV)
Ecua_Potencia[:,:,0] = cv2.equalizeHist(Ecua_Potencia[:,:,0])
Ecualizacion_Potencia = cv2.cvtColor(Ecua_Potencia,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Potencia)
ax[2, 1].set_title('Imagen Potencia Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen POTENCIA ecualizada

for i, c in enumerate(color):
    hist_ecua_Potencia = cv2.calcHist([Ecualizacion_Potencia], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Potencia, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Potencia Ecua')
ax[3, 1].axis('off')




plt.show()


####################################################################
#                             POTENCIA                             #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#POTENCIA
Potencia = cv2.pow(res1,2)

#Imagen POTENCIA
ax[0, 1].imshow(Potencia)
ax[0, 1].set_title('Potencia')
ax[0, 1].axis('off')


#Histograma Imagen POTENCIA sin ecualizar

for i, c in enumerate(color):
    hist_Potencia = cv2.calcHist([Potencia], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Potencia, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Potencia')
ax[1, 1].axis('off')


#Ecualizacion imagen POTENCIA
Ecua_Potencia= cv2.cvtColor(Potencia,cv2.COLOR_BGR2YUV)
Ecua_Potencia[:,:,0] = cv2.equalizeHist(Ecua_Potencia[:,:,0])
Ecualizacion_Potencia = cv2.cvtColor(Ecua_Potencia,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Potencia)
ax[2, 1].set_title('Imagen Potencia Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen POTENCIA ecualizada

for i, c in enumerate(color):
    hist_ecua_Potencia = cv2.calcHist([Ecualizacion_Potencia], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Potencia, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Potencia Ecua')
ax[3, 1].axis('off')



plt.show()


####################################################################
#                           CONJUNCION                             #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#CONJUNCION
Conjuncion = cv2.bitwise_and(res1,res2)

#Imagen CONJUNCION
ax[0, 1].imshow(Conjuncion)
ax[0, 1].set_title('Conjuncion')
ax[0, 1].axis('off')


#Histograma Imagen CONJUNCION sin ecualizar

for i, c in enumerate(color):
    hist_Conjuncion = cv2.calcHist([Conjuncion], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Conjuncion, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Conjuncion')
ax[1, 1].axis('off')


#Ecualizacion imagen CONJUNCION
Ecua_Conjuncion = cv2.cvtColor(Conjuncion,cv2.COLOR_BGR2YUV)
Ecua_Conjuncion[:,:,0] = cv2.equalizeHist(Ecua_Conjuncion[:,:,0])
Ecualizacion_Conjuncion = cv2.cvtColor(Ecua_Conjuncion,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Conjuncion)
ax[2, 1].set_title('Imagen Conjuncion Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen CONJUNCION ecualizada

for i, c in enumerate(color):
    hist_ecua_Conjuncion = cv2.calcHist([Ecualizacion_Conjuncion], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Conjuncion, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Conjuncion Ecua')
ax[3, 1].axis('off')


plt.show()




####################################################################
#                           DISYUNCION                             #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#DISYUNCION
Disyuncion = cv2.bitwise_or(res1,res2)

#Imagen DISYUNCION
ax[0, 1].imshow(Disyuncion)
ax[0, 1].set_title('Disyuncion')
ax[0, 1].axis('off')


#Histograma Imagen DISYUNCION sin ecualizar

for i, c in enumerate(color):
    hist_Disyuncion = cv2.calcHist([Disyuncion], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Disyuncion, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Disyuncion')
ax[1, 1].axis('off')


#Ecualizacion imagen DISYUNCION
Ecua_Disyuncion = cv2.cvtColor(Disyuncion,cv2.COLOR_BGR2YUV)
Ecua_Disyuncion[:,:,0] = cv2.equalizeHist(Ecua_Disyuncion[:,:,0])
Ecualizacion_Disyuncion = cv2.cvtColor(Ecua_Disyuncion,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Disyuncion)
ax[2, 1].set_title('Imagen Disyuncion Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen DISYUNCION ecualizada

for i, c in enumerate(color):
    hist_ecua_Disyuncion = cv2.calcHist([Ecualizacion_Disyuncion], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Disyuncion, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Disyuncion Ecua')
ax[3, 1].axis('off')


plt.show()



####################################################################
#                           NEGACION 1                             #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#NEGACION
Negacion = cv2.resize(Img1, dsize=(280, 280))
height, width, _ = Negacion.shape

for i in range(0, height - 1):
    for j in range(0, width -1):
        pixel = Negacion[i,j]
        pixel[0] = 255 - pixel[0]
        pixel[1] = 255 - pixel[1]
        pixel[2] = 255 - pixel[2]
        Negacion[i,j] = pixel

#Imagen NEGACION
ax[0, 1].imshow(Negacion)
ax[0, 1].set_title('Negacion')
ax[0, 1].axis('off')


#Histograma Imagen NEGACION sin ecualizar

for i, c in enumerate(color):
    hist_Negacion = cv2.calcHist([Negacion], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Negacion, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Negacion')
ax[1, 1].axis('off')


#Ecualizacion imagen NEGACION
Ecua_Negacion = cv2.cvtColor(Negacion,cv2.COLOR_BGR2YUV)
Ecua_Negacion[:,:,0] = cv2.equalizeHist(Ecua_Negacion[:,:,0])
Ecualizacion_Negacion = cv2.cvtColor(Ecua_Negacion,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Negacion)
ax[2, 1].set_title('Imagen Negacion Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen NEGACION ecualizada

for i, c in enumerate(color):
    hist_ecua_Negacion = cv2.calcHist([Ecualizacion_Negacion], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Negacion, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Negacion Ecua')
ax[3, 1].axis('off')


plt.show()


####################################################################
#                           NEGACION 2                             #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#NEGACION 2
Negacion2 = 1 - res1


#Imagen NEGACION
ax[0, 1].imshow(Negacion2)
ax[0, 1].set_title('Negacion 2')
ax[0, 1].axis('off')


#Histograma Imagen NEGACION sin ecualizar

for i, c in enumerate(color):
    hist_Negacion2 = cv2.calcHist([Negacion2], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Negacion2, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Negacion 2')
ax[1, 1].axis('off')


#Ecualizacion imagen NEGACION
Ecua_Negacion2 = cv2.cvtColor(Negacion2,cv2.COLOR_BGR2YUV)
Ecua_Negacion2[:,:,0] = cv2.equalizeHist(Ecua_Negacion2[:,:,0])
Ecualizacion_Negacion2 = cv2.cvtColor(Ecua_Negacion2,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Negacion2)
ax[2, 1].set_title('Imagen Negacion Ecualizada 2')
ax[2, 1].axis('off')



#Histograma Imagen NEGACION ecualizada

for i, c in enumerate(color):
    hist_ecua_Negacion2 = cv2.calcHist([Ecualizacion_Negacion2], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Negacion2, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Negacion Ecua 2')
ax[3, 1].axis('off')


plt.show()



####################################################################
#                           TRASLACIÓN                             #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#TRASLACIÓN
ancho = res1.shape[1] #columnas
alto = res1.shape[0] # filas
    
M = np.float32([[1,0,10],[0,1,100]]) #Construccion de la matriz
Traslacion = cv2.warpAffine(res1,M,(ancho,alto))


#Imagen TRASLACIÓN
ax[0, 1].imshow(Traslacion)
ax[0, 1].set_title('Traslacion')
ax[0, 1].axis('off')


#Histograma Imagen TRASLACIÓN sin ecualizar

for i, c in enumerate(color):
    hist_Traslacion = cv2.calcHist([Traslacion], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Traslacion, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Traslacion')
ax[1, 1].axis('off')


#Ecualizacion imagen TRASLACIÓN
Ecua_Traslacion = cv2.cvtColor(Traslacion,cv2.COLOR_BGR2YUV)
Ecua_Traslacion[:,:,0] = cv2.equalizeHist(Ecua_Traslacion[:,:,0])
Ecualizacion_Traslacion = cv2.cvtColor(Ecua_Traslacion,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Traslacion)
ax[2, 1].set_title('Imagen Traslacion Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen TRASLACIÓN ecualizada

for i, c in enumerate(color):
    hist_ecua_Traslacion = cv2.calcHist([Ecualizacion_Traslacion], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Traslacion, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Traslacion Ecua')
ax[3, 1].axis('off')


plt.show()



####################################################################
#                              ESCALADO                            #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#ESCALADO
Escalado = cv2.resize(res1, dsize=(480, 480))


#Imagen ESCALADO
ax[0, 1].imshow(Escalado)
ax[0, 1].set_title('Escalado')
ax[0, 1].axis('off')


#Histograma Imagen ESCALADO sin ecualizar

for i, c in enumerate(color):
    hist_Escalado = cv2.calcHist([Escalado], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Escalado, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Escalado')
ax[1, 1].axis('off')


#Ecualizacion imagen ESCALADO
Ecua_Escalado = cv2.cvtColor(Escalado,cv2.COLOR_BGR2YUV)
Ecua_Escalado[:,:,0] = cv2.equalizeHist(Ecua_Escalado[:,:,0])
Ecualizacion_Escalado = cv2.cvtColor(Ecua_Escalado,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Escalado)
ax[2, 1].set_title('Imagen Escalado Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen ESCALADO ecualizada

for i, c in enumerate(color):
    hist_ecua_Escalado = cv2.calcHist([Ecualizacion_Escalado], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Escalado, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Escalado Ecua')
ax[3, 1].axis('off')


plt.show()


####################################################################
#                              ROTACION   1                        #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#ROTACION 1
Noche = io.imread("Imagen_Noche.jpg")
type(Noche)
Noche.shape

#Imagen ROTACION
ax[0, 1].imshow(Noche[::-1])
ax[0, 1].set_title('Rotacion')
ax[0, 1].axis('off')


#Histograma Imagen ROTACION sin ecualizar

for i, c in enumerate(color):
    hist_Noche = cv2.calcHist([Noche], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Noche, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Rotacion')
ax[1, 1].axis('off')


#Ecualizacion imagen ROTACION
Ecua_Noche = cv2.cvtColor(Noche[::-1],cv2.COLOR_BGR2YUV)
Ecua_Noche[:,:,0] = cv2.equalizeHist(Ecua_Noche[:,:,0])
Ecualizacion_Noche = cv2.cvtColor(Ecua_Noche,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Noche)
ax[2, 1].set_title('Imagen Rotacion Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen ROTACION ecualizada

for i, c in enumerate(color):
    hist_ecua_Noche = cv2.calcHist([Ecualizacion_Noche], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Noche, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Rotacion Ecua')
ax[3, 1].axis('off')


plt.show()



####################################################################
#                              ROTACION  2                         #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#ROTACION 2
ancho = res2.shape[1] #columnas
alto = res2.shape[0] # filas
    
Rota = cv2.getRotationMatrix2D((ancho//2,alto//2),110,1)
Rotacion = cv2.warpAffine(res2,Rota,(ancho,alto))

#Imagen ROTACION
ax[0, 1].imshow(Rotacion)
ax[0, 1].set_title('Rotacion')
ax[0, 1].axis('off')


#Histograma Imagen ROTACION sin ecualizar

for i, c in enumerate(color):
    hist_Rotacion = cv2.calcHist([Rotacion], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Rotacion, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Rotacion')
ax[1, 1].axis('off')


#Ecualizacion imagen ROTACION
Ecua_Rotacion = cv2.cvtColor(Rotacion,cv2.COLOR_BGR2YUV)
Ecua_Rotacion[:,:,0] = cv2.equalizeHist(Ecua_Rotacion[:,:,0])
Ecualizacion_Rotacion = cv2.cvtColor(Ecua_Rotacion,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Rotacion)
ax[2, 1].set_title('Imagen Rotacion Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen ROTACION ecualizada

for i, c in enumerate(color):
    hist_ecua_Rotacion = cv2.calcHist([Ecualizacion_Rotacion], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Rotacion, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Rotacion Ecua')
ax[3, 1].axis('off')


plt.show()


####################################################################
#                         TRASLACION A FIN   1                     #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#TRASLACION 1
traslacion1 = cv2.warpAffine(res1,M,(ancho,alto))


#Imagen TRASLACION
ax[0, 1].imshow(traslacion1)
ax[0, 1].set_title('traslacion 1')
ax[0, 1].axis('off')


#Histograma Imagen TRASLACION sin ecualizar

for i, c in enumerate(color):
    hist_traslacion1 = cv2.calcHist([traslacion1], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_traslacion1, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma traslacion 1')
ax[1, 1].axis('off')


#Ecualizacion imagen TRASLACION
Ecua_traslacion1 = cv2.cvtColor(traslacion1,cv2.COLOR_BGR2YUV)
Ecua_traslacion1[:,:,0] = cv2.equalizeHist(Ecua_traslacion1[:,:,0])
Ecualizacion_traslacion1 = cv2.cvtColor(Ecua_traslacion1,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_traslacion1)
ax[2, 1].set_title('Imagen traslacion 1 Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen TRASLACION ecualizada

for i, c in enumerate(color):
    hist_ecua_traslacion1 = cv2.calcHist([Ecualizacion_traslacion1], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_traslacion1, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img traslacion 1 Ecua')
ax[3, 1].axis('off')


plt.show()


####################################################################
#                         TRASLACION A FIN   2                     #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#TRASLACION 2
rows, cols, ch = res1.shape
pts1 = np.float32([[50, 50],
                       [200, 50], 
                       [50, 200]])
      
pts2 = np.float32([[10, 100],
                       [200, 50], 
                       [100, 250]])

M2 = cv2.getAffineTransform(pts1, pts2)
traslacion2 = cv2.warpAffine(res1, M2, (cols, rows))


#Imagen TRASLACION
ax[0, 1].imshow(traslacion2)
ax[0, 1].set_title('traslacion 2')
ax[0, 1].axis('off')


#Histograma Imagen TRASLACION sin ecualizar

for i, c in enumerate(color):
    hist_traslacion2 = cv2.calcHist([traslacion2], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_traslacion2, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma traslacion 2')
ax[1, 1].axis('off')


#Ecualizacion imagen TRASLACION
Ecua_traslacion2 = cv2.cvtColor(traslacion2,cv2.COLOR_BGR2YUV)
Ecua_traslacion2[:,:,0] = cv2.equalizeHist(Ecua_traslacion2[:,:,0])
Ecualizacion_traslacion2 = cv2.cvtColor(Ecua_traslacion2,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_traslacion2)
ax[2, 1].set_title('Imagen traslacion 2 Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen TRASLACION ecualizada

for i, c in enumerate(color):
    hist_ecua_traslacion2 = cv2.calcHist([Ecualizacion_traslacion2], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_traslacion2, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img traslacion 2 Ecua')
ax[3, 1].axis('off')


plt.show()



####################################################################
#                         TRASLACION A FIN   3                     #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#TRASLACION 3
rows,cols,ch = res1.shape
     
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M3 = cv2.getPerspectiveTransform(pts1,pts2)
traslacion3 = cv2.warpPerspective(res1,M3,(300,300))


#Imagen TRASLACION
ax[0, 1].imshow(traslacion3)
ax[0, 1].set_title('Traslacion 3')
ax[0, 1].axis('off')


#Histograma Imagen TRASLACION sin ecualizar

for i, c in enumerate(color):
    hist_traslacion3 = cv2.calcHist([traslacion3], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_traslacion3, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Traslacion 3')
ax[1, 1].axis('off')


#Ecualizacion imagen TRASLACION
Ecua_traslacion3 = cv2.cvtColor(traslacion3,cv2.COLOR_BGR2YUV)
Ecua_traslacion3[:,:,0] = cv2.equalizeHist(Ecua_traslacion3[:,:,0])
Ecualizacion_traslacion3 = cv2.cvtColor(Ecua_traslacion3,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_traslacion3)
ax[2, 1].set_title('Imagen Traslacion 3 Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen TRASLACION ecualizada

for i, c in enumerate(color):
    hist_ecua_traslacion3 = cv2.calcHist([Ecualizacion_traslacion3], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_traslacion3, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Traslacion 3 Ecua')
ax[3, 1].axis('off')


plt.show()



####################################################################
#                             TRANSPUESTA      1                   #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#TRANSPUESTA 1

Transpuesta1 = cv2.transpose(res1)

#Imagen TRANSPUESTA
ax[0, 1].imshow(Transpuesta1)
ax[0, 1].set_title('Transpuesta 1')
ax[0, 1].axis('off')


#Histograma Imagen TRANSPUESTA sin ecualizar

for i, c in enumerate(color):
    hist_Transpuesta1 = cv2.calcHist([Transpuesta1], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Transpuesta1, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Transpuesta 1')
ax[1, 1].axis('off')


#Ecualizacion imagen TRANSPUESTA
Ecua_Transpuesta1 = cv2.cvtColor(Transpuesta1,cv2.COLOR_BGR2YUV)
Ecua_Transpuesta1[:,:,0] = cv2.equalizeHist(Ecua_Transpuesta1[:,:,0])
Ecualizacion_Transpuesta1 = cv2.cvtColor(Ecua_Transpuesta1,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Transpuesta1)
ax[2, 1].set_title('Imagen Transpuesta 1 Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen TRANSPUESTA ecualizada

for i, c in enumerate(color):
    hist_ecua_Transpuesta1 = cv2.calcHist([Ecualizacion_Transpuesta1], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Transpuesta1, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Transpuesta 1 Ecua')
ax[3, 1].axis('off')


plt.show()



####################################################################
#                             TRANSPUESTA      2                   #
####################################################################

Figura_Completa, ax = plt.subplots(4, 3)
Figura_Completa.set_size_inches(12, 42)

#Imagen 1
ax[0, 0].imshow(res1)
ax[0, 0].set_title('Imagen 1')
ax[0, 0].axis('off')


#Histograma Imagen 1 sin ecualizar

for i, c in enumerate(color):
    hist = cv2.calcHist([res1], [i], None, [256], [0, 256])
    ax[1, 0].plot(hist, color = c)
    plt.xlim([0,256])
ax[1, 0].set_title('Histograma Img 1')
ax[1, 0].axis('off')


#Ecualizacion imagen 1
Ecua1 = cv2.cvtColor(res1,cv2.COLOR_BGR2YUV)
Ecua1[:,:,0] = cv2.equalizeHist(Ecua1[:,:,0])
Ecualizacion1 = cv2.cvtColor(Ecua1,cv2.COLOR_YUV2BGR)

ax[2, 0].imshow(Ecualizacion1)
ax[2, 0].set_title('Imagen 1 Ecualizada')
ax[2, 0].axis('off')



#Histograma Imagen 1 ecualizada

for i, c in enumerate(color):
    hist_ecua = cv2.calcHist([Ecualizacion1], [i], None, [256], [0, 256])
    ax[3, 0].plot(hist_ecua, color = c)
    plt.xlim([0,256])
ax[3, 0].set_title('Histograma Img Ecua 1')
ax[3, 0].axis('off')



#Imagen 2
ax[0, 2].imshow(res2)
ax[0, 2].set_title('Imagen 2')
ax[0, 2].axis('off')




#Histograma Imagen 2 sin ecualizar

for i, c in enumerate(color):
    hist2 = cv2.calcHist([res2], [i], None, [256], [0, 256])
    ax[1, 2].plot(hist2, color = c)
    plt.xlim([0,256])
ax[1, 2].set_title('Histograma Img 2')
ax[1, 2].axis('off')


#Ecualizacion imagen 2
Ecua2 = cv2.cvtColor(res2,cv2.COLOR_BGR2YUV)
Ecua2[:,:,0] = cv2.equalizeHist(Ecua2[:,:,0])
Ecualizacion2 = cv2.cvtColor(Ecua2,cv2.COLOR_YUV2BGR)

ax[2, 2].imshow(Ecualizacion2)
ax[2, 2].set_title('Imagen 2 Ecualizada')
ax[2, 2].axis('off')



#Histograma Imagen 2 ecualizada

for i, c in enumerate(color):
    hist_ecua2 = cv2.calcHist([Ecualizacion2], [i], None, [256], [0, 256])
    ax[3, 2].plot(hist_ecua2, color = c)
    plt.xlim([0,256])
ax[3, 2].set_title('Histograma Img Ecua 2')
ax[3, 2].axis('off')



#TRANSPUESTA 2

def transponer(res1):
    t = []
    for i in range(len(res1[0])):
        t.append([])
        for j in range(len(res1)):
            t[i].append(res1[j][i])
    return t
Transpuesta1 = np.concatenate((res1,transponer(res1), res2), axis=1)

#Imagen TRANSPUESTA
ax[0, 1].imshow(Transpuesta1)
ax[0, 1].set_title('Transpuesta 1')
ax[0, 1].axis('off')


#Histograma Imagen TRANSPUESTA sin ecualizar

for i, c in enumerate(color):
    hist_Transpuesta1 = cv2.calcHist([Transpuesta1], [i], None, [256], [0, 256])
    ax[1, 1].plot(hist_Transpuesta1, color = c)
    plt.xlim([0,256])
ax[1, 1].set_title('Histograma Transpuesta 1')
ax[1, 1].axis('off')


#Ecualizacion imagen TRANSPUESTA
Ecua_Transpuesta1 = cv2.cvtColor(Transpuesta1,cv2.COLOR_BGR2YUV)
Ecua_Transpuesta1[:,:,0] = cv2.equalizeHist(Ecua_Transpuesta1[:,:,0])
Ecualizacion_Transpuesta1 = cv2.cvtColor(Ecua_Transpuesta1,cv2.COLOR_YUV2BGR)

ax[2, 1].imshow(Ecualizacion_Transpuesta1)
ax[2, 1].set_title('Imagen Transpuesta 1 Ecualizada')
ax[2, 1].axis('off')



#Histograma Imagen TRANSPUESTA ecualizada

for i, c in enumerate(color):
    hist_ecua_Transpuesta1 = cv2.calcHist([Ecualizacion_Transpuesta1], [i], None, [256], [0, 256])
    ax[3, 1].plot(hist_ecua_Transpuesta1, color = c)
    plt.xlim([0,256])
ax[3, 1].set_title('Histograma Img Transpuesta 1 Ecua')
ax[3, 1].axis('off')


plt.show()
