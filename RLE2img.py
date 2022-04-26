import cv2
import numpy as np
import math

# Import fungsi zigzag
from zigzag import *

QUANTIZATION_MAT = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56 ],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])

# Menentukan ukuran blok
block_size = 8

# Membaca image.txt ke decode 
with open('image.txt', 'r') as myfile:
    image=myfile.read() 

details = image.split()

# Mendapatkan integer dari h dan w
h = int(''.join(filter(str.isdigit, details[0])))
w = int(''.join(filter(str.isdigit, details[1])))

# Deklarasi array 
array = np.zeros(h*w).astype(int)


# Melakukan loop 
k = 0
i = 2
x = 0
j = 0


# Loop memberikan susunan gambar 
while k < array.shape[0]:
    if(details[i] == ';'):
        break
    if "-" not in details[i]:
        array[k] = int(''.join(filter(str.isdigit, details[i])))        
    else:
        array[k] = -1*int(''.join(filter(str.isdigit, details[i])))        

    if(i+3 < len(details)):
        j = int(''.join(filter(str.isdigit, details[i+3])))

    if j == 0:
        k = k + 1
    else:                
        k = k + j + 1        

    i = i + 2

array = np.reshape(array,(h,w))

# Membangun matriks
i = 0
j = 0
k = 0

# Menginisialisasi gambar yang dikompres
padded_img = np.zeros((h,w))

while i < h:
    j = 0
    while j < w:        
        temp_stream = array[i:i+8,j:j+8]                
        block = inverse_zigzag(temp_stream.flatten(), int(block_size),int(block_size))            
        de_quantized = np.multiply(block,QUANTIZATION_MAT)                
        padded_img[i:i+8,j:j+8] = cv2.idct(de_quantized)        
        j = j + 8        
    i = i + 8

# Menggapai ke nilai maksimal ke minimal 8 bit 
padded_img[padded_img > 255] = 255
padded_img[padded_img < 0] = 0

# Mengkompres gambar dari compressed_image.mp file
cv2.imwrite("compressed_image.png",np.uint8(padded_img))
