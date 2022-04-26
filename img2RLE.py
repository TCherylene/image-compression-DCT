import cv2
import numpy as np
import math

from zigzag import *


def get_run_length_encoding(image):
    i = 0
    skip = 0
    stream = []    
    bitstream = ""
    image = image.astype(int)
    while i < image.shape[0]:
        if image[i] != 0:            
            stream.append((image[i],skip))
            bitstream = bitstream + str(image[i])+ " " +str(skip)+ " "
            skip = 0
        else:
            skip = skip + 1
        i = i + 1

    return bitstream

# mendefinisikan blok size 
block_size = 8

# Quantization Matriks 
QUANTIZATION_MAT = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56 ],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])

# Membaca gambar dalam grayscale style
img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

# Mendapat ukuran dari gambar
[h , w] = img.shape

# Jumlah blok yang dibutuhkan : Perhitungan 
height = h
width = w
h = np.float32(h) 
w = np.float32(w) 

nbh = math.ceil(h/block_size)
nbh = np.int32(nbh)

nbw = math.ceil(w/block_size)
nbw = np.int32(nbw)


# gambarnya di padding, karena terkadang ukuran pada gambar tidak dapat di bagi untuk ukuran blok
# Mendapatkan ukuran dari gambar yang sudah di padding dengan mengalikan ukuran blok dengan jumlah blok di tinggi/lebar

# Tinggi dari padding gambar 
H =  block_size * nbh

# Lebar dari padding gambar
W =  block_size * nbw

# Membuat matriks dengan ukuran H,W
padded_img = np.zeros((H,W))

# atau cara lain menggunakan sebagai berikut
padded_img[0:height,0:width] = img[0:height,0:width]

cv2.imwrite('uncompressed.png', np.uint8(padded_img))

# Memulai encoding 
# Membagi gambar dengan ukuran blok 8 x 8 
for i in range(nbh):
    
        #Menghitung indeks baris awal dan akhir blok
        row_ind_1 = i*block_size                
        row_ind_2 = row_ind_1+block_size
        
        for j in range(nbw):
            
            # Mehitung indeks kolom awal & akhir blok
            col_ind_1 = j*block_size                       
            col_ind_2 = col_ind_1+block_size
                        
            block = padded_img[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]                      
            DCT = cv2.dct(block)            

            DCT_normalized = np.divide(DCT,QUANTIZATION_MAT).astype(int)            
    
            reordered = zigzag(DCT_normalized)

            # Membentuk kembali ukuran 8x8 
            reshaped= np.reshape(reordered, (block_size, block_size)) 
            
            # Menyalin matrix kembali 
            padded_img[row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = reshaped                        

cv2.imshow('encoded image', np.uint8(padded_img))

arranged = padded_img.flatten()

bitstream = get_run_length_encoding(arranged)
bitstream = str(padded_img.shape[0]) + " " + str(padded_img.shape[1]) + " " + bitstream + ";"

# Dituliskan ke image.txt
file1 = open("image.txt","w")
file1.write(bitstream)
file1.close()

cv2.destroyAllWindows()
