import numpy as np
from keras.models import load_model
from PIL import Image
from skimage.measure import compare_psnr


IMG_SIZE = 127
NUM_CHANELS = 3

SRmodel = 'srcnn_v4.h5'
model = load_model(SRmodel)

img_name = '220930100.jpg'
ref = Image.open('dataset/reference/' + img_name)
ref_arr = np.asarray(ref, np.uint8)
bic = Image.open('dataset/bicubic/' + img_name)

bic_arr_8 = np.asarray(bic, dtype=np.uint8)
bic_arr = np.asarray(bic, dtype=np.float32)
bic_arr1 = (bic_arr - 127) / 128
bic_arr1 = np.reshape(bic_arr1, (1, IMG_SIZE, IMG_SIZE, NUM_CHANELS))
sr_res = model.predict(bic_arr1)
sr_res = sr_res * 128 + 127
sr_res = np.reshape(sr_res, (IMG_SIZE, IMG_SIZE, NUM_CHANELS))
sr_res = np.asarray(sr_res, dtype=np.uint8)
res = Image.fromarray(sr_res)
print('Bicubic:', compare_psnr(bic_arr_8, ref_arr))
print('SRCNN:', compare_psnr(sr_res, ref_arr))

res.save('res.jpg')
ref.save('ref.jpg')
bic.save('bic.jpg')
