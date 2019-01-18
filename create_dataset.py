from PIL import Image
import os
import os.path


IMAGE_DIR = 'images'
SAVE_DIR = 'dataset'
IMG_SIZE = 256
LOW_RES = 128


images = os.listdir(IMAGE_DIR)

for image in images:
    img_path = os.path.join(IMAGE_DIR, image)
    img = Image.open(img_path)
    width, height = img.size
    for i in range(width // IMG_SIZE - 1):
        for j in range(height // IMG_SIZE - 1):
            ref_img = img.crop((i*IMG_SIZE, j*IMG_SIZE,
                (i+1)*IMG_SIZE, (j+1)*IMG_SIZE))
            resized_img = ref_img.resize(
                (LOW_RES, LOW_RES))
            bicubic_img = resized_img.resize(
                (IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            crop_name = image.split('.')[:-1]
            crop_name.append(str(i))
            crop_name.append(str(j))
            crop_name.append('.jpg')
            crop_name = ''.join(crop_name) 
            bicubic_path = os.path.join(SAVE_DIR, 'bicubic', crop_name)
            ref_path = os.path.join(SAVE_DIR, 'reference', crop_name)
            ref_img.save(ref_path)
            bicubic_img.save(bicubic_path)
            
            
