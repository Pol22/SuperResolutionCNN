from PIL import Image
import os
import os.path


IMAGE_DIR = 'SR_training_datasets/T91'
SAVE_DIR = 'dataset'
IMG_SIZE = 127
LOW_RES = 63


images = os.listdir(IMAGE_DIR)
counter = 1
for image in images:
    img_path = os.path.join(IMAGE_DIR, image)
    img = Image.open(img_path)
    width, height = img.size
    for i in range(0, width-IMG_SIZE, 50):
        for j in range(0, height-IMG_SIZE, 50):
            ref_img = img.crop((i, j,
                i + IMG_SIZE, j + IMG_SIZE))
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
    
    print('[%i] Croped image: %s' % (counter, image))
    counter = counter + 1
            
            
