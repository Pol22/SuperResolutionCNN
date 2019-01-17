from PIL import Image
import os
import os.path


IMAGE_DIR = 'images'
IMG_SIZE = 256


images = os.listdir(IMAGE_DIR)

for image in images:
    img_path = os.path.join(IMAGE_DIR, image)
    img = Image.open(img_path)
    width, height = img.size
    for i in range(width // IMG_SIZE - 1):
        for j in range(height // IMG_SIZE - 1):
            crop_img = img.crop((i*IMG_SIZE, j*IMG_SIZE,
                (i+1)*IMG_SIZE - 1, (j+1)*IMG_SIZE - 1))
            
