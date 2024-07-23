from PIL import Image
import os
import cv2
import numpy as np

def resize_black(desired_size, im_pth, overwrite=False, print_oldsize=True):
    with Image.open(im_pth) as im:
        old_size = im.size
        ratio = desired_size / max(old_size)
        new_size = tuple(int(x * ratio) for x in old_size)

        im_resized = im.resize(new_size, resample=Image.BILINEAR)

        new_im = Image.new("RGB", (desired_size, desired_size))
        paste_position = ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
        new_im.paste(im_resized, paste_position)

        if overwrite:
            new_im.save(im_pth)

    return new_im, im_pth

def color_to_3_channels(img_path, overwrite=False):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = np.zeros_like(img)
    img2[:,:,0] = gray
    img2[:,:,1] = gray
    img2[:,:,2] = gray
    if overwrite:
        cv2.imwrite(img_path, img2)
    return img2, img_path

if __name__ == "__main__":
    
    TRAIN_DIR = 'DATASETS/train'
    TEST_DIR = 'DATASETS/test'
    
    folders = [TRAIN_DIR, TEST_DIR]
    desired_size = 299  
    for folder in folders:
        for subfol in os.scandir(folder):
            for img in os.scandir(subfol):
                if img.is_file() and img.name.endswith(('.jpg', '.jpeg', '.png')):  
                    print(img.name)
                    # Convert image to 3 channels
                    img_path = img.path
                    img_rgb, _ = color_to_3_channels(img_path, overwrite=True)
                    
                    # Resize and pad image with black background
                    resized_img, _ = resize_black(desired_size, img_path, overwrite=True)
                    