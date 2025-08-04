from PIL import Image, ImageEnhance
import numpy as np

img = Image.open('test-img/sep/mix.png').convert('RGB')
arr = np.array(img)

W, H = img.size

arr_new = [np.empty((H//2, W//2, 3), dtype='uint8'), np.empty((H//2, W//2, 3), dtype='uint8')]

for y in range(H):
    for x in range(W):
        arr_new[x%2!=y%2][y//2][x//2] = arr[y][x]

img_new = [Image.fromarray(arrn, 'RGB') for arrn in arr_new]
[imgn.save(f'test-img/sep/{i}.png') for i, imgn in enumerate(img_new)]

def lighten(img, factor):
    ie = ImageEnhance.Brightness(img)
    return ie.enhance(factor)

import code
code.interact(local=globals())