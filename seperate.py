from PIL import Image
import numpy as np

def seperate_imgs(img: Image.Image, n):
    sub_size = tuple(map(lambda x: x // n, img.size))
    full_size = tuple(map(lambda x: x * n, sub_size))
    img = img.crop((0, 0, *full_size))
    arr = np.array(img)
    sub_arr = []
    X, Y, _ = np.meshgrid(np.arange(full_size[0]), np.arange(full_size[1]), np.zeros(3))
    for i in range(n):
        sub_arr.append(arr[((X-Y)%n==i)*(Y//n*n==Y)].reshape((*sub_size[::-1], 3)))
    return [Image.fromarray(img) for img in sub_arr]

if __name__ == '__main__':
    test_image = lambda x: Image.open(f'test-img/sep/{x}.png')
    [img.save(f'test{~i}.png') for i, img in enumerate(seperate_imgs(test_image('mix'), 2))]
