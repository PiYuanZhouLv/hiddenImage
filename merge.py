from PIL import Image, ImageEnhance
import numpy as np
import util

# resize: See 'util.py'
def merge_imgs(
    img_and_factor,
    base_size=None,
    out_size=None,
    resize="crop&resizeToB",
    fill_color="auto",
    fill_autocalc_reduce = 16,
    sence_width=3,
    scale_down=1,
    seperate=False,
):
    if base_size and out_size:
        raise ValueError("base_size and out_size can't be used together")
    n = len(img_and_factor)
    imgs, factors = tuple(zip(*img_and_factor))
    
    size, processed = util.resize(imgs, None if not (base_size or out_size) else (base_size if base_size else tuple(map(lambda x: x//n, out_size))), resize, fill_color, fill_autocalc_reduce, sence_width, scale_down)

    processed = [
        np.array(ImageEnhance.Brightness(img).enhance(factor).convert("RGB"))
        for img, factor in zip(processed, factors)
    ]

    new_image = (
        np.empty((size[1] * n, size[0] * n, 3), dtype="uint8")
        if not seperate
        else [np.zeros((size[1] * n, size[0] * n, 4), dtype="uint8") for _ in range(n)]
    )

    for y in range(size[1] * n):
        for x in range(size[0] * n):
            if not seperate:
                new_image[y][x] = processed[(x - y) % n][y // n][x // n]
            else:
                new_image[(x - y) % n][y][x][:3] = processed[(x - y) % n][y // n][
                    x // n
                ]
                new_image[(x - y) % n][y][x][3] = 255

    return (
        Image.fromarray(new_image)
        if not seperate
        else [Image.fromarray(new, "RGBA") for new in new_image]
    )


if __name__ == "__main__":
    test_img = lambda i: Image.open(f"test-img/mer/{i}.png")
    merge_imgs(
        [(test_img(8), 1), (test_img(4), 0.04), (test_img(17), 0.2)],
        out_size=(1920, 1080),
    ).save("test.png")
