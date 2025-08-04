from collections.abc import Iterable
from PIL import Image, ImageEnhance
import numpy as np

calc_ratio = lambda s: s[0] / s[1]


from collections import defaultdict

def quantize_color(color, reduce):
    """将RGB颜色分量除以reduce取整"""
    return tuple((c // reduce) for c in color)

def find_dominant_edge_color(image, w=1, reduce=16):
    """
    统计图片边缘w像素范围内颜色（分量除以reduce取整后视为同一颜色）
    并返回这些颜色的平均值
    
    参数:
        image: numpy数组形式的图片，形状为(H,W,C)
        w: 边缘宽度(像素数)，默认为1
        
    返回:
        tuple: 平均颜色值
        int: 总颜色数量
    """
    if len(image.shape) == 2:
        image = image.reshape((*image.shape[::-1], 1))
    cn = image.shape[2]

    # 获取边缘区域
    top = image[:w, :, :]  # 只取前三个通道(RGB)
    bottom = image[-w:, :, :]
    left = image[w:-w, :w, :]
    right = image[w:-w, -w:, :]
    
    # 合并所有边缘像素
    edges = np.concatenate([
        top.reshape(-1, cn),
        bottom.reshape(-1, cn),
        left.reshape(-1, cn),
        right.reshape(-1, cn)
    ])
    
    # 统计量化后的颜色及其原始颜色总和
    color_groups = defaultdict(lambda: {'sum': np.zeros(cn), 'count': 0})
    
    for pixel in edges:
        quantized = quantize_color(pixel, reduce)
        color_groups[quantized]['sum'] += pixel
        color_groups[quantized]['count'] += 1
    
    if not color_groups:
        return (0,)*cn, 0
    
    # 找到出现次数最多的量化颜色组
    max_group = max(color_groups.items(), key=lambda x: x[1]['count'])
    _, data = max_group
    
    # 计算该组的平均颜色
    avg_color = tuple((data['sum'] / data['count']).astype(int).tolist())
    
    return avg_color, data['count']

# resize:
#   crop&resizeToB
#   crop&resizeToS
#   fill&resizeToB
#   fill&resizeToS
#   resizeToB
#   resizeToS
#   crop(ToS)
#   fill(ToB)
def resize(
    imgs,
    size=None,
    resize="crop&resizeToB",
    fill_color="auto",
    fill_autocalc_reduce = 16,
    sence_width=3,
    scale_down=1
):
    imgs = [img.reduce(scale_down) for img in imgs]
    sizes = [img.size for img in imgs]
    if not size:
        b, s = (
            max(enumerate(sizes), key=lambda x: x[1][0] * x[1][1])[0],
            min(enumerate(sizes), key=lambda x: x[1][0] * x[1][1])[0],
        )
        if "B" in resize or resize == "fill":
            base = b
        else:  # 'S' in resize or resize == 'crop'
            base = s
        size = sizes[base]
    else:
        base = -1
    ratio = calc_ratio(size)
    processed = []
    for i, img in enumerate(imgs):
        if i == base:
            processed.append(img)
            continue
        if "crop" in resize:
            if 'resize' not in resize:
                w, h = size
            elif calc_ratio(img.size) > ratio:
                w, h = int(img.size[1] * ratio), img.size[1]
            else:
                w, h = img.size[0], int(img.size[0] / ratio)
            l = (img.size[0] - w) // 2
            r = l + w
            u = (img.size[1] - h) // 2
            d = u + h
            img = img.crop((l, u, r, d))
        elif "fill" in resize:
            if 'resize' not in resize:
                w, h = size
            elif calc_ratio(img.size) < ratio:
                w, h = int(img.size[1] * ratio), img.size[1]
            else:
                w, h = img.size[0], int(img.size[0] / ratio)
            if fill_color == "auto":
                fill, count = find_dominant_edge_color(np.array(img), sence_width, reduce=fill_autocalc_reduce)
                print(count)
            else:
                fill = fill_color
            l = (w - img.size[0]) // 2
            u = (h - img.size[1]) // 2
            print(fill)
            nimg = Image.new(img.mode, (w, h), fill if not isinstance(fill[0], Iterable) else fill[i])
            nimg.paste(img, (l, u))
            img = nimg
        else:  # resize only
            pass
        processed.append(img.resize(size))

    return size, processed