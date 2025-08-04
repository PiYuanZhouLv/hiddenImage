import util
from PIL import Image
import numpy as np
from scipy.optimize import minimize

def solve_exp_minimization(img1, img2):
    """
    最小化exp(alpha)+exp(beta)的优化求解
    
    参数:
    img1: 第一张图片的像素矩阵(0-255)
    img2: 第二张图片的像素矩阵(0-255)
    
    返回:
    (alpha, beta): 最优参数对
    """
    # 预处理输入数据
    A = img1.flatten().astype('float64')
    B = img2.flatten().astype('float64')
    n = len(A)
    
    # 定义目标函数
    def objective(x):
        return np.exp(x[0]) + np.exp(x[1])
    
    # 定义约束条件函数
    def constraint_func(x):
        return (255-A)*x[0] + B*x[1] - (B-A)
    
    # 约束条件的Jacobian矩阵
    def constraint_jac(x):
        return np.column_stack([255-A, B])
    
    # 边界条件
    bounds = [(0, 1), (0, 1)]
    
    # 初始猜测
    x0 = [0.5, 0.5]
    
    # 构造约束条件
    constraints = {
        'type': 'ineq',
        'fun': constraint_func,
        'jac': constraint_jac
    }
    
    # 求解优化问题
    res = minimize(objective, x0, 
                  constraints=[constraints],
                  bounds=bounds,
                  method='SLSQP',)
                #   options={'maxiter': 1000})
    
    if res.success:
        alpha, beta = np.clip(res.x, 0, 1)
        return alpha, beta
    else:
        raise ValueError(f"优化失败: {res.message}")


# resolve:
#   light
#   dark
#   blend
def mirageTank(light, dark, resolve='light', **kwargs):
    size, (light, dark) = util.resize([light, dark], **kwargs)
    if resolve == 'blend':
        a, b = solve_exp_minimization(np.array(light), np.array(dark))
        print(a, b)
        light = (np.array(light) * (1-a) + 255 * a) / 255
        dark = np.array(dark) * (1-b) / 255
        # assert (light >= dark).all()
        a = 1 - (light - dark)
        c = dark / (a+1e-100)
        c[np.isclose(a, 0)] = 0
        new = np.empty((*size[::-1], 2), dtype='uint8')
        new[:, :, 0] = (c*255).astype('uint8')
        new[:, :, 1] = (a*255).astype('uint8')
    else:
        light = np.array(light) / 255
        dark = np.array(dark) / 255
        a = 1 - light + dark
        c = (light + a - 1) / (a+1e-100) if resolve == 'light' else dark / (a + 1e-100)
        c[np.isclose(a, 0)] = 0
        c[a >= 253/255] = (light if resolve == 'light' else dark)[a>=253/255]
        a[a >= 253/255] = 1
        new = np.empty((*size[::-1], 2), dtype='uint8')
        new[:, :, 0] = (c*255).astype('uint8')
        new[:, :, 1] = (a*255).astype('uint8')
    return Image.fromarray(new, mode="LA")

if __name__ == '__main__':
    test_img = lambda x: Image.open(f'test-img/mer/{x}.png').convert('L')
    mirageTank(test_img(35), test_img(28), resolve="blend").save('testMT.png')
