# import torch
import paddle
import numpy as np


def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.



def convert_rgb_to_ycbcr(img, dim_order='hwc'):
    if dim_order == 'hwc':
        y = 16.0 + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 *
            img[..., 2]) / 256.0
        cb = 128.0 + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 
            112.439 * img[..., 2]) / 256.0
        cr = 128.0 + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 *
            img[..., 2]) / 256.0
    else:
        y = 16.0 + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]
            ) / 256.0
        cb = 128.0 + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]
            ) / 256.0
        cr = 128.0 + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]
            ) / 256.0
    return np.array([y, cb, cr]).transpose([1, 2, 0])

def denormalize(x):
    if isinstance(x, paddle.Tensor):
        return paddle.clip(x * 255.0, min=0.0, max=255.0)
    elif isinstance(x, np.ndarray):
        return np.clip(x * 255.0, 0.0, 255.0)
    else:
        raise Exception(
            'The denormalize function supports paddle.Tensor or np.ndarray types.'
            , type(x))


def preprocess(img, device):
    img = np.array(img).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.0
    x = paddle.to_tensor(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr


# def calc_psnr(img1, img2, max=255.0):
#     return 10. * ((max ** 2) / ((img1 - img2) ** 2).mean()).log10()
def calc_psnr(img1, img2,max=255.0):
    # return 10. * paddle.log10(1. / paddle.mean((img1 - img2) ** 2))
    return 10. * paddle.log10((max ** 2) / paddle.mean((img1 - img2) ** 2))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
