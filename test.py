import argparse
import paddle
# import torch
# import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import paddle.vision.transforms as transforms
from models import RDN
from utils import convert_rgb_to_y, denormalize, calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()

    device = paddle.CUDAPlace(0)

    model = RDN(scale_factor=args.scale,
                num_channels=3,
                num_features=args.num_features,
                growth_rate=args.growth_rate,
                num_blocks=args.num_blocks,
                num_layers=args.num_layers).to(device)
    

    state_dict = model.state_dict() # model.state_dict()是浅拷贝,model.load_state_dict(xxx) 是深拷贝
    
    # for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():   在pytorch中map_location=lambda storage, loc: storage将数据加载在GPU
    #     if n in state_dict.keys():
    #         state_dict[n].copy_(p)
    #     else:
    #         raise KeyError(n)
    
    # for name, param in state_dict.items():
    #     print(name, param.shape)

    # img_tensor = transforms.ToTensor()(args.image_file)
    # print("Image tensor shape:", img_tensor.shape)
    # print("Image tensor data type:", img_tensor.dtype)

    # weights = paddle.load(args.weights_file)
    # # 更新模型的状态字典
    # for n, p in weights.items():
    #     if n in state_dict.keys():
    #         state_dict[n].set_value(p.numpy())
    #     else:
    #         raise KeyError(n)

    # # 设置模型的状态字典
    # model.set_state_dict(state_dict)


    for n, p in paddle.load(args.weights_file).items():  
        if n in state_dict.keys():
            state_dict[n].set_value(p.numpy())
        else:
             raise KeyError(n)
    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
    hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
    # lr = torch.from_numpy(lr).to(device)
    # hr = torch.from_numpy(hr).to(device)
    lr = paddle.to_tensor(lr)
    hr = paddle.to_tensor(hr)
    with paddle.no_grad():
        preds = model(lr).squeeze(0)

    preds_y = convert_rgb_to_y(denormalize(preds), dim_order='chw')
    hr_y = convert_rgb_to_y(denormalize(hr.squeeze(0)), dim_order='chw')

    preds_y = preds_y[args.scale:-args.scale, args.scale:-args.scale]
    hr_y = hr_y[args.scale:-args.scale, args.scale:-args.scale]

    psnr = calc_psnr(hr_y, preds_y)
    print('PSNR: {:.2f}'.format(psnr.numpy()[0]))

    # output = pil_image.fromarray(denormalize(preds).permute(1, 2, 0).byte().cpu().numpy())

    # 将输出张量转换为numpy数组并进行维度重排
    output_np = preds.numpy().transpose(1, 2, 0)

    # 对数据进行反归一化、类型转换等操作
    output_np = denormalize(output_np)
    output_np = output_np.astype(np.uint8)

    # 将numpy数组转换为PIL图像
    output = pil_image.fromarray(output_np)
    output.save(args.image_file.replace('.', '_rdn_x{}.'.format(args.scale)))
