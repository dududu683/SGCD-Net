import os
import torch
import cv2
import numpy as np
from models.model import ConditionedUNet
from models.diffusion import DiffusionModel
from models.utils import get_named_beta_schedule

def load_model(checkpoint_path, device, model_config):

    model = ConditionedUNet(**model_config).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def enhance_image(model, diffusion, degraded_img, device, num_steps=50, ddim=True):

    model.eval()

    if degraded_img.min() >= 0:
        degraded_img = degraded_img * 2 - 1
    degraded_img = degraded_img.to(device)
    shape = degraded_img.shape

    with torch.no_grad():
        enhanced = diffusion.p_sample_loop(
            cond=degraded_img,
            shape=shape,
            num_steps=num_steps,
            only_last=True,
            ddim=ddim,
            ddim_eta=0.0
        )

    enhanced = (enhanced + 1) / 2
    enhanced = torch.clamp(enhanced, 0, 1)
    return enhanced

def main():

    test_dir = '/path/to/your/test/images'          # 待增强的测试图像文件夹
    checkpoint_path = '/path/to/your/checkpoint/best_model.pth'   # 训练好的模型权重
    output_dir = '/path/to/your/output/folder'      # 增强结果保存文件夹
    # ---------------------------------------------------------------------

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")


    os.makedirs(output_dir, exist_ok=True)

    model_config = {
        'in_channels': 3,
        'model_channels': 64,
        'out_channels': 3,
        'num_res_blocks': 2,
        'channel_mult': (1, 2, 4, 8),
        'attention_resolutions': (16,),
    }

    model = load_model(checkpoint_path, device, model_config)


    betas = get_named_beta_schedule('linear', num_timesteps=1000).to(device)
    diffusion = DiffusionModel(model, betas, device)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    filenames = [f for f in os.listdir(test_dir) if f.lower().endswith(image_extensions)]
    print(f"Found {len(filenames)} images to enhance.")

    for filename in filenames:
        input_path = os.path.join(test_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img_bgr = cv2.imread(input_path)
        if img_bgr is None:
            print(f"Warning: Cannot read {input_path}, skipping.")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        enhanced_tensor = enhance_image(model, diffusion, img_tensor, device,
                                        num_steps=50, ddim=True)

        enhanced_np = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255
        enhanced_np = enhanced_np.astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, enhanced_bgr)
        print(f"Enhanced and saved: {output_path}")

    print("All done!")

if __name__ == '__main__':
    main()