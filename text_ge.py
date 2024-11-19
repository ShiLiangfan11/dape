import torch
import os
import json
from ram.models.ram_lora import ram
from ram import inference_ram as inference
from PIL import Image
from torchvision import transforms
from tqdm import tqdm  # 导入tqdm，用于进度条显示

# 加载模型
tag_model = ram(pretrained='models/ram_swin_large_14m.pth',
                pretrained_condition='models/DAPE.pth',
                image_size=384,
                vit='swin_l')
tag_model.eval()

# 确保模型在CUDA上运行
device = "cuda"
weight_dtype = torch.float16
tag_model.to(device, dtype=weight_dtype)

# 使用给定的输入图像生成文本的函数
@torch.no_grad()
def generate_text_from_image(input_image, user_prompt: str, positive_prompt: str) -> str:
    """
    通过给定的图像生成相关描述文本。
    :param input_image: 输入图像 (PIL Image)
    :param user_prompt: 用户提供的额外提示
    :param positive_prompt: 正向提示（用于描述图像的质量或内容）
    :return: 生成的文本描述
    """
    # 确保图像大小符合模型要求（384x384）
    resize_transform = transforms.Compose([
        transforms.Resize((384, 384)),  # 将图像调整为 384x384
        transforms.ToTensor()
    ])
    
    lq = resize_transform(input_image).unsqueeze(0).to(device).half()

    # 使用RAM模型生成图像嵌入
    res = inference(lq, tag_model)
    ram_encoder_hidden_states = tag_model.generate_image_embeds(lq)

    # 生成最终的文本提示
    validation_prompt = f"{res[0]}, {positive_prompt},"
    validation_prompt = validation_prompt if user_prompt == '' else f"{user_prompt}, {validation_prompt}"

    return validation_prompt

def process_images_in_folder(input_folder: str, output_json: str, user_prompt: str = '', positive_prompt: str = ''):
    """
    处理文件夹中的所有图像，并将生成的文本保存为JSON文件。
    :param input_folder: 输入文件夹路径
    :param output_json: 输出的JSON文件路径
    :param user_prompt: 用户自定义的额外提示
    :param positive_prompt: 正向提示
    """
    result = {}

    # 获取文件夹中的所有文件
    files = os.listdir(input_folder)

    # 过滤出所有图像文件（不包括目录）
    image_files = [f for f in files if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # 添加进度条
    for filename in tqdm(image_files, desc="Processing images", unit="image"):
        file_path = os.path.join(input_folder, filename)

        try:
            # 加载图像
            input_image = Image.open(file_path)

            # 生成文本描述
            generated_text = generate_text_from_image(input_image, user_prompt, positive_prompt)

            # 存储结果
            result[filename] = generated_text
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # 将结果写入JSON文件
    with open(output_json, 'w') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_json}")

# 调用示例
input_folder = '/home/shi/data/kodak'  # 输入文件夹路径
output_json = '/home/shi/NVC/dif_sr/SeeSR-main/pic/results.json'  # 输出结果的JSON文件路径

user_prompt = "A futuristic cityscape"  # 用户自定义提示
positive_prompt = "clean, high-resolution, 8k, best quality"  # 正向提示

process_images_in_folder(input_folder, output_json, user_prompt, positive_prompt)
