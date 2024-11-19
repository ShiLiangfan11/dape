# 更新日志

### 24.11.19

#### 环境配置
参照 [SeeSR](https://github.com/cswry/SeeSR) 进行环境配置。

#### 模型下载
1. 创建 `model` 文件夹。
2. 下载以下两个文件，并将其放入 `model` 文件夹中：
   - `DAPE.pth`
   - `ram_swin_large_14m.pth`

   下载链接：[Google Drive](https://drive.google.com/drive/folders/12HXrRGEXUAnmHRaf0bIn-S8XSK4Ku0JO)

#### 代码运行
1. 修改以下内容并运行：

   input_folder = '/home/shi/data/kodak'  # 输入文件夹路径
   output_json = '/home/shi/NVC/dif_sr/SeeSR-main/pic/results.json'  # 输出结果的JSON文件路径

