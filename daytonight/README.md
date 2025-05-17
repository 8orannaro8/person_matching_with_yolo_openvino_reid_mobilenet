
![image](https://github.com/user-attachments/assets/f53a7bf8-4652-4714-8874-e327aadbb534)


```
# High-Resolution Day-to-Night Image Translation with ResNet-based CycleGAN

This project performs **high-resolution image translation from day to night** using a ResNet-based generator, as originally proposed in [CycleGAN](https://arxiv.org/abs/1703.10593). It processes large images in patches to preserve detail and enables high-quality conversion without reducing resolution.

## 🔍 Features

- ✅ ResNet-based generator model (CycleGAN style)
- ✅ High-resolution image processing using overlapping patches
- ✅ Adaptive blending to reduce seams between patches
- ✅ Batch processing of entire folders
- ✅ JPEG quality control (default 95)
- ✅ Auto-skips already processed images

---

## 📁 Folder Structure

```

project\_root/
│
├── day\_to\_night.py            # Main code file
├── latest\_net\_G\_A.pth         # Pretrained generator model
├── processed\_images.txt       # Automatically managed
├── failed\_images.txt          # Automatically managed
├── README.md                  # This file
└── input\_folder/              # Folder with .jpg images

````

---

## 🔧 Installation

1. **Clone the repo:**

```bash
git clone https://github.com/your_username/day-to-night-cyclegan.git
cd day-to-night-cyclegan
````

2. **Install dependencies (Python 3.7+):**

```bash
pip install torch torchvision tqdm pillow
```

---

## 🚀 How to Run

1. **Place images to be processed** under a folder (e.g., `./images_to_process/`).
2. **Download a pre-trained generator model** and name it `latest_net_G_A.pth`, or modify the path in `model_path`.
3. **Run the script:**

```bash
python day_to_night.py
```

Make sure to set the path at the bottom of the file:

```python
input_folder = r"C:\path\to\your\images"
model_path = "latest_net_G_A.pth"
```

---

## 🧠 Model Details

This generator is based on CycleGAN’s ResNet architecture:

* 2 downsampling layers
* 9 residual blocks
* 2 upsampling layers
* Uses reflection padding and `BatchNorm2d`

It supports loading state dicts from multiple formats, such as:

* `state_dict`
* `net_G_A`
* `G_A`

---

## 🖼️ Patch-Based Image Processing

To handle **high-resolution images**, this tool:

* Splits the image into overlapping patches (default `256x256` with 32px overlap)
* Applies the generator to each patch
* Blends patches using a custom Gaussian-like weight map

---

## 📌 Example

Before:
![Day Image](https://via.placeholder.com/256x256.png?text=Day)

After:
![Night Image](https://via.placeholder.com/256x256.png?text=Night)

---

## ⚙️ Parameters

| Parameter    | Description                           | Default |
| ------------ | ------------------------------------- | ------- |
| `patch_size` | Patch size per inference              | `256`   |
| `overlap`    | Overlap between patches               | `32`    |
| `batch_size` | Number of images per processing batch | `50`    |
| `quality`    | JPEG output quality                   | `95`    |

---

## 📄 Logging

* `processed_images.txt`: Tracks successfully converted images
* `failed_images.txt`: Tracks failed image paths for review

---

## 📜 License

MIT License.

---

## 🙋‍♀️ Acknowledgements

* Based on [CycleGAN by Jun-Yan Zhu et al.](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* High-resolution patch stitching inspired by techniques used in medical image segmentation

```


