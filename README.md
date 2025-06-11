
# 🎨 Neural Style Transfer in PyTorch

This project implements **Neural Style Transfer** to apply the artistic style of one image (style) to a set of content images using PyTorch. It supports **training from scratch** and works with **custom image inputs**.

---

## 🧑‍💻 Intern Details

**Company:** CODTECH IT SOLUTIONS  
**Name:** Emil Saj Abraham  
**Intern ID:** CT08DL252  
**Domain:** AI  
**Duration:** 8 weeks  
**Mentor:** Neela Santhosh  

---

## 📸 Sample Result

<table>
  <tr>
    <th>Content Image</th>
    <th>Style Image</th>
    <th>Stylized Output</th>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/8750739d-8d67-4c87-b6c6-5f2c89c674ee" width="200"/></td>
    <td><img src="https://github.com/user-attachments/assets/ed25c4ee-93bc-4691-b5c2-ac2f4709b27e" width="200"/></td>
    <td><img src="https://github.com/user-attachments/assets/da54f9da-0621-419a-b166-8a77b69bf4ce" width="200"/></td>
  </tr>
</table>



---

## 🧰 Requirements

Install the required Python packages:

```bash
pip install torch torchvision matplotlib pillow
```

Recommended to use a virtual environment (`venv`, `conda`, etc).

---

## 📁 Project Structure

```
├── style_transfer.ipynb      # Main Jupyter Notebook
├── style-image/
│   └── images.jpg            # Artistic style image
├── input-image/
│   ├── image1.jpg            # Content images
│   └── image2.jpg
└── output/
    ├── styled_image1.jpg
    └── styled_image2.jpg
```

---

## 🧠 Model

Uses **VGG-19** pretrained on ImageNet as the feature extractor. The loss is a weighted combination of:

- **Content Loss (MSE):** Preserves structure of input image.
- **Style Loss (Gram Matrix MSE):** Matches style features.

---

## 🚀 Usage

### Run in a Python script or Jupyter Notebook:

```python
# Load style image
style_img = load_image("style-image/images.jpg")

# Load and loop through all content images
content_dir = "/kaggle/input/input-image"
for fname in os.listdir(content_dir):
    content_img = load_image(os.path.join(content_dir, fname))
    input_img = content_img.clone()

    # Run Style Transfer
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)

    # Save result
    save_image(output, f"output/styled_{fname}")
```

---

## 📊 Loss Behavior

- Style Loss reduces rapidly and plateaus → style captured early.
- Content Loss gradually decreases → structure preserved.

| Step | Style Loss | Content Loss |
|------|------------|--------------|
| 50   | 0.000077   | 19.384       |
| 500  | 0.000001   | 10.857       |
| 1000 | 0.000001   | 10.116       |

> 🟢 Model convergence is smooth and balanced.

---

## 📄 License

MIT License — free to use, modify, and distribute.
