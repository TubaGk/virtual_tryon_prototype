import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from models.u2net.data_loader import RescaleT, ToTensorLab
from models.u2net.u2net import U2NET

MODEL_PATH = 'models/u2net/u2net.pth'
INPUT_DIR = 'inputs/persons'
OUTPUT_DIR = 'outputs/masks'

transform = transforms.Compose([
    RescaleT(320),
    ToTensorLab(flag=0)
])

def segment_image(image_path, model, device):
    image = Image.open(image_path).convert('RGB')
    image_name = os.path.basename(image_path)

    sample = {'image': image, 'label': image}
    sample = transform(sample)
    inputs_test = sample['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        d1, _, _, _, _, _, _ = model(inputs_test)
        pred = d1[:, 0, :, :]
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred_np = pred.squeeze().cpu().numpy() * 255
        mask = Image.fromarray(pred_np.astype(np.uint8)).resize(image.size)

        # Maskeden yüz ve pantolon  kısmı çıkarılıyor
        mask_np = np.array(mask)
        h = mask_np.shape[0]
        mask_np[:int(h * 0.2), :] = 0
        mask_np[int(h * 0.75):, :] = 0
        mask = Image.fromarray(mask_np)

        return mask

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = U2NET(3, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    for img_name in os.listdir(INPUT_DIR):
        img_path = os.path.join(INPUT_DIR, img_name)
        if img_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            print(f"Processing: {img_name}")
            mask = segment_image(img_path, model, device)
            mask.save(os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_mask.png"))

if __name__ == '__main__':
    main()
