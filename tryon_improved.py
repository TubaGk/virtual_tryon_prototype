import cv2
import numpy as np
import os

def apply_clothing_masked(body_image_path, clothing_image_path, mask_path, output_path):
    body = cv2.imread(body_image_path)
    clothing = cv2.imread(clothing_image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if body is None or clothing is None or mask is None:
        print("Görsellerden biri yüklenemedi.")
        return

    mask = mask / 255.0
    mask_3ch = cv2.merge([mask, mask, mask])

    clothing_resized = cv2.resize(clothing, (body.shape[1], body.shape[0]))

    # Sadece üst vücut
    clothing_masked = clothing_resized * mask_3ch
    body_masked = body * (1 - mask_3ch)
    result = (body_masked + clothing_masked).astype(np.uint8)

    cv2.imwrite(output_path, result)
    print(f"Giydirme tamamlandı: {output_path}")

if __name__ == "__main__":
    body_image = "inputs/persons/person2.jpg"
    clothing_image = "outputs/tryon_results/warped_result.jpg"
    mask_image = "outputs/masks/person2_mask.png"
    output_file = "outputs/tryon_results/final_tryon.jpg"

    apply_clothing_masked(body_image, clothing_image, mask_image, output_file)
