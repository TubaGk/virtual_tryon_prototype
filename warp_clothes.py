import cv2
import numpy as np

def warp_clothing(body_img_path, clothing_img_path, src_points, dest_points, output_path):
    body = cv2.imread(body_img_path)
    clothing = cv2.imread(clothing_img_path)

    # affine matris
    matrix = cv2.getAffineTransform(np.float32(src_points), np.float32(dest_points))
    warped_clothing = cv2.warpAffine(clothing, matrix, (body.shape[1], body.shape[0]))

    cv2.imwrite(output_path, warped_clothing)
    print(f" Warp işlemi tamamlandı: {output_path}")

if __name__ == "__main__":
    # t-shirt2.jpg içinden (sol omuz, sağ omuz, bel orta)
    src_pts = [(36, 43), (167, 41), (100, 228)]
    # person2.jpg içinden (sol omuz, sağ omuz, bel orta)
    dest_pts = [(431, 401), (780, 392), (623, 804)]

    warp_clothing(
        "inputs/persons/person2.jpg",
        "inputs/clothes/tshirt2.jpg",
        src_pts,
        dest_pts,
        "outputs/tryon_results/warped_result.jpg"
    )
