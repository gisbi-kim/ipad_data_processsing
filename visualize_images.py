import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def check_bit_depth(image_path):
    # 이미지를 로드합니다. -1은 이미지를 원래 깊이로 불러옵니다.
    image = cv2.imread(image_path, -1)

    # dtype을 사용하여 이미지의 데이터 타입을 얻습니다.
    dtype = image.dtype

    if dtype == "uint8":
        return 8
    elif dtype == "uint16":
        return 16
    else:
        return "Unknown bit depth"


def extract_index(image_name):
    return int(image_name.split("_")[-1].split(".")[0])


def colorize_image(image_path):
    img_depth = check_bit_depth(image_path)

    if img_depth == 8:
        img_depth_scale = 255
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        color_rescaler = 100.0
        colormap = cv2.COLORMAP_BONE
    if img_depth == 16:
        img_depth_scale = 255 * 255
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        color_rescaler = 11.0
        colormap = cv2.COLORMAP_JET

    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    normalized = 255.0 * (image / img_depth_scale * color_rescaler)
    normalized = np.clip(normalized, 0, 255)  # clip values to [0, 255] range
    normalized_uint8 = normalized.astype(np.uint8)

    colorized = cv2.applyColorMap(normalized_uint8, colormap)

    return colorized


def read_and_resize_rgb_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image at {path}")
    resized_image = cv2.resize(image, (192, 256))
    return resized_image


def merge_images(conf_path, depth_path, rgb_path):
    conf_colored = colorize_image(conf_path)
    depth_colored = colorize_image(depth_path)
    resized_rgb_image = read_and_resize_rgb_image(rgb_path)
    merged = np.hstack((conf_colored, depth_colored, resized_rgb_image))
    return merged


def save_to_video(images, output_path, fps=10, width=256, height=192):
    """
    주어진 이미지 리스트를 비디오로 저장합니다.

    :param images: 저장할 이미지의 리스트
    :param output_path: 비디오를 저장할 경로
    :param fps: 비디오의 프레임 속도 (default: 10hz)
    :param width: 비디오의 너비
    :param height: 비디오의 높이
    """
    print(f"save_to_video to {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 코덱으로 변경
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        out.write(image)

    out.release()


def main():
    folder_path = "2023_07_30_09_51_53"
    output_folder = "merged_images"
    os.makedirs(output_folder, exist_ok=True)

    # 이미지들을 이름으로 정렬
    conf_images = sorted(
        [
            f
            for f in os.listdir(folder_path)
            if f.startswith("conf_") and f.endswith(".png")
        ]
    )
    depth_images = sorted(
        [
            f
            for f in os.listdir(folder_path)
            if f.startswith("depth_") and f.endswith(".png")
        ]
    )
    rgb_images = sorted(
        [
            f
            for f in os.listdir(folder_path)
            if f.startswith("frame_") and f.endswith(".jpg")
        ]
    )

    # rgb 이미지의 인덱스에 맞게 conf_images와 depth_images를 필터링
    rgb_indices = [extract_index(rgb_image) for rgb_image in rgb_images]
    conf_images = [f"conf_{str(idx).zfill(5)}.png" for idx in rgb_indices]
    depth_images = [f"depth_{str(idx).zfill(5)}.png" for idx in rgb_indices]

    output_images = []  # 합쳐진 이미지들을 저장하기 위한 리스트
    for ii, (conf, depth, rgb) in enumerate(zip(conf_images, depth_images, rgb_images)):
        conf_path = os.path.join(folder_path, conf)
        depth_path = os.path.join(folder_path, depth)
        rgb_path = os.path.join(folder_path, rgb)

        merged_image = merge_images(conf_path, depth_path, rgb_path)
        output_path = os.path.join(output_folder, f"merged_{rgb.split('_')[1]}")

        cv2.imwrite(output_path, merged_image)
        print(f"Saved {output_path}")

        output_images.append(merged_image)

    # 이미지 리스트를 사용하여 비디오 파일을 생성
    print(merged_image.shape)
    save_to_video(
        output_images,
        "output_video.mp4",
        width=merged_image.shape[1],
        height=merged_image.shape[0],
    )


if __name__ == "__main__":
    main()
