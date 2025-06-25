from pathlib import Path
import sys
import numpy as np
import cv2
import cmapy
import matplotlib.pyplot as plt


def load_npy_images_from_folder(folder_path: Path) -> list[np.ndarray]:
    return [np.load(f) for f in sorted(folder_path.glob("*.npy"))]


def interactive_crop_files(images: list[np.ndarray], crop_output_dir: Path):
    image_index = 0
    image = images[image_index]

    bounding_box_size = 120
    x, y, w, h = 100, 100, bounding_box_size, bounding_box_size

    window_name = "image_display"
    cropped_window_name = "cropped_image_display"

    while True:
        cropped_image = image[y:y + h, x:x + w]
        cropped_image_rgb = cropped_image.copy()

        display_image = image.copy()
        display_image_norm = cv2.normalize(display_image, None, 0, 255, cv2.NORM_MINMAX)
        display_image = cv2.applyColorMap(display_image_norm.astype(np.uint8), cmapy.cmap("afmhot"))
        cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow(window_name, display_image)

        cropped_image_rgb = cv2.normalize(cropped_image_rgb, None, 0, 255, cv2.NORM_MINMAX)
        cropped_image_rgb = cv2.applyColorMap(cropped_image_rgb.astype(np.uint8), cmapy.cmap("afmhot"))
        cv2.imshow(cropped_window_name, cropped_image_rgb)

        key = cv2.waitKey(1)

        if key == ord("a"):
            x -= 10
        elif key == ord("d"):
            x += 10
        elif key == ord("w"):
            y -= 10
        elif key == ord("s"):
            y += 10
        elif key == ord("e"):
            w -= 10
            h -= 10
        elif key == ord("r"):
            w += 10
            h += 10
        elif key == ord("f"):
            if image_index < len(images) - 1:
                image_index += 1
                image = images[image_index]
            else:
                print(f"image index: {image_index}, cannot go higher")
        elif key == ord("g"):
            if image_index > 0:
                image_index -= 1
                image = images[image_index]
            else:
                print(f"image index: {image_index}, cannot go lower")
        elif key == ord(" "):
            current_output_file_list = list(crop_output_dir.glob("*.npy"))
            output_index = max([int(f.stem.split("image_")[1]) for f in current_output_file_list if "image_" in f.stem] + [-1]) + 1

            filename = f"image_{output_index}"
            np.save(crop_output_dir / f"{filename}.npy", cropped_image)
            plt.imsave(crop_output_dir / f"{filename}.png", cropped_image, vmin=image.min(), vmax=image.max())
            print(f"Saved {filename}.npy and {filename}.png")
        elif key != -1:
            print(f"key not bound: {key}")

        if x < 0:
            x = 10
            w = 100
            h = 100
        if y < 0:
            y = 10
            w = 100
            h = 100
        if x + w > image.shape[1]:
            x = 10
            w = 100
            h = 100
        if y + h > image.shape[0]:
            y = 10
            w = 100
            h = 100

        elif key == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python crop_tool.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])
    output_folder.mkdir(exist_ok=True)

    images = load_npy_images_from_folder(input_folder)
    if not images:
        print("No .npy files found in the input folder.")
        sys.exit(1)

    interactive_crop_files(images, output_folder)