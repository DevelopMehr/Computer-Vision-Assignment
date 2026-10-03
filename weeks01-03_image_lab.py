import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def inspect_image(image_path: str) -> dict:
    """Task 1: Load the image and return its measured image-data properties."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # cv2.imread loads images in BGR format by default
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Unable to read image at path: {image_path}")

    height, width, channels = img_bgr.shape
    shape_list = [height, width, channels]
    pixel_count = height * width
    estimated_bytes = pixel_count * channels * 1  # 8 bits = 1 byte per channel
    color_order = "BGR"

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "shape": shape_list,
        "pixel_count": pixel_count,
        "estimated_bytes": estimated_bytes,
        "color_order": color_order
    }


def create_pixel_views(image_path: str, output_dir: str) -> dict:
    """Task 2: Create the labeled channel, grayscale, and downsampled views."""
    os.makedirs(output_dir, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    height, width = img_bgr.shape[:2]
    original_size = [width, height]

    # Separate Red, Green, Blue channels
    r_channel = img_rgb[:, :, 0]
    g_channel = img_rgb[:, :, 1]
    b_channel = img_rgb[:, :, 2]

    # Grayscale conversion
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Downsample by half width and height
    new_width = width // 2
    new_height = height // 2
    downsampled = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    downsampled_size = [new_width, new_height]

    # Plot figure with 6 subplots
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0, 0].imshow(r_channel, cmap='Reds')
    axes[0, 0].set_title('Red Channel')
    
    axes[0, 1].imshow(g_channel, cmap='Greens')
    axes[0, 1].set_title('Green Channel')
    
    axes[0, 2].imshow(b_channel, cmap='Blues')
    axes[0, 2].set_title('Blue Channel')
    
    axes[1, 0].imshow(gray, cmap='gray')
    axes[1, 0].set_title('Grayscale')
    
    axes[1, 1].imshow(downsampled)
    axes[1, 1].set_title(f'Downsampled ({new_width}x{new_height})')
    
    axes[1, 2].imshow(img_rgb)
    axes[1, 2].set_title('Original RGB')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "pixel_views.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    return {
        "original_size": original_size,
        "downsampled_size": downsampled_size,
        "output_path": output_path
    }


def create_adjustments(
    image_path: str,
    output_dir: str,
    brightness_delta: int = 40,
    contrast_factor: float = 1.5,
    threshold: int = 127,
) -> dict:
    """Task 3: Create labeled brightness, contrast, and threshold results."""
    if not (0 <= threshold <= 255):
        raise ValueError("Threshold must be in the range 0-255.")

    os.makedirs(output_dir, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Brightness adjustment
    brighter = np.clip(gray.astype(np.int16) + brightness_delta, 0, 255).astype(np.uint8)

    # Contrast adjustment
    contrast = np.clip(gray.astype(np.float32) * contrast_factor, 0, 255).astype(np.uint8)

    # Binary Thresholding (Pixels > threshold become 255/white, else 0/black)
    _, thresh_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Plotting montage
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Grayscale')

    axes[0, 1].imshow(brighter, cmap='gray')
    axes[0, 1].set_title(f'Brighter (+{brightness_delta})')

    axes[1, 0].imshow(contrast, cmap='gray')
    axes[1, 0].set_title(f'Higher Contrast (x{contrast_factor})')

    axes[1, 1].imshow(thresh_img, cmap='gray')
    axes[1, 1].set_title(f'Thresholded (T={threshold})')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "adjustments.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    return {
        "brightness_delta": brightness_delta,
        "contrast_factor": contrast_factor,
        "threshold": threshold,
        "output_path": output_path
    }


def create_blur_and_edges(
    image_path: str,
    output_dir: str,
    kernel_size: int = 5,
) -> dict:
    """Task 4: Create labeled grayscale, mean-blur, and Sobel-edge results."""
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    os.makedirs(output_dir, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Mean Blur (Box Filter)
    blurred = cv2.blur(gray, (kernel_size, kernel_size))

    # Sobel Edge Detection helper
    def get_sobel_magnitude(src):
        sobelx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobelx, sobely)
        return np.clip(magnitude, 0, 255).astype(np.uint8)

    edges_orig = get_sobel_magnitude(gray)
    edges_blur = get_sobel_magnitude(blurred)

    # Plotting montage
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Grayscale')

    axes[0, 1].imshow(blurred, cmap='gray')
    axes[0, 1].set_title(f'Mean Blur (Kernel {kernel_size}x{kernel_size})')

    axes[1, 0].imshow(edges_orig, cmap='gray')
    axes[1, 0].set_title('Sobel Edges (Original)')

    axes[1, 1].imshow(edges_blur, cmap='gray')
    axes[1, 1].set_title('Sobel Edges (Blurred)')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "blur_and_edges.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    return {
        "kernel_size": kernel_size,
        "output_path": output_path
    }


def run_lab(image_path: str, output_dir: str) -> dict:
    """Task 5: Run Tasks 1-4 and return their results together."""
    res1 = inspect_image(image_path)
    res2 = create_pixel_views(image_path, output_dir)
    res3 = create_adjustments(image_path, output_dir)
    res4 = create_blur_and_edges(image_path, output_dir)

    return {
        "task1": res1,
        "task2": res2,
        "task3": res3,
        "task4": res4
    }

def main() -> None:
    """Run the lab using the required repository paths."""
    image_path = "images/original.jpg"
    output_dir = "outputs"
    results = run_lab(image_path, output_dir)
    print("\n--- TASK 1 RESULTS ---")
    print(results["task1"])


if __name__ == "__main__":
    main()