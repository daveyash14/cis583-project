import cv2
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.models.segmentation as segmentation
from PIL import Image
import open3d as o3d

# Segment the Image to Isolate the Person
def segment_image(image_path):
    model = segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # Capture original size for resizing mask

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    predictions = output.argmax(0).byte().numpy()

    # Resize predictions mask back to the original image size
    person_mask = (predictions == 15)  # class 15 is 'person'
    person_mask = cv2.resize(person_mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
    return person_mask.astype(bool)

# Generate Depth Map
def generate_depth_map(image_path, depth_map_path):
    image = Image.open(image_path)
    depth_map = np.random.uniform(0, 1, image.size[::-1]).astype(np.float32)
    depth_image_16bit = np.clip(depth_map * 1000, 0, 65535).astype(np.uint16)
    cv2.imwrite(depth_map_path, depth_image_16bit)
    print(f"Depth map saved to {depth_map_path}, size: {image.size}")

# Apply Segmentation Mask to Depth Map
def apply_mask_to_depth(mask, depth_image_path):
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    masked_depth = np.where(mask, depth_image, 0)
    return masked_depth

# Generate 3D Point Cloud from Masked Depth Map
def create_point_cloud_from_images(color_image_path, masked_depth_image, point_cloud_path):
    print(f"Attempting to create Open3D Image from depth data of shape: {masked_depth_image.shape}")

    if masked_depth_image.dtype != np.uint16:
        masked_depth_image = np.clip(masked_depth_image, 0, 65535).astype(np.uint16)

    try:
        depth_image_o3d = o3d.geometry.Image(masked_depth_image)
        print("Successfully converted to Open3D Image.")
    except ValueError as e:
        print(f"Error converting numpy array to Open3D Image: {str(e)}")
        return

    color_image = o3d.io.read_image(color_image_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image_o3d, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        masked_depth_image.shape[1], masked_depth_image.shape[0],
        525.0, 525.0,
        masked_depth_image.shape[1] / 2, masked_depth_image.shape[0] / 2
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    o3d.io.write_point_cloud(point_cloud_path, pcd)
    o3d.visualization.draw_geometries([pcd])

# Define your paths
color_image_path = 'generated_image.png'
depth_map_path = 'depth_map.png'
point_cloud_path = 'point_cloud.pcd'

# Execute workflow
generate_depth_map(color_image_path, depth_map_path)
mask = segment_image(color_image_path)
masked_depth = apply_mask_to_depth(mask, depth_map_path)
create_point_cloud_from_images(color_image_path, masked_depth, point_cloud_path)
