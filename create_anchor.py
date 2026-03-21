import numpy as np
import cv2
import open3d as o3d
import os


def create_point_cloud_from_rgbd(rgb_path, depth_path, npz_path, output_ply="init_anchor_frame0.ply"):
    print("Đang xử lý dữ liệu để tạo mỏ neo 3D...")

    # 1. Đọc ảnh RGB
    color_image = cv2.imread(rgb_path)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    h, w, _ = color_image.shape

    # 2. Đọc ảnh Depth (Ảnh chiều sâu 16-bit)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    # Đưa độ sâu về đơn vị mét (m)
    depth_scale = 1000.0
    depth_in_meters = depth_image.astype(np.float32) / depth_scale

    # 3. Đọc thông số Camera từ .npz với các keys CHÍNH XÁC
    cam_data = np.load(npz_path)

    intrinsic = cam_data['camera_intrinsics']  # <--- ĐÃ SỬA ĐÚNG KEY
    c2w = cam_data['camera_pose']  # <--- ĐÃ SỬA ĐÚNG KEY

    # Lấy tiêu cự (fx, fy) và tâm quang học (cx, cy)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # 4. Thuật toán Unprojection (Từ 2D -> 3D Camera Space)
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Lọc bỏ các pixel không có dữ liệu chiều sâu (depth == 0)
    valid = depth_in_meters > 0
    u = u[valid]
    v = v[valid]
    z = depth_in_meters[valid]
    colors = color_image[valid]

    # Tính tọa độ x, y trong không gian camera
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Tập hợp thành các điểm 3D trong Camera Space (x, y, z, 1)
    points_cam = np.vstack((x, y, z, np.ones_like(z)))

    # 5. Chuyển từ Camera Space ra World Space (Không gian thực)
    # Nhân ma trận c2w với các điểm để đặt chúng vào đúng vị trí trong thế giới
    points_world = (c2w @ points_cam)[:3, :].T

    # 6. Xuất ra file .ply bằng Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Open3D nhận màu từ 0-1

    o3d.io.write_point_cloud(output_ply, pcd)
    print(f"Thành công! Đã lưu mỏ neo Point Cloud 1:1 tại: {output_ply}")
    print(f"Tổng số điểm 3D: {len(points_world)}")


if __name__ == "__main__":
    # Đường dẫn thư mục
    base_dir = r"D:\gaussian_splat\ScanNet++ data\data\scannet_test\scene0686_01"

    rgb_file = os.path.join(base_dir, "images", "000000.jpg")
    depth_file = os.path.join(base_dir, "depths", "000000.png")
    npz_file = os.path.join(base_dir, "images", "000000.npz")

    create_point_cloud_from_rgbd(rgb_file, depth_file, npz_file, "init_anchor_frame0.ply")