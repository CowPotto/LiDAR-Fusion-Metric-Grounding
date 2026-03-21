import numpy as np
import cv2
import open3d as o3d
import os
import glob


def fuse_multiple_frames(base_dir, max_frames=1000, output_ply="global_anchor.ply"):
    print(f"Bắt đầu gộp dữ liệu từ {base_dir}...")

    # Tìm tất cả các file ảnh (sắp xếp theo thứ tự)
    rgb_files = sorted(glob.glob(os.path.join(base_dir, "images", "*.jpg")))

    # Giới hạn số lượng frame để test cho nhanh (bạn có thể tăng lên sau)
    rgb_files = rgb_files[:max_frames]

    all_points = []
    all_colors = []

    depth_scale = 1000.0

    for idx, rgb_path in enumerate(rgb_files):
        # Lấy tên gốc của file (vd: '000010')
        base_name = os.path.splitext(os.path.basename(rgb_path))[0]

        depth_path = os.path.join(base_dir, "depths", f"{base_name}.png")
        npz_path = os.path.join(base_dir, "images", f"{base_name}.npz")

        # Bỏ qua nếu thiếu file tương ứng
        if not (os.path.exists(depth_path) and os.path.exists(npz_path)):
            continue

        print(f"Đang xử lý frame {idx + 1}/{len(rgb_files)}: {base_name}")

        # 1. Đọc ảnh và độ sâu
        color_image = cv2.imread(rgb_path)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        h, w, _ = color_image.shape

        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth_in_meters = depth_image.astype(np.float32) / depth_scale

        # 2. Đọc Camera params
        cam_data = np.load(npz_path)
        intrinsic = cam_data['camera_intrinsics']
        c2w = cam_data['camera_pose']

        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        # 3. Unprojection
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        valid = depth_in_meters > 0
        u, v, z = u[valid], v[valid], depth_in_meters[valid]
        colors = color_image[valid]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points_cam = np.vstack((x, y, z, np.ones_like(z)))
        points_world = (c2w @ points_cam)[:3, :].T

        # Lưu trữ điểm và màu của frame hiện tại vào mảng tổng
        all_points.append(points_world)
        all_colors.append(colors / 255.0)

    # Gộp tất cả các mảng numpy lại
    print("\nĐang nối các frame lại với nhau...")
    global_points = np.vstack(all_points)
    global_colors = np.vstack(all_colors)

    # Tạo Point Cloud tổng
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(global_points)
    pcd.colors = o3d.utility.Vector3dVector(global_colors)

    print(f"Tổng số điểm trước khi nén: {len(global_points)}")

    # 4. Tối ưu hóa (Voxel Downsampling) - Cực kỳ quan trọng
    # Gộp các điểm nằm trong cùng một khối lập phương cỡ 1cm (0.01m) thành 1 điểm
    voxel_size = 0.01
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)

    print(f"Tổng số điểm SAU khi nén (độ phân giải {voxel_size}m): {len(pcd_downsampled.points)}")

    # 5. Xuất file
    o3d.io.write_point_cloud(output_ply, pcd_downsampled)
    print(f"Thành công! Mỏ neo toàn cục đã được lưu tại: {output_ply}")


if __name__ == "__main__":
    # Thay đường dẫn cụ thể trên máy của bạn
    base_dir = r"D:\gaussian_splat\ScanNet++ data\data\scannet_test\scene0686_02"

    # Thử gộp 50 frames đầu tiên (đủ để thấy một góc phòng rộng)
    fuse_multiple_frames(base_dir, max_frames=1000, output_ply="global_anchor.ply")