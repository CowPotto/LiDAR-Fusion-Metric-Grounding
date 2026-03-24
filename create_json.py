import numpy as np
import os
import glob
import json


def create_dataset_json(base_dir, output_json="dataset.json"):
    print("Đang tổng hợp dữ liệu thành file JSON cho 3DGS...")

    # Lấy danh sách file
    rgb_files = sorted(glob.glob(os.path.join(base_dir, "images", "*.jpg")))

    dataset_info = {
        "camera_model": "PINHOLE",
        "frames": []
    }

    for idx, rgb_path in enumerate(rgb_files):
        base_name = os.path.splitext(os.path.basename(rgb_path))[0]

        # Đường dẫn tương đối (để file JSON dễ di chuyển)
        rel_rgb_path = f"images/{base_name}.jpg"
        rel_depth_path = f"depths/{base_name}.png"
        npz_path = os.path.join(base_dir, "images", f"{base_name}.npz")

        if not os.path.exists(npz_path):
            continue

        # Đọc thông số camera
        cam_data = np.load(npz_path)
        intrinsic = cam_data['camera_intrinsics']
        c2w = cam_data['camera_pose']

        # Trích xuất tiêu cự và tâm
        fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])

        # Đưa ma trận c2w về dạng list để lưu JSON
        c2w_list = c2w.tolist()

        # Thêm vào danh sách frame
        frame_data = {
            "file_path": rel_rgb_path,
            "depth_path": rel_depth_path,
            "transform_matrix": c2w_list,
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy
        }
        dataset_info["frames"].append(frame_data)

    # Ghi ra file JSON
    json_path = os.path.join(base_dir, output_json)
    with open(json_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)

    print(f"Thành công! Đã tạo file {output_json} chứa {len(dataset_info['frames'])} frames.")
    print(f"Đường dẫn file: {json_path}")


if __name__ == "__main__":
    # Thay đường dẫn thư mục dataset của bạn vào đây
    base_dir = r"D:\gaussian_splat\ScanNet++ data\data\scannet_test\scene0686_01"
    create_dataset_json(base_dir)