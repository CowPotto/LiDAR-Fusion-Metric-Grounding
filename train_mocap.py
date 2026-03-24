import os
import json
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData
from torch.utils.data import Dataset, DataLoader

# Kiểm tra xem gsplat đã sẵn sàng chưa
try:
    from gsplat import rasterization

    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("Cảnh báo: Không tìm thấy thư viện gsplat. Xin vui lòng cài đặt bằng 'pip install gsplat'.")


# ==========================================
# 1. BỘ ĐỌC DỮ LIỆU TỪ JSON VÀ THƯ MỤC SCANNET++
# ==========================================
class ScanNetDataset(Dataset):
    def __init__(self, json_path, base_dir):
        with open(json_path, 'r') as f:
            self.meta = json.load(f)
        self.base_dir = base_dir
        self.frames = self.meta['frames']
        print(f"[*] Đã tải thành công {len(self.frames)} frames từ dataset.json")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]

        # 1. Đọc ảnh RGB và chuẩn hóa về [0, 1], đưa lên GPU
        img_path = os.path.join(self.base_dir, frame['file_path'])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image, dtype=torch.float32, device="cuda") / 255.0

        # 2. Đọc ảnh Depth LiDAR (đơn vị mm) -> Chuyển sang mét (m) cho chuẩn tỷ lệ 1:1
        depth_path = os.path.join(self.base_dir, frame['depth_path'])
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh depth: {depth_path}")
        depth_tensor = torch.tensor(depth.astype(np.float32) / 1000.0, dtype=torch.float32, device="cuda")

        # 3. Xử lý Ma trận Camera
        c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32, device="cuda")
        w2c = torch.linalg.inv(c2w)  # Chuyển từ Camera-to-World sang World-to-Camera

        # Tạo ma trận nội (Intrinsics Matrix - K) 3x3
        K = torch.tensor([
            [frame['fl_x'], 0, frame['cx']],
            [0, frame['fl_y'], frame['cy']],
            [0, 0, 1]
        ], dtype=torch.float32, device="cuda")

        camera_data = {
            'w2c': w2c,
            'intrinsics': K
        }

        return camera_data, image_tensor, depth_tensor


# ==========================================
# 2. MÔ HÌNH 3D GAUSSIAN KHỞI TẠO TỪ LiDAR ANCHOR
# ==========================================
class GaussianModel(nn.Module):
    def __init__(self, ply_path):
        super().__init__()
        print(f"[*] Đang nạp Mỏ neo hình học từ: {ply_path}")

        plydata = PlyData.read(ply_path)
        vtx = plydata['vertex']

        positions = np.vstack([vtx['x'], vtx['y'], vtx['z']]).T
        colors = np.vstack([vtx['red'], vtx['green'], vtx['blue']]).T / 255.0

        num_points = positions.shape[0]
        print(f"[*] Đã đẩy {num_points} hạt Gaussian lên GPU để huấn luyện!")

        # Chuyển đổi thành tham số có thể học được (Learnable Parameters)
        self.means = nn.Parameter(torch.tensor(positions, dtype=torch.float32, device="cuda"))
        self.colors = nn.Parameter(torch.tensor(colors, dtype=torch.float32, device="cuda"))

        # Khởi tạo kích thước hạt = 0.01m (1cm - bằng với voxel_size lúc nén)
        scales = np.ones((num_points, 3)) * np.log(0.01)
        self.scales = nn.Parameter(torch.tensor(scales, dtype=torch.float32, device="cuda"))

        # Khởi tạo Quaternion góc xoay mặc định (1, 0, 0, 0)
        quats = np.zeros((num_points, 4))
        quats[:, 0] = 1.0
        self.quats = nn.Parameter(torch.tensor(quats, dtype=torch.float32, device="cuda"))

        # Độ đục mặc định = 0.1 (hơi trong suốt)
        opacities = np.ones((num_points, 1)) * -2.197
        self.opacities = nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device="cuda"))


# ==========================================
# 3. HÀM HUẤN LUYỆN & TÍNH JOINT LOSS (LiGSM Core)
# ==========================================
def train_step(model, optimizer, camera_data, gt_image, gt_depth):
    optimizer.zero_grad()  # Xóa rác đạo hàm vòng lặp trước

    if not GSPLAT_AVAILABLE:
        raise RuntimeError("Thư viện gsplat chưa được load. Dừng chương trình.")

    # [BẢN CẬP NHẬT GSPLAT V1.0+]: Thêm chiều batch (unsqueeze(0)) cho camera
    viewmats = camera_data['w2c'].unsqueeze(0)  # Shape: [1, 4, 4]
    Ks = camera_data['intrinsics'].unsqueeze(0)  # Shape: [1, 3, 3]

    # 1. FORWARD PASS: Dùng gsplat để vẽ hạt 3D thành ảnh 2D
    render_rgb, render_alpha, meta = rasterization(
        means=model.means,
        quats=model.quats,
        scales=torch.exp(model.scales),
        opacities=torch.sigmoid(model.opacities).squeeze(-1),  # <--- THÊM .squeeze(-1) VÀO ĐÂY
        colors=model.colors,
        viewmats=viewmats,
        Ks=Ks,
        width=gt_image.shape[1],
        height=gt_image.shape[0]
    )

    # Do gsplat render theo batch, ta lấy ảnh đầu tiên [0] ra
    render_rgb = render_rgb[0]

    # Lấy bản đồ chiều sâu từ meta (tên key bản mới là 'depths')
    if "depths" in meta:
        render_depth = meta["depths"][0].squeeze(-1)
    else:
        render_depth = torch.zeros_like(gt_depth)

    # 2. TÍNH JOINT LOSS (Điểm ăn tiền của bài Paper)
    loss_rgb = F.l1_loss(render_rgb, gt_image)

    # Loss chiều sâu (Ép tỷ lệ 1:1 theo LiDAR dToF)
    valid_depth_mask = (gt_depth > 0)
    if valid_depth_mask.any():
        loss_depth = F.l1_loss(render_depth[valid_depth_mask], gt_depth[valid_depth_mask])
    else:
        loss_depth = torch.tensor(0.0, device="cuda")

    lambda_depth = 0.2  # Trọng số cho độ sâu
    total_loss = loss_rgb + lambda_depth * loss_depth

    # 3. BACKWARD PASS & CẬP NHẬT TỌA ĐỘ
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), loss_rgb.item(), loss_depth.item()


# ==========================================
# 4. VÒNG LẶP HUẤN LUYỆN CHÍNH (MAIN PIPELINE)
# ==========================================
if __name__ == "__main__":
    print("\n--- BẮT ĐẦU PIPELINE 3DGS CHO MOCAP ---")

    # !!! CHÚ Ý: BẠN CHỈ CẦN SỬA ĐƯỜNG DẪN NÀY CHO KHỚP VỚI MÁY CỦA BẠN !!!
    base_directory = r"D:\gaussian_splat\ScanNet++ data\data\scannet_test\scene0686_01"

    ply_path = os.path.join(base_directory, "global_anchor.ply")
    json_file = os.path.join(base_directory, "dataset.json")

    # Khởi tạo Mô hình và Bộ đọc dữ liệu
    model = GaussianModel(ply_path)
    dataset = ScanNetDataset(json_file, base_directory)

    # Bộ tối ưu hóa (Optimizer)
    optimizer = torch.optim.Adam([
        {'params': [model.means], 'lr': 0.00016},
        {'params': [model.colors], 'lr': 0.0025},
        {'params': [model.scales], 'lr': 0.005},
        {'params': [model.quats], 'lr': 0.001},
        {'params': [model.opacities], 'lr': 0.05}
    ])

    num_iterations = 200  # Số vòng lặp chạy thử
    print(f"\n[*] Tiến hành huấn luyện {num_iterations} bước...")

    # Bật chế độ huấn luyện cho mô hình
    model.train()

    for i in range(num_iterations):
        # Lấy xoay vòng các frame để train
        frame_idx = i % len(dataset)
        camera_data, gt_image, gt_depth = dataset[frame_idx]

        # Chạy 1 bước tiến và lùi
        total_loss, loss_rgb, loss_depth = train_step(model, optimizer, camera_data, gt_image, gt_depth)

        # In kết quả sau mỗi 10 bước
        if i % 10 == 0:
            print(
                f"Bước {i:04d} (Frame {frame_idx:03d}) | Tổng Loss: {total_loss:.4f} [RGB: {loss_rgb:.4f} | Depth: {loss_depth:.4f}]")

    print("\n--- HOÀN TẤT CHẠY THỬ! MÔ HÌNH ĐÃ HỌC THÀNH CÔNG ---")