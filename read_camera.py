import numpy as np

# Đường dẫn đến một file .npz bất kỳ trong dataset của bạn
file_path = r"D:\gaussian_splat\ScanNet++ data\data\scannet_test\scene0686_01\images\000000.npz"


def inspect_npz(path):
    print(f"Đang đọc file: {path}")
    try:
        # Load dữ liệu từ file .npz
        data = np.load(path)

        # In ra tất cả các "chìa khóa" (keys) được lưu trong file
        print("\nDanh sách các mảng dữ liệu (keys) bên trong:")
        for key in data.files:
            array_data = data[key]
            print(f" - Key: '{key}' | Kích thước (Shape): {array_data.shape} | Kiểu dữ liệu: {array_data.dtype}")

            # In thử ma trận ra để xem trước hình thù
            print(f"   Giá trị:\n{array_data}\n")

    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")


if __name__ == "__main__":
    inspect_npz(file_path)