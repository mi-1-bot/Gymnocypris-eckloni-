import os
import shutil
import random

def split_data(file_path, label_path, new_file_path, train_rate, val_rate):
    images = os.listdir(file_path)
    labels = os.listdir(label_path)

    images_no_ext = {os.path.splitext(image)[0]: image for image in images}
    labels_no_ext = {os.path.splitext(label)[0]: label for label in labels}
    matched_data = [(img, images_no_ext[img], labels_no_ext[img]) for img in images_no_ext if img in labels_no_ext]

    # 打印未匹配情况
    unmatched_images = [img for img in images_no_ext if img not in labels_no_ext]
    unmatched_labels = [label for label in labels_no_ext if label not in images_no_ext]
    if unmatched_images:
        print("⚠️ 未匹配的图片文件:", len(unmatched_images))
    if unmatched_labels:
        print("⚠️ 未匹配的标签文件:", len(unmatched_labels))

    # 打乱数据
    random.shuffle(matched_data)
    total = len(matched_data)

    # 计算划分
    train_data = matched_data[:int(train_rate * total)]
    val_data = matched_data[int(train_rate * total):]

    # === 自动清空输出目录 ===
    if os.path.exists(new_file_path):
        shutil.rmtree(new_file_path)
    os.makedirs(new_file_path, exist_ok=True)

    # 处理训练集
    for img_name, img_file, label_file in train_data:
        old_img_path = os.path.join(file_path, img_file)
        old_label_path = os.path.join(label_path, label_file)
        new_img_dir = os.path.join(new_file_path, 'train', 'images')
        new_label_dir = os.path.join(new_file_path, 'train', 'labels')
        os.makedirs(new_img_dir, exist_ok=True)
        os.makedirs(new_label_dir, exist_ok=True)
        shutil.copy(old_img_path, os.path.join(new_img_dir, img_file))
        shutil.copy(old_label_path, os.path.join(new_label_dir, label_file))

    # 处理验证集
    for img_name, img_file, label_file in val_data:
        old_img_path = os.path.join(file_path, img_file)
        old_label_path = os.path.join(label_path, label_file)
        new_img_dir = os.path.join(new_file_path, 'val', 'images')
        new_label_dir = os.path.join(new_file_path, 'val', 'labels')
        os.makedirs(new_img_dir, exist_ok=True)
        os.makedirs(new_label_dir, exist_ok=True)
        shutil.copy(old_img_path, os.path.join(new_img_dir, img_file))
        shutil.copy(old_label_path, os.path.join(new_label_dir, label_file))

    # 打印数量统计
    print("✅ 数据集划分完成 (8:2)")
    print(f"总样本数: {total}")
    print(f"训练集: {len(train_data)}")
    print(f"验证集: {len(val_data)}")
    print(f"合计: {len(train_data) + len(val_data)}")

if __name__ == '__main__':
    file_path = r"D:\Deep Learning\YOLOv11\MACHAO\hua\images"   # 图片文件夹
    label_path = r"D:\Deep Learning\YOLOv11\MACHAO\hua\labels"  # 标签文件夹
    new_file_path = r"D:\Deep Learning\YOLOv11\MACHAO\VOCdevkit"  # 新数据存放位置
    split_data(file_path, label_path, new_file_path, train_rate=0.8, val_rate=0.2)
