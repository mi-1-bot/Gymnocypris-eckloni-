import os
import xml.etree.ElementTree as ET

# 配置路径
xml_dir = 'D:\Deep Learning\YOLOv11\MACHAO\hua\Annotations'       # XML 文件夹路径
txt_dir = 'D:\Deep Learning\YOLOv11\MACHAO\hua\labels'       # 输出 TXT 文件夹路径
class_names = ["1", "2", "3", "4", "5", "6", "7", "8"]  # 类别名称列表，按顺序编号

# 创建输出文件夹
os.makedirs(txt_dir, exist_ok=True)

def convert(xml_path, txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    with open(txt_path, 'w') as f:
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in class_names:
                continue
            class_id = class_names.index(name)

            box = obj.find('bndbox')
            xmin = float(box.find('xmin').text)
            ymin = float(box.find('ymin').text)
            xmax = float(box.find('xmax').text)
            ymax = float(box.find('ymax').text)

            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

# 批量转换
for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith('.xml'):
        continue
    xml_path = os.path.join(xml_dir, xml_file)
    txt_path = os.path.join(txt_dir, xml_file.replace('.xml', '.txt'))
    convert(xml_path, txt_path)