import os
import pandas as pd

# 输入路径（存放21个文件夹）
base_path = r'D:\Deep Learning\YOLOv11\ultralytics-8.3.86\runs\detect\experimental data\predict'
# 输出路径
output_file = os.path.join(base_path, 'Raw data summary.xlsx')

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    for dir_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, dir_name)
        if os.path.isdir(folder_path):
            excel_path = os.path.join(folder_path, f"{dir_name}.xlsx")
            if os.path.exists(excel_path):
                try:
                    df = pd.read_excel(excel_path)
                    df.to_excel(writer, sheet_name=dir_name[:31], index=False)  # 注意Excel限制：sheet名最多31字符
                    print(f"已合并: {excel_path}")
                except Exception as e:
                    print(f"处理 {excel_path} 时出错: {e}")

print(f"\n✅ 所有Excel已整合到 {output_file}")
