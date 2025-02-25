import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 類別對應表
category_names = {
    0: "Person",
    1: "Ear",
    2: "Earmuffs",
    3: "Face",
    4: "Face-guard",
    5: "Face-mask-medical",
    6: "Foot",
    7: "Tools",
    8: "Glasses",
    9: "Gloves",
    10: "Helmet",
    11: "Hands",
    12: "Head",
    13: "Medical-suit",
    14: "Shoes",
    15: "Safety-suit",
    16: "Safety-vest"
}

# 載入 JSON 文件
# json_path = 'VISUALIZATION_LABEL.json'  # 替換成你的 json 文件路徑
json_path = './test_r12945042.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# 載入圖像
# image_dir = 'VISUALIZATION_IMG'  # 替換成你的圖像文件夾路徑
image_dir = './test/images'  # 替換成你的圖像文件夾路徑

pic_number = input('Enter the picture name (only file name): ')  # 比如 "pexels-photo-5619458"
image_file_name = pic_number + '.jpeg'  # 替換為你的圖像格式
image_path = f"{image_dir}/{image_file_name}"
print('image_path:', image_path)

# 檢查該圖像是否在 JSON 資料中
if image_file_name not in data:
    print(f"image {image_file_name} not found")
else:
    # 載入圖像並顯示
    image = Image.open(image_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(image)

    # 取得該圖像的標註資料
    boxes = data[image_file_name]['boxes']
    labels = data[image_file_name]['labels']

    # 在圖像上繪製邊界框和標籤
    ax = plt.gca()
    for bbox, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min  # 計算寬度
        height = y_max - y_min  # 計算高度
        
        # 繪製邊界框
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # 顯示類別名稱
        category_name = category_names.get(label, 'Unknown')
        plt.text(x_min, y_min - 5, category_name, fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))
        print(f"Categories ID: {label}, Categories Name: {category_name}")

    plt.axis('off')
    plt.savefig('a_visualization_result.png', bbox_inches='tight', pad_inches=0)
    plt.show()


