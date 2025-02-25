from PIL import Image
import os
import json
import glob

# 讀取圖像大小的函數
def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        return width, height

# 類別對應（根據你提供的類別列表）
categories = [
        {'id': 0, 'name':'Person'},
        {'id': 1, 'name':'Ear'},
        {'id': 2, 'name':'Earmuffs'},
        {'id': 3, 'name':'Face'},
        {'id': 4, 'name':'Face-guard'},
        {'id': 5, 'name':'Face-mask-medical'},
        {'id': 6, 'name':'Foot'},
        {'id': 7, 'name':'Tools'},
        {'id': 8, 'name':'Glasses'},
        {'id': 9, 'name':'Gloves'},
        {'id': 10, 'name':'Helmet'},
        {'id': 11, 'name':'Hands'},
        {'id': 12, 'name':'Head'},
        {'id': 13, 'name':'Medical-suit'},
        {'id': 14, 'name':'Shoes'},
        {'id': 15, 'name':'Safety-suit'},
        {'id': 16, 'name':'Safety-vest'},
    ]

# COCO 格式模板
coco_format = {
    "images": [],
    "annotations": [],
    "categories": categories
}

annotation_id = 0

def convert_yolo_to_coco(yolo_label_path, image_id, image_width, image_height):
    global annotation_id
    annotations = []

    with open(yolo_label_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            # YOLO 標籤格式：class, x_center, y_center, width, height
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1]) * image_width
            y_center = float(parts[2]) * image_height
            width = float(parts[3]) * image_width
            height = float(parts[4]) * image_height

            # 轉換為 COCO 格式的 [x_min, y_min, width, height]
            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0
            }
            annotations.append(annotation)
            annotation_id += 1
    return annotations

def find_image_file(image_basename, image_dir):
    # 嘗試不同的圖像格式
    for ext in ['.jpeg', '.jpg', '.png']:
        image_path = os.path.join(image_dir, image_basename + ext)
        if os.path.exists(image_path):
            return image_path
    return None

def process_images_and_labels(label_dir, image_dir, output_json):
    image_id = 0

    for label_path in glob.glob(os.path.join(label_dir, "*.txt")):
        # 從標籤檔案中取得基底名稱
        image_basename = os.path.splitext(os.path.basename(label_path))[0]

        # 尋找對應的圖像檔案
        image_path = find_image_file(image_basename, image_dir)

        if image_path:
            # 自動讀取圖像大小
            image_width, image_height = get_image_size(image_path)

            image = {
                "id": image_id,
                "file_name": os.path.basename(image_path),
                "width": image_width,
                "height": image_height
            }

            coco_format['images'].append(image)

            # 轉換 YOLO 標籤為 COCO 格式
            annotations = convert_yolo_to_coco(label_path, image_id, image_width, image_height)
            coco_format['annotations'].extend(annotations)

            image_id += 1
        else:
            print(f"圖像未找到: {image_basename}")

    # 保存為 COCO 格式的 JSON
    with open(output_json, 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)

# 主函數
label_directory = 'dataset/labels/val'
image_directory = 'dataset/images/val'
output_coco_json = 'dataset/instances_val2017.json'
process_images_and_labels(label_directory, image_directory, output_coco_json)
