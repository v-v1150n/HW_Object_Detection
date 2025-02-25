import torch
import torchvision.transforms as T
from PIL import Image
import json
import os
import time
from tqdm import tqdm
import argparse

def get_model(model_path):
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False)
    checkpoint = torch.load(model_path, map_location='cuda') 
    model.load_state_dict(checkpoint['model'])
    model.eval().to('cuda')  
    return model

def detect_objects(model, image_path, threshold=0.5):
    transform = T.Compose([T.Resize((800, 800)), T.ToTensor()]) 
    image = Image.open(image_path)
    img_tensor = transform(image).unsqueeze(0).to('cuda')
    outputs = model(img_tensor)
    
    scores = outputs['pred_logits'].softmax(-1)[0, :, :-1].max(-1)[0]
    keep = scores > threshold  
    
    boxes = outputs['pred_boxes'][0, keep].cpu().detach().numpy()  
    boxes = convert_boxes(boxes, image.width, image.height)
    
    labels = outputs['pred_logits'][0, keep].argmax(-1).cpu().detach().numpy()
    return boxes, labels

def convert_boxes(boxes, img_width, img_height):
    return [[(x_center - 0.5 * width) * img_width,
             (y_center - 0.5 * height) * img_height,
             (x_center + 0.5 * width) * img_width,
             (y_center + 0.5 * height) * img_height]
            for x_center, y_center, width, height in boxes]

def save_detections_to_json(image_ids, all_boxes, all_labels, output_file):
    results = {image_id: {"boxes": boxes, "labels": labels.tolist()}
               for image_id, boxes, labels in zip(image_ids, all_boxes, all_labels)}
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

def process_folder(folder_path, model, output_file, threshold=0.5):
    image_ids, all_boxes, all_labels = [], [], []
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for file_name in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(folder_path, file_name)
        boxes, labels = detect_objects(model, image_path, threshold)
        image_ids.append(file_name)
        all_boxes.append(boxes)
        all_labels.append(labels)
        
        torch.cuda.empty_cache()
    
    save_detections_to_json(image_ids, all_boxes, all_labels, output_file)

def main(args):
    start_time = time.time()  

    model = get_model(args.model_path)

    process_folder(args.folder_path, model, args.output_file, args.threshold)

    total_time = time.time() - start_time
    print(f"Total prediction time: {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DETR object detection on a folder of images.")
    parser.add_argument('--folder_path', type=str, default='/home/vv1150n/113-1_HW/CVDL_HW1/dataset/val2017', help='資料夾路徑')
    parser.add_argument('--model_path', type=str, default='/home/vv1150n/113-1_HW/CVDL_HW1/DETR_output_150epochs/checkpoint.pth',help='模型檔案路徑')
    parser.add_argument('--output_file', type=str, default='',  required=True, help='輸出 JSON 檔案名稱')
    parser.add_argument('--threshold', type=float, default=0.95, help='物件偵測分數閾值')
    
    args = parser.parse_args()
    main(args)
