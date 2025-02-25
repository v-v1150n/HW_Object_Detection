
# DETR Model Training and Inference

## 1. Download the DETR repository

Clone the official [DETR GitHub](https://github.com/facebookresearch/detr) repository and install the necessary dependencies:

```bash
git clone https://github.com/facebookresearch/detr.git
cd detr
conda install -c pytorch pytorch torchvision
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

## 2. Modify the code
Change the num_classes in `detr/models/detr.py` to 17 to match the number of custom object classes.

## 3. Download the pre-trained model
Download the required pre-trained model from the [Model Zoo](https://github.com/facebookresearch/detr) such as `detr-r50-e632da11.pth`.

## 4. Data preprocessing
Convert the YOLO formatted train data and valid data to COCO format. 
Use the `yolo_to_coco.py` script to perform the conversion and save the files as `instances_train2017.json` & `instances_val2017.json`. 
The folder structure should be as follows:

```bash
path/to/coco/
  ├── annotations/  # annotation json files (instances_train2017.json & instances_val2017.json)
  ├── train2017/    # train images
  └── val2017/      # val images
```

## 5. Start training
Run the following command in the DETR directory to start training the model:

```bash
python main.py  --batch_size 8 --epochs 150 \
                --coco_path /home/vv1150n/113-1_HW/CVDL_HW1/dataset \
                --output_dir /home/vv1150n/113-1_HW/CVDL_HW1/DETR_output_150epochs \
                --resume /home/vv1150n/113-1_HW/CVDL_HW1/checkpoint_file/detr-r50-e632da11.pth \
                --num_workers 4
```

```bash
python main.py  --batch_size 1 --start_epoch 151 \
                --epochs 201 --lr 1e-5 --num_workers 4 \
                --coco_path /home/vv1150n/113-1_HW/CVDL_HW1/dataset \                             
                --output_dir /home/vv1150n/113-1_HW/CVDL_HW1/DETR_output_150epoch_50epochs \
                --resume /home/vv1150n/113-1_HW/CVDL_HW1/DETR_output_150epochs/checkpoint.pth \
                
```

## 6. Evaluation
Run the following command in the DETR directory to start evaluation the model:

```bash
python main.py  --batch_size 8 --no_aux_loss --eval \
                --resume /home/vv1150n/113-1_HW/CVDL_HW1/DETR_output_150epochs/checkpoint.pth \
                --coco_path /home/vv1150n/113-1_HW/CVDL_HW1/dataset \
                --num_workers 4
```
## 7. Inference on the Validation set
Use inference.py to make predictions on the validation set, and save the output as `valid_<student-id>.json`. 
You can adjust the prediction threshold as needed:

```bash
python inference.py --folder_path ./valid/images \
                    --model_path /home/vv1150n/113-1_HW/CVDL_HW1/DETR_output_150epochs/checkpoint.pth \
                    --output_file valid_r12945042.json \
                    --threshold 0.95               
```

## 8. Evaluation on the Validation set
Use `eval_new.py` to evaluate the model's predictions on the validation set. 
Provide the paths for `val_submission.json` and `valid_target.json` to get the mAP (50-95) evaluation results:

```bash
python eval_new.py valid_r12945042.json valid_target.json 
```

## 9. Inference on the Test set
Use `inference.py` to make predictions on the test set and save the output as `test_<student-id>.json`:

```bash
python inference.py --folder_path ./test/images \                 
                    --model_path /home/vv1150n/113-1_HW/CVDL_HW1/DETR_output_150epoch_50epochs/checkpoint.pth \     
                    --output_file test_r12945042.json \           
                    --threshold 0.95  
```

## 10. Visualizing Object Detection Results
Run `visualization.py`, enter the image file name, and the final result will be output as `visualization_result.png`.

```bash
python visualization.py
```