import argparse
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from qwen_vl_utils import process_vision_info
import torch
import json
from PIL import Image as PILImage
from tqdm import tqdm
import os
import re
import numpy as np

from utils.dataset import ValDataset
from torchvision.transforms.functional import to_pil_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="Ricky06662/Seg-Zero-7B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--num_parts", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=50)
    return parser.parse_args()

def extract_bbox_points_think(output_text, x_factor, y_factor):
    json_pattern = r'{[^}]+}'  # 匹配最简单的JSON对象
    json_match = re.search(json_pattern, output_text)
    # pdb.set_trace()
    if json_match:
        data = json.loads(json_match.group(0))
        # 查找bbox键
        bbox_key = next((key for key in data.keys() if 'bbox' in key.lower()), None)
        # pdb.set_trace()
        if bbox_key and len(data[bbox_key]) == 4:
            content_bbox = data[bbox_key]
            content_bbox = [round(int(content_bbox[0])*x_factor), round(int(content_bbox[1])*y_factor), round(int(content_bbox[2])*x_factor), round(int(content_bbox[3])*y_factor)]
        # 查找points键
        points_keys = [key for key in data.keys() if 'points' in key.lower()][:2]  # 获取前两个points键
        if len(points_keys) == 2:
            point1 = data[points_keys[0]]
            point2 = data[points_keys[1]]
            point1 = [round(int(point1[0])*x_factor), round(int(point1[1])*y_factor)]
            point2 = [round(int(point2[0])*x_factor), round(int(point2[1])*y_factor)]
            points = [point1, point2]
    
    think_pattern = r'<think>([^<]+)</think>'
    think_match = re.search(think_pattern, output_text)
    if think_match:
        think_text = think_match.group(1)
    
    return content_bbox, points, think_text

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection, union

def main():
    args = parse_args()
    
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)

    reasoning_model.eval()

    # default processer
    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")
    
    resize_size = 840
    # dataset = load_from_disk(args.test_data_path)['test']
    
    val_dataset = ValDataset(
    "/mnt/e/Task_Infer_VLM/VLM_dataset/",   
    'RoboRefit|testA',   
    1024,    
    )
        
    dataset = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
        )
    
    total_len = len(val_dataset)
    part_size = total_len // args.num_parts
    start_idx = args.idx * part_size
    end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len
    
    QUESTION_TEMPLATE = \
        "Please find '{Question}' with bbox and points." \
        "Compare the difference between objects and find the most closely matched one." \
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
        "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
        "i.e., <think> thinking process here </think>" \
        "<answer>{Answer}</answer>"
    
    messages = []
    id_list = []
    print("load dataset")
    for item in tqdm(dataset):
        image, sentence, mask, image_id, ann_id, w, h = item
        #print(image_id, ann_id, w, h)
        #print(list(sentence[0])[0])
        message = [{
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": to_pil_image(image[0].permute(2,0,1)).resize((resize_size, resize_size), PILImage.BILINEAR)
                },
                {   
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=list(sentence[0])[0].lower().strip("."), 
                                                      Answer="{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}")
                }
            ]
        }]
        messages.append(message)
        id_list.append({
            "image_id": image_id.item(),
            "ann_id": ann_id.item(),
            "image": to_pil_image(image[0].permute(2,0,1)),
            "mask": mask,
            "img_height": h.item(),
            "img_width": w.item()
        })

    all_outputs = []
    for i in tqdm(range(0, len(messages), args.batch_size)):
        batch_messages = messages[i:i + args.batch_size]
        batch_id_list = id_list[i:i + args.batch_size]
        
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference: Generation of the output
        generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # pdb.set_trace()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for id_idx in range(len(batch_output_text)):
                try:
                    bbox, points, think = extract_bbox_points_think(
                                            batch_output_text[id_idx], 
                                            batch_id_list[id_idx]["img_width"]/resize_size, 
                                            batch_id_list[id_idx]["img_height"]/resize_size
                                        )
                    segmentation_model.set_image(batch_id_list[id_idx]["image"])
                    masks, scores, _ = segmentation_model.predict(
                        point_coords=points,
                        point_labels=[1,1],
                        box=bbox
                    )
                    sorted_ind = np.argsort(scores)[::-1]
                    masks = masks[sorted_ind]
                    mask = masks[0].astype(bool)
                    #################
                    '''
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(8, 4))
                    plt.subplot(1, 2, 1)
                    plt.imshow(batch_id_list[id_idx]["image"])
                    plt.title('Original Image')
                        
                    plt.subplot(1, 2, 2)
                    plt.imshow(batch_id_list[id_idx]["image"], alpha=0.6)
                    mask_overlay = np.zeros_like(batch_id_list[id_idx]["image"])
                    mask_overlay[mask == 1] = [255, 0, 0]
                    plt.imshow(mask_overlay, alpha=0.4)
                    plt.title('Image with Predicted Mask')
                    plt.show()
                        
                    plt.tight_layout()
                    plt.savefig(args.output_path)
                    plt.close()
                    '''
                    #################
                    gt_mask = np.array(batch_id_list[id_idx]["mask"])
                    # pdb.set_trace()
                    try:
                        intersection, union = compute_iou(mask, gt_mask)
                    except Exception as e:
                        # skip this because the image or mask is not correct
                        continue
                except Exception as e:
                    # add penalty in this situation
                    think = ""
                    intersection = 0
                    union = np.array(batch_id_list[id_idx]["mask"]).sum()
                print(intersection, union, intersection / union)
                all_outputs.append({
                    "image_id": batch_id_list[id_idx]["image_id"],
                    "ann_id": batch_id_list[id_idx]["ann_id"],
                    "think": think,
                    "intersection": int(intersection),
                    "union": int(union)
                })
        #print(f"Processed batch {i//args.batch_size + 1}/{(len(messages) + args.batch_size - 1)//args.batch_size}")
        
        # clean GPU memory
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

    
    # Modify the output file name, add idx
    output_file = os.path.join(args.output_path, f"output_{args.idx}.json")
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
