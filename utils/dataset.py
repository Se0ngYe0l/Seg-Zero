import glob
import os

import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from utils.transforms import ResizeLongestSide
from PIL import Image

class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            if ds == "ReasonSeg":
                images = glob.glob(
                    os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
                )
                self.images = images
                self.data_type = "reason_seg"
            elif ds == "RoboRefit":
                self.json_data = json.load(open(os.path.join(self.base_image_dir, ds, split, "roborefit_" + split + '.json'), 'r', encoding="utf-8"))
                self.img_path = os.path.join(self.base_image_dir, ds, split, "image")
                self.data_type = "roborefit"
            elif ds == "OCID_VLG":
                self.json_data = json.load(open(os.path.join(self.base_image_dir, ds, "refer", "multiple", split + '_expressions.json'), 'r', encoding="utf-8"))
                self.data_type = "ocid_vlg"

        self.ds = ds
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        elif self.data_type == "roborefit":
            #return len(os.listdir(self.img_path))
            return len(self.json_data)
        elif self.data_type == "ocid_vlg":
            return len(self.json_data["data"])
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        self.h, self.w = x.shape[-2:]
        padh = self.img_size - self.h
        padw = self.img_size - self.w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = Image.open(image_path).convert('RGB')
            # image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.data_type == "roborefit":
            data_info = self.json_data[idx]
            image_path = data_info['rgb_path'].replace('\\', '/').replace('final_dataset','RoboRefit')
            mask_path = data_info['mask_path'].replace('\\', '/').replace('final_dataset','RoboRefit')
            image_path = os.path.join(self.base_image_dir, image_path)
            mask_path = os.path.join(self.base_image_dir, mask_path)
            image_id = data_info['num']
            ann_id = data_info['num']
            image = Image.open(image_path).convert('RGB')
            # image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask_img = cv2.imread(mask_path)

            sampled_sents = [data_info['text']]
        elif self.data_type == "ocid_vlg":
            data_info = self.json_data["data"][idx]
            seq_path, im_name = data_info['image_filename'].split(',')
            image_path = os.path.join(self.base_image_dir, self.ds, seq_path, "rgb", im_name)
            mask_path = os.path.join(self.base_image_dir, self.ds, seq_path, "seg_mask_instances_combi", im_name)

            image = Image.open(image_path).convert('RGB')
            # image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            
            objID = data_info['answer']
            sampled_sents = [data_info['question']]

        # preprocess image for sam
        # image = self.transform.apply_image(image)
        # image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        elif self.data_type == "roborefit":
            if mask_img.shape[-1] == 1:
                masks = np.expand_dims(mask_img, 0)
            else:
                mask_img = np.sum(mask_img, -1)
                mask_img[mask_img>0] = 1
                masks = np.expand_dims(mask_img, 0)
        elif self.data_type == "ocid_vlg":
            masks = np.where(mask_img == objID, True, False)
            masks = np.expand_dims(masks, 0)

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks.astype(np.float32))

        w, h = image.size
        return np.array(image), sampled_sents, masks,image_id, ann_id, w, h
        