import os
import numpy as np
import torch
import torch.utils.data
import json 
import pandas as pd 

from PIL import Image
from pycocotools.coco import COCO


class FacadesDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, coco_json_path:str,  transforms=None):
        self.root = data_path
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        
        im_pth = os.path.join(data_path, "images")
        msks_pth = os.path.join(data_path, "masks")

        # Checkiing all paths & create masks if '/masks' is empty 
        if not os.path.exists(im_pth): 
            os.makedirs(im_pth)

        if not os.path.exists(msks_pth): 
            os.makedirs(msks_pth)

        self.imgs = list(sorted(os.listdir(os.path.join(data_path, "images"))))

        # Creating masks process (if is empty): 
        if (len(os.listdir(im_pth)) != len(os.listdir(msks_pth))) | (len(os.listdir(msks_pth)) == 0): 
            
            with open(coco_json_path, 'r') as f: 
                res_json = json.load(f)
            images_df = pd.DataFrame(res_json['images'])

            # Coco-helper for all dataset:  
            coco = COCO(coco_json_path)
            cat_ids = coco.getCatIds()

            anns_df = pd.DataFrame(res_json['annotations'])
            self.anns = anns_df
            
            # Process for all imgs: 
            for id in images_df.id.unique():
                # Getting image from coco-object:
                img = coco.imgs[id]
                # Getting all the IDs from image:
                anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=False)
                anns = coco.loadAnns(anns_ids)

                # Get mask for all annotation in image: 
                mask = coco.annToMask(anns[0])
                for i in range(len(anns)):
                    mask += coco.annToMask(anns[i])
                
                im = Image.fromarray(mask)
                im.putpalette([
                    # Use a few colors in case of many houses on the photos 
                    # (only graphical view and saving)
                    0, 0, 0, # black background
                    0, 255, 0, 
                    0, 100, 0, 
                    255, 255, 0, 
                    154, 205, 50, 
                    154, 205, 50
                ])
                mask_n = os.path.splitext(img['file_name'][7:])[0] # (name, ext)[0]
                mask_fn = os.path.join(msks_pth, mask_n + '_mask.png')
                im.save(mask_fn)

        self.masks = list(sorted(os.listdir(os.path.join(data_path, "masks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        anns = self.anns
        obj = anns[anns['image_id']==idx]
        # obj_ids = obj.id # Series of masks

        # # note that we haven't converted the mask to RGB,
        # # because each color corresponds to a different instance
        # # with 0 being background
        # mask = Image.open(mask_path)
        # mask = np.array(mask)
        # # instances are encoded as different colors
        # obj_ids = np.unique(mask)

        # # first id is the background, so remove it
        # obj_ids = obj_ids[1:]

        # # split the color-encoded mask into a set
        # # of binary masks
        # masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        # num_objs = len(obj_ids)
        # boxes = []
        # for i in range(num_objs):
        #     pos = np.where(masks[i])
        #     xmin = np.min(pos[1])
        #     xmax = np.max(pos[1])
        #     ymin = np.min(pos[0])
        #     ymax = np.max(pos[0])
        #     boxes.append([xmin, ymin, xmax, ymax])

        boxes = np.array([obj.bbox]) #torch.as_tensor(
            # np.array([obj.bbox]), dtype=torch.float32
            # )
        labels = np.array([obj.category_id]) #  torch.tensor(
            # , dtype=torch.int64
            # )
        masks = torch.tensor(
            [obj.segmentation], dtype=torch.int32
            )
        image_id = torch.tensor(idx)
        area = torch.as_tensor(obj.area, dtype=torch.float32)
        iscrowd =  torch.tensor(obj.iscrowd, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)