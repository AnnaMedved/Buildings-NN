import os
import numpy as np
import torch
import torch.utils.data
import json 
import pandas as pd 

from PIL import Image
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pad_sequence


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
        with open(coco_json_path, 'r') as f: 
            res_json = json.load(f)
        images_df = pd.DataFrame(res_json['images'])
        anns_df = pd.DataFrame(res_json['annotations'])
        self.anns = anns_df

        # Creating masks process 
        if (len(os.listdir(im_pth)) != len(os.listdir(msks_pth))) | (len(os.listdir(msks_pth)) == 0): 
            # Coco-helper for all dataset:  
            coco = COCO(coco_json_path)
            cat_ids = coco.getCatIds()
            
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
        def box_xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
            """
            Converts bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2) format.
            (x, y) refers to top left of bouding box.
            (w, h) refers to width and height of box.
            Arguments:
                boxes (Tensor[N, 4]): boxes in (x, y, w, h) which will be converted.
            Returns:
                boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format.
            """
            x, y, w, h = boxes.unbind(-1)
            boxes = torch.stack([x, y, x + w, y + h], dim=-1)
            return boxes
        

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
        
        # Getting masks:
        vals_b = np.array([v for v in obj.bbox])
        boxes = torch.from_numpy(vals_b) 
        # Transform initial bboxes to right format: 
        boxes = box_xywh_to_xyxy(boxes)

        # Getting labels:
        vals_l = np.array([np.int64(l) for l in obj.category_id])
        labels = torch.from_numpy(vals_l) 

        # Getting masks:
        masks_t = [torch.tensor(obj[0]) for obj in obj.segmentation]
        vals_m = pad_sequence(masks_t, batch_first=True)
        # Shape of [h, w, num_classses]:
        vals_m = vals_m.unsqueeze_(-1).expand(-1, -1, 2)
                    
        image_id = torch.tensor(idx)

        vals_a = np.array([a for a in obj.area])
        area = torch.from_numpy(vals_a)

        vals_cr = np.array([cr for cr in obj.iscrowd])
        iscrowd =  torch.from_numpy(vals_cr)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["masks"] = vals_m

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)