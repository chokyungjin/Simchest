from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import json
import os
import time
import tqdm

from datasets.datasets_Supcon import ClassPairDataset
from networks.resnet_big import SupConResNet

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def parse_option():
    parser = argparse.ArgumentParser('argument for inference')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--path', type=str, default=None, help='path to custom dataset')
    
    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--pretrained', type=str, default=None)

    opt = parser.parse_args()


def inference(test_loader, model):

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    activation = nn.Sigmoid()
    samples = {'imgs':[], 'patient_name' : [], 'change_labels':[], 'logits':[]}

    with torch.no_grad():
        end = time.time()
        for idx, (base_img_1, pair_img_1, change_labels, labels, name) in tqdm.tqdm(enumerate(test_loader)):
            try:
                gt_label = change_labels[0].cpu().detach().numpy()        
                patient_name = labels[0]
                pairs =[name[0][0], name[1][0]]
                images = torch.cat([base_img_1, pair_img_1], dim=0)
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)

                bsz = 1
                features  = model(images)
                
                f3, f4 = torch.split(features, [bsz, bsz], dim=0)
                cos_sim2 = cos(f3, f4)[0].cpu().detach().numpy()
                after_sigmoid = activation(cos_sim2)

                samples['imgs'].append(pairs)
                samples['patient_name'].append(labels[0])
                samples['change_labels'].append(gt_label)
                samples['logits'].append(after_sigmoid[0].cpu().detach().numpy())
            except Exception as err:
                print(err)

    return samples

def main():

    opt = parse_option()

    dataset = 'real'
    train_path = opt.path

    test_datasets = ClassPairDataset(train_path, dataset=dataset, 
    fov=None, margin=None,  aug=None, mode='test')

    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=opt.batch_size, 
        num_workers=opt.num_workers, pin_memory=True, shuffle=False)

    model = SupConResNet(name=opt.model)
    pretrained = opt.pretrained
    checkpoint = torch.load(pretrained)
    pretrained_dict = checkpoint['model']
    pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    model.load_state_dict(pretrained_dict)

    model.eval()

    json = inference(test_loader, model.cuda())

    json_name = './result.json'
    with open(json_name, "w") as f:
        json.dump(json, f)


if __name__ == '__main__':
    main()