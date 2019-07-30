import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, utils
from skimage import io, transform

device = 'cuda'

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, img_size, bbox_size):
        assert isinstance(img_size, (int, tuple))
        assert isinstance(bbox_size, (int, tuple))
        self.img_size = img_size
        self.bbox_size = bbox_size

    def __call__(self, sample):
        image = sample["image"]
        bbox_image = sample["bbox_image"]
        '''
        h, w = image.shape[:2]
        if isinstance(self.img_size, int):
            if h > w:
                new_h, new_w = self.img_size * h / w, self.img_size
            else:
                new_h, new_w = self.img_size, self.img_size * w / h
        else:
            new_h, new_w = self.img_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        
        h, w = bbox_image.shape[:2]
        if isinstance(self.bbox_size, int):
            if h > w:
                new_h, new_w = self.bbox_size * h / w, self.bbox_size
            else:
                new_h, new_w = self.bbox_size, self.bbox_size * w / h
        else:
            new_h, new_w = self.bbox_size

        new_h, new_w = int(new_h), int(new_w)
        bb = transform.resize(bbox_image, (new_h, new_w))
        '''
        bb = transform.resize(bbox_image, self.bbox_size)
        img = transform.resize(image, self.img_size)

        return {'image': img, "bbox_image": bb, "params": sample["params"]}
    
class LocalizationDataset(Dataset):
    
    def __init__(self, data_pickle, data_path, transform=None):
        """
        Args:
            data_pickle (string or pandas DataFrame): Path to the pickle file or the DataFrame itself that has all sample parameters.
            data_path (string): Path to data directory.
            transform (callable, optional): Optional transform to be applied on the full image and the image part that is inside the bbox.
        """
        if type(data_pickle) == "str":
            self.whole_data = pickle.load(open(data_pickle, "rb"))
        else:
            self.whole_data = data_pickle
        
        self.data_path = data_path
        if data_path[len(data_path)-1] not in ["/", "\\"]:
            self.data_path += "/"
                                               
        self.transform = transform

    def __len__(self):
        return self.whole_data.shape[0]

    def __getitem__(self, idx):
        params = self.whole_data.iloc[idx,:]
        img_path = self.data_path + "images/" + params["img"] + ".jpg"
        image = io.imread(img_path)
        xmin, ymin, xmax, ymax = params["loc_act"]
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        bbox_image =  image[ymin:(ymax+1), xmin:(xmax+1), :]
                 
        sample = {'image': image, 'bbox_image': bbox_image, 'params': params}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bbox_image = sample["image"], sample["bbox_image"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        bbox_image = bbox_image.transpose((2, 0, 1))
        params = sample["params"]
        loc_rel = torch.FloatTensor(params['loc_rel'])
        embedding = torch.FloatTensor(params['emb_expr'])
        return {"image": torch.from_numpy(image),
                "bbox_image": torch.from_numpy(bbox_image),
                "loc_rel": loc_rel,
                "embedding": embedding,
                "IoU" : params['IoU']
               }

class myModel(nn.Module):
    def __init__(self, hidden_dim_img = 1024, box_data_size = 4, embedding_size=512):
        super(myModel, self).__init__()
        self.img_feature_extractor = models.vgg16(pretrained=True).features
        self.box_feature_extractor = models.vgg16(pretrained=True).features
        '''
        for child in self.img_feature_extractor.children():
            for param in child.parameters():
                param.requires_grad = False
        for child in self.box_feature_extractor.children():
            for param in child.parameters():
                param.requires_grad = False
        '''
        self.conv_img = nn.Conv2d(512, 64, 1)    
        self.conv_box = nn.Conv2d(512, 64, 1)    
        self.linear = torch.nn.Linear(64*7*7*2, 1024 - box_data_size)
        
        self.fcn_img = nn.Sequential(
                                nn.Linear(1024, hidden_dim_img),
                                nn.ReLU(),
                                nn.Linear(hidden_dim_img, embedding_size),
                                nn.Tanh(),
        )
        self.fcn_nl = nn.Sequential(
                        nn.Linear(512, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, embedding_size),
                        nn.Tanh(),
        )


        
    def forward(self, img, box, box_data, embedding):
        img_pre_features = self.img_feature_extractor(img)
        img_features = self.conv_img(img_pre_features)
        img_features = img_features.view(img_features.size(0), -1)
        
        box_pre_features = self.box_feature_extractor(box)
        box_features = self.conv_box(box_pre_features)
        box_features = box_features.view(box_features.size(0), -1)
   
        #print(img_features.shape)
        #print(box_features.shape)
        #print(box_data.shape)

        concat_img = torch.cat((img_features, box_features), 1)
        image_all = self.linear(concat_img)
        concat = torch.cat((image_all, box_data), 1)

        x_img = self.fcn_img(concat)
        img_norm = x_img.norm(p=2, dim=1, keepdim=True)
        x_img_normalized = x_img.div(img_norm.expand_as(x_img))

        x_nl = self.fcn_nl(embedding)
        nl_norm = x_nl.norm(p=2, dim=1, keepdim=True)
        x_nl_normalized = x_nl.div(nl_norm.expand_as(x_nl))

        x = torch.cat((x_img_normalized,x_nl_normalized), 1) 

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
def my_loss(output, iou):
    img_seg, word_seg = torch.split(output, 512, dim=1)
    
    #cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    #cos_sim = cos(img_seg, word_seg)    
    cos_sim = torch.sum(img_seg * word_seg, dim=1)

    dist = (1-cos_sim)/2
    loss = nn.MSELoss()
    target = 1 - iou
    l = loss(dist, target)
    return l

def get_torch_data(sample):
    image = sample['image'].to(device).float()
    bbox_image = sample['bbox_image'].to(device).float()
    loc_rel = sample['loc_rel'].to(device)
    embedding = sample['embedding'].to(device)
    IoU =  sample['IoU'].to(device).float()
    return image, bbox_image, loc_rel, embedding, IoU

