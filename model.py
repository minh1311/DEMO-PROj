from lib import *
from config import *
from image_transform import *
from dataset import *
from utils import *

def model():

    train_list=make_DataPath_List('train')
    val_list= make_DataPath_List('val')

    train_dataset = MyDataset(train_list,transform=ImageTransform(size,mean,std),phase='train')       
    val_dataset = MyDataset(val_list,transform=ImageTransform(size,mean,std),phase='val')

    # train_loader=torch.utils.data.DataLoader(train_dataset,batch_size,shuffle='False')
    # val_loader=torch.utils.data.DataLoader(val_dataset,batch_size,shuffle='False')
    # dataloader_dict={'train':train_loader,'val':val_loader}

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle='False')
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size,shuffle='False')
    dataloader_dict={'train':train_dataloader,'val':val_dataloader}

    use_pretrained = True
    net = models.vgg16(weights=use_pretrained)
    net.classifier[6]=nn.Linear(in_features=4096,out_features=3)

    criteror = nn.CrossEntropyLoss()
   
   
    params = params_to_update(net)
    optimizer = optim.SGD(params, lr=0.0001, momentum=0.9)

    train_model(net, dataloader_dict, criteror, optimizer, num_epochs)