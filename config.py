from lib import *

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

size=224
mean = (0.485, 0.456, 0.406) 
std = (0.229, 0.224, 0.225)

batch_size = 4
num_epochs = 4
save_path='./weight_transfer_classifier6.pth'