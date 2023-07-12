from lib import *
from config import *
from image_transform import *
from utils import *

class_index = ['car','motobike','bike']

class Predictor():
    def __init__(self,class_index):
        self.clas_index = class_index

    def predict_max(self, output): # [0.9, 1]
        max_id = np.argmax(output.detach().numpy())
        predicted_label = self.clas_index[max_id]
        return predicted_label

predictor = Predictor(class_index)

def predict(img):
    #prepare network
    use_predicted = True
    net = models.vgg16(weights= use_predicted)
    net.classifier[6]=nn.Linear(in_features = 4096,out_features = 3)
    net.eval()

    #prepare model
    model = load_model(net,save_path)

    #prepare model
    transform = ImageTransform(size,mean,std)
    img = transform(img,phase='test')
    img = img.unsqueeze_(0)

    #predict
    output = model(img)
    response = predictor.predict_max(output)

    return response