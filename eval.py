import os
import argparse
import string
from tqdm import tqdm
import numpy as np
import cv2
import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model
#from tensorflow.keras.models.Model import load_weights

from utils import pad_image, resize_image, create_result_subdir
from STN.spatial_transformer import SpatialTransformer
from models import CRNN, CRNN_STN

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./result/000/prediction_model.050.hdf5')
parser.add_argument('--data_path', type=str, default='./data/test')
parser.add_argument('--gpus', type=int, nargs='*', default=[4])
parser.add_argument('--characters_file', type=str, default='./characters.txt')
parser.add_argument('--characters', type=list, default=[])
parser.add_argument('--label_len', type=int, default=32)
parser.add_argument('--nb_channels', type=int, default=1)
parser.add_argument('--width', type=int, default=300)
parser.add_argument('--height', type=int, default=50)
parser.add_argument('--model', type=str, default='CRNN', choices=['CRNN_STN', 'CRNN'])
parser.add_argument('--conv_filter_size', type=int, nargs=7, default=[64, 128, 256, 256, 512, 512, 512])
parser.add_argument('--lstm_nb_units', type=int, nargs=2, default=[128, 128])
parser.add_argument('--timesteps', type=int, default=75)
parser.add_argument('--dropout_rate', type=float, default=0.25)
cfg = parser.parse_args()

#Load characters
cfg.characters = list();
for line in open(cfg.characters_file):
    line = line.rstrip('\n');
    cfg.characters.append(line);

def set_gpus():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)[1:-1]

def create_output_directory():
    os.makedirs('eval', exist_ok=True)
    output_subdir = create_result_subdir('eval')
    print('Output directory: ' + output_subdir)
    return output_subdir

def collect_data():
    if os.path.isfile(cfg.data_path):
        return [cfg.data_path]
    else:
        files = [os.path.join(cfg.data_path, f) for f in os.listdir(cfg.data_path) if f[-4:] in ['.tif', '.tiff', '.TIF', '.TIFF']]
        return files

def load_image(img_path):
    if cfg.nb_channels == 1:
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(img_path)    

def preprocess_image(img):
    if img.shape[1] / img.shape[0] < 8.0:
        img = pad_image(img, (cfg.width, cfg.height), cfg.nb_channels)
    else:
        img = resize_image(img, (cfg.width, cfg.height))
    if cfg.nb_channels == 1:
        img = img.transpose([1, 0])
    else:
        img = img.transpose([1, 0, 2])
    img = np.flip(img, 1)
    img = img / 255.0
    if cfg.nb_channels == 1:
        img = img[:, :, np.newaxis]
    return img

def predict_text(model, img):
    y_pred = model.predict(img[np.newaxis, :, :, :])
    shape = y_pred[:, 2:, :].shape
    ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0])*shape[1])[0][0]
    ctc_out = K.get_value(ctc_decode)[:, :cfg.label_len]
    result_str = ''
    for c in ctc_out[0]:
        if (cfg.characters[c] != '<eps>'):
            result_str = result_str + cfg.characters[c]
    return result_str

def evaluate(model, data, output_subdir):
    print ('In evaluate function');
    if len(data) == 1:
        evaluate_one(model, data)
    else:
        evaluate_batch(model, data, output_subdir)

def evaluate_one(model, data):
    img = load_image(data[0])
    img = 255.0 - img
    img = preprocess_image(img)
    result = predict_text(model, img)
    print('Detected result: {}'.format(result))

def evaluate_batch(model, data, output_subdir):
    for filepath in tqdm(data):        
        print (filepath);
        img = load_image(filepath)
        img = 255.0 - img
        img = preprocess_image(img)
        result = predict_text(model, img)
        output_file = os.path.basename(filepath)
        output_file = output_file[:-4] + '.txt'
        with open(os.path.join(output_subdir, output_file), 'w') as f:
            f.write(result)

if __name__ == '__main__':
    set_gpus()
    output_subdir = create_output_directory()
    data = collect_data()
    _, model = CRNN(cfg)    
    print (cfg.model_path);
    Model.load_weights(model, cfg.model_path)
    print (model.summary());
    evaluate(model, data, output_subdir)
