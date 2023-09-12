from keras.models import load_model
from PIL import Image

import numpy as np

import cv2
from keras.models import model_from_json
def convert_to_array(img):
    im = cv2.imread(img)
    img_ = Image.fromarray(im, 'RGB')
    image = img_.resize((100, 100))
    return np.array(image)

def predict_cell(file):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("PLANT_DISEASE.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Predicting Type of Cell Image.................................")
    ar=convert_to_array(file)
    ar=ar/255
    plt.imshow(ar)
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=loaded_model.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    lables_data = labels
    value = {i for i in lables_data if lables_data[i]==label_index}
    print("your disease name is :",value)
    #Cell=get_cell_name(label_index)
    #print(Cell,"The predicted Cell is a "+Cell+" with accuracy = "+str(acc))
    #print(label_index)


predict_cell("/content/drive/MyDrive/PlantVillage/Pepper_bell_healthy/002f87b7-e1a5-49e5-a422-bb423630ded5__JR_HL 8068.JPG")