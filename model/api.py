from flask import Flask, request, jsonify
from keras.models import model_from_json
import numpy as np
import cv2
from PIL import Image

app = Flask(__name__)

# Assuming the labels dictionary is defined somewhere in your code as 'labels'
# labels = {0: 'label1', 1: 'label2', ...}
label_data_index= {
'Pepper_bell__Bacterial_spot': 0,
 'Pepper_bell__healthy': 1,
 'Potato___Early_blight': 2,
 'Potato___Late_blight': 3,
 'Potato___healthy': 4,
 'Tomato_Bacterial_spot': 5,
 'Tomato_Early_blight': 6,
 'Tomato_Late_blight': 7,
 'Tomato_Leaf_Mold': 8,
 'Tomato_Septoria_leaf_spot': 9,
 'Tomato_Spider_mites_Two_spotted_spider_mite': 10,
 'Tomato__Target_Spot': 11,
 'Tomato_Tomato_YellowLeaf_Curl_Virus': 12,
    'Tomato__Tomato_mosaic_virus': 13,
 'Tomato_healthy': 14
 }

def convert_to_array(img):
    im = cv2.imread(img)
    img_ = Image.fromarray(im, 'RGB')
    image = img_.resize((100, 100))
    return np.array(image)

def predict_cell(file_path):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("PLANT_DISEASE.h5")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    ar = convert_to_array(file_path)
    ar = ar/255
    a = [ar]
    a = np.array(a)
    score = loaded_model.predict(a, verbose=1)
    label_index = np.argmax(score)
    acc = np.max(score)
    lables_data = label_data_index
    value = {i for i in lables_data if lables_data[i] == label_index}
    return {"disease_name": list(value)[0], "accuracy": str(acc)}

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the image is posted
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if file:
        file_path = "./" + file.filename
        file.save(file_path)
        result = predict_cell(file_path)
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)