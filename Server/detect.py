import base64
import os
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import json

from Server import globalVariables

model = None

def detect(image):
    global model
 
    if globalVariables.isModelChanged:
        model = load_model(globalVariables.model_file, compile=False)
        globalVariables.isModelChanged = False

    image = np.frombuffer(image.read(), np.uint8)
    read_image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(read_image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)    

    predictions = model.predict(image)
    if predictions is None or predictions.size == 0:
        return False, []
    
    print(globalVariables.LABEL_MAP[np.argmax(predictions)]['english_label'])

    indexes = [i for i in range(0, len(predictions[0]), 1)]

    for i in range(0, len(predictions[0])-1, 1):
      max = predictions[0][i]
      u = i
      for j in range(i + 1, len(predictions[0]), 1):
        if predictions[0][j] > max:
            max = predictions[0][j]
            u = j

      predictions[0][u] = predictions[0][i]
      predictions[0][i] = max
      max = indexes[u]
      indexes[u] = indexes[i]
      indexes[i] = max

    result = []
    sample_flower_path = os.path.join(os.path.abspath(os.getcwd()), 'Server', 'SampleFLowerImages')

    for index, prediction in enumerate(predictions[0]):
        if prediction < 0.05:
            break
        
        folder_path = os.path.join(sample_flower_path, str(indexes[index]))
        file_path = os.path.join(folder_path, os.listdir(folder_path)[0])    

        image = cv2.imread(file_path)
        _, img_encoded = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        result.append({'label': json.dumps(globalVariables.LABEL_MAP[indexes[index]]), 'confidence': prediction * 100, 'image': img_base64})     
        
    return True, result
  

