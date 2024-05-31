import base64
from tensorflow.keras.models import load_model
import cv2
import numpy as np

from Server import globalVariables

model = None

def showOnScreen(image):
    height, width = image.shape[:2]
    scale_factor = 614/height  # You can adjust this value as needed

    resize_image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))
    
    cv2.imshow("Detected Faces", resize_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    label = globalVariables.LABEL_MAP[np.argmax(predictions)]['english_label']
    label2 = globalVariables.LABEL_MAP[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    fontSize = 3*read_image.shape[1]/800
    color = (255, 255, 0)
    weight = 3

    pos = (30, 30)

    cv2.putText(read_image, f'Id: {str(label)}', pos, cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, weight)

    print('label: ', label, 'confidence: ', confidence, '%', flush=True)    

    _, img_encoded = cv2.imencode('.png', read_image)
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

    # only use at local (not at server)
    # showOnScreen(read_image)

    return True, {'label': label2, 'confidence': confidence}, img_base64
  

