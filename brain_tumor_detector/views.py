from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from django.conf import settings

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
MODEL_PATH = os.path.join('static',
                          'model', 'brain_tumor.h5')
model = tf.keras.models.load_model(MODEL_PATH)

# Training performance metrics
MODEL_ACCURACY = 99.12  # %
MODEL_PRECISION = 98.45  # %


# Load test data to evaluate accuracy and precision dynamically
# test_dir = os.path.join('brain_tumor_detector', 'test')  # Adjust path to test set
# test_datagen = ImageDataGenerator(rescale=1./255)

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(128, 128),
#     batch_size=32,
#     class_mode='binary',
#     shuffle=False
# )

# loss, accuracy, precision = model.evaluate(test_generator, verbose=0)

def preprocess_image(image_path):
    """
    Preprocess the uploaded image to match your custom CNN model input.
    - Converts image to RGB
    - Resizes to 128x128
    - Normalizes pixel values to [0, 1]
    - Expands dimensions to (1, 128, 128, 3)
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def home(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        # full_path = os.path.join('brain_tumor_detector',
        #                          'media', filename)
        full_path = os.path.join(settings.MEDIA_ROOT, filename)

        try:
            img_array = preprocess_image(full_path)
            prediction = model.predict(img_array)[0][0]

            tumor_prob = round(prediction * 100, 2)
            no_tumor_prob = round((1 - prediction) * 100, 2)

            result = 'Tumor Detected' if prediction > 0.5 else 'No Tumor Detected'
            print(full_path)
            return render(request, 'result.html', {
                'prediction': result,
                'tumor_prob': tumor_prob,
                'no_tumor_prob': no_tumor_prob,
                'image': fs.url(filename)
            })
        except Exception as e:
            return render(request, 'result.html', {
                'error': f"Error processing image: {str(e)}"
            })

    return render(request, 'home.html')
