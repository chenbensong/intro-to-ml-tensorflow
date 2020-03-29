"""predict.py: use trained model.h5 to predict top K categories for a given image.

Sample usage:

    python predict.py --top_k 3 --category_names label_map.json --img_path test_images/wild_pansy.jpg --model model.h5
"""


import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import layers

num_classes=102


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model.h5')
    parser.add_argument('--img_path', default='test_images/wild_pansy.jpg')
    parser.add_argument('--top_k', nargs='?', default=5,
                    help='Return the top K most likely classes')
    parser.add_argument('--category_names', default='label_map.json')
    return parser.parse_args()


def predict(image_path=None, model=None, top_k=None):
    image = np.asarray(Image.open(image_path))
    processed_image = tf.image.resize(tf.convert_to_tensor(image, tf.float32), (224, 224)) / 255  
    processed_image = np.expand_dims(processed_image, 0)
    probs=model.predict(processed_image)
    
    return tf.nn.top_k(probs, k=top_k)


def filtered(classes=None,class_names=None):
    return [class_names.get(str(key)) if key else "Placeholder" for key in classes.numpy().squeeze().tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_names', default='label_map.json')
    parser.add_argument('--img_path', default='test_images/wild_pansy.jpg')
    parser.add_argument('--model', default='model.h5')
    args = parser.parse_args()

    with open(args.category_names, 'r') as f:
       class_names = json.load(f)
        
    print('Loading model {}'.format(args.model))
    classifier_url = 'https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4'
    classifier = tf.keras.Sequential([
        # mobilenet
        hub.KerasLayer(classifier_url, input_shape=(224, 224, 3)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
     ])

    # Early stop callback
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
      monitor='val_accuracy', min_delta=0.0001,
      patience=1)

    # Checkpoint
    checkpoint = 'checkpoint.ckpt'

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint,
       save_weights_only=True,
       verbose=1)

    # Compile
    classifier.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 metrics=['accuracy'],
                 callbacks=[cp_callback, earlystop_callback])
    classifier.load_weights(args.model)
    
    print('\nPredicting...')
    probs, classes = predict(image_path=args.img_path, model=classifier, top_k=args.top_k)
    classes = [class_names[str(value+1)] for value in classes.numpy()[0]]
    print('Top {} predicted classes:'.format(args.top_k))
    for i in range(len(classes)):
       print('{0}  {1:.5f}%'.format(classes[i], probs[0][i] * 100))
    
    
if __name__ == '__main__':
    main()
