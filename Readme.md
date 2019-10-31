Basic Usage:
- Uses Tensorflow 2 
- Create Conda Environment with the yml file >>  conda env create -f udacity_intro_to_tensorflow.yml
- Python Notebook: intro_to_tensorflow/projects/p2_image_classifier >> Project_Image_Classifier_Project.ipynb
- Command Line Application: intro_to_tensorflow/projects/p2_image_classifier/src >> predict.py
- Use python -W ignore predict.py -h to get arguments to be passed in the applications
- Following are the default arguments:
    parser.add_argument('--model_path', nargs='?', default='../training_1/best_weights.h5',
                        help='Model Path')
    parser.add_argument('--img_path', nargs='?', default='../test_images/training_data_check.jpg',
                        help='Image Location')
    parser.add_argument('--top_k', nargs='?', default=5,
                    help='Return the top K most likely classes')
    parser.add_argument('--category_names', nargs='?', default='../label_map.json',
                    help='Path to a JSON file mapping labels to flower names:')

- Use python -W ignore predict.py to run with default arguments

Others:
- Images in test folder are very different from the tensorhub model: https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4
- Command Line returns probs, classes,filtered(classes,class_names),pred_dict  

- All the necessary packages and modules are imported at the beginning of the notebook. << Please use provided yml file to create the environment
