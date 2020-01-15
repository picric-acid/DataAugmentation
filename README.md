# Data Augmentation for Image Classification

- requirements

``Python 3.x``

- Installation

``pip install git+https://github.com/picric-acid/DataAugmentation.git -t "Library Root Path"``

---

- Random Erasing

``from DataAugmentation.random_erasing import RandomErase``

input_data : single image of ndarray

input_shape : (height, width, channels)

output_data : erased image of ndarray

output_shape : same as input_shape


- Mixup

``from DataAugmentation.mixup import MixupDataset``

input_data(image) : all of images of ndarray

input_data(label) : require One-Hot Encoding, all of labels of ndarray

input_shape(image) : (data_size, height, width, channels)

input_shape(label) : (data_size, number_of_label)

output_data : mixuped image and label(One-Hot Encoding)
