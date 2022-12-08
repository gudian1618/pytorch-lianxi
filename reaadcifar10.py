import os
import cv2
import numpy as np
import pickle
import glob


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


label_name = ["airplane",
              "automoblile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]

# train_list = glob.glob("C:/Users/Administrator/Desktop/cifar-10-python/cifar-10-batches-py/data_batch_*")
train_list = glob.glob("C:/Users/Administrator/Desktop/cifar-10-python/cifar-10-batches-py/test_batch*")
print(train_list)

# save_path = "C:/Users/Administrator/Desktop/cifar-10-python/cifar-10-batches-py/TRAIN"
save_path = "C:/Users/Administrator/Desktop/cifar-10-python/cifar-10-batches-py/TEST"

for l in train_list:
    print(l)
    l_dict = unpickle(l)
    print(l_dict)
    print(l_dict.keys())

    for im_idx, im_data in enumerate(l_dict[b'data']):
        # print(im_idx)
        # print(im_data)

        im_label = l_dict[b'labels'][im_idx]
        im_name = l_dict[b'filenames'][im_idx]

        print(im_label, im_name, im_data)

        im_label_name = label_name[im_label]
        im_data = np.reshape(im_data, [3, 32, 32])
        im_data = np.transpose(im_data, (1, 2, 0))

        # cv2.imshow("im_data", cv2.resize(im_data, (200, 200)))
        # cv2.waitKey(0)

        if not os.path.exists("{}/{}".format(save_path, im_label_name)):
            os.mkdir("{}/{}".format(save_path, im_label_name))

        cv2.imwrite("{}/{}/{}".format(save_path, im_label_name, im_name.decode("utf-8")), im_data)

