from tqdm import tqdm
import os
import cv2
import numpy as np

# from  .image_processing import  IMG_HEIGHT,IMG_WIDTH

# set to true to one once, then back to false unless you want to change something in your training data.
REBUILD_DATA = True

data_dir = "./fonts/train/"


class Fonts():
    IMG_WIDTH = 500
    IMG_HEIGHT = 50
    RIKA = f"{data_dir}rika"
    NESIH = f"{data_dir}nesih"
    LABELS = {RIKA: 0, NESIH: 1}
    training_data = []

    rikacount = 0
    nesihcount = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if "png" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(
                            img, (self.IMG_WIDTH, self.IMG_HEIGHT))
                        # do something like print(np.eye(2)[1]), just makes one_hot
                        self.training_data.append(
                            [np.array(img), np.eye(2)[self.LABELS[label]]])

                        if label == self.NESIH:
                            self.nesihcount += 1
                        elif label == self.RIKA:
                            self.rikacount += 1

                    except Exception as e:
                        pass
                        #print(label, f, str(e))

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)


if REBUILD_DATA:
    fonts = Fonts()
    fonts.make_training_data()
