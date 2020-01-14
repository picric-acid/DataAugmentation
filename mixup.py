import numpy as np


class MixupGenerator():

    # Require labels converted to One-Hot Encoding
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y

class MixupDataset(MixupGenerator):

    def replace_data(self, batch_size=64, alpha=0.5, iter=2):
        gen_field = MixupGenerator(self.X_train, self.y_train, batch_size, alpha)()

        data_X =[]
        data_y = []

        for i in range(iter):
            x, y = next(gen_field)
            print('---------------------------')
            print(str(i) + ': iterations started')
            print('---------------------------')
            for j in range(x.shape[0]):
                print(str(j) + ': epochs started')
                gen_img = x[j]
                gen_label = y[j]
                print(gen_label.shape)
                print(gen_label)
                # print(gen_img.shape)
                gen_img = gen_img.tolist()
                gen_label = gen_label.tolist()
                # print(gen_label)
                data_X.append(gen_img)
                data_y.append(gen_label)
                # data_y = np.insert(data_y, 128 * i + j, gen_label, axis=0)
                # data_X = np.insert(data_X, 128 * i + j, gen_img, axis=0)

        data_X = np.array(data_X)
        data_y = np.array(data_y)

        return data_X, data_y

    def add_data(self, batch_size=64, alpha=0.5, iter=30):
        gen_field = MixupGenerator(self.X_train, self.y_train, batch_size, alpha)

        data_X = np.copy(self.X_train)
        data_y = np.copy(self.y_train)

        data_len = data_X.shape[0]

        for i in range(iter):
            x, y = next(gen_field)
            print('---------------------------')
            print(str(i) + ': iterations started')
            print('---------------------------')
            for j in range(x.shape[0]):
                print(str(j) + ': epochs started')
                gen_img = x[j]
                gen_label = y[j]
                # data_X.append(gen_img)
                # data_y.append(gen_label)
                data_y = np.insert(data_y, int(data_len) + (128 * i + j), gen_label, axis=0)
                data_X = np.insert(data_X, int(data_len) + (128 * i + j), gen_img, axis=0)


        return data_X, data_X
