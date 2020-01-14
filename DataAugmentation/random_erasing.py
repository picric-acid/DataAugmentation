import numpy as np


class RandomErase(object):

    # append erased data to existing data
    def add_data(self, img, p=0, s_l=0.02, s_h=0.4, r1=0.3, r2=1/0.3):

        p1 = np.random.uniform(0, 1)
        if p1 < p:
            return img
        else:
            H = img.shape[0]
            W = img.shape[1]
            S = H * W

            while True:
                S_e = S * np.random.uniform(low=s_l, high=s_h)
                r_e = np.random.uniform(low=r1, high=r2)

                H_e = np.sqrt(S_e * r_e)
                W_e = np.sqrt(S_e / r_e)

                x_e = np.random.randint(0, W)
                y_e = np.random.randint(0, H)

                if x_e + W_e <= W and y_e + H_e <= H:
                    img_erased = np.copy(img)
                    img_erased[y_e:int(y_e + H_e + 1), x_e:int(x_e + W_e + 1), :] = np.random.uniform(0, 256)

                    # return single image
                    return img_erased

    # replace existing data to erased data
    def replace_data(self, img, p=0.15, s_l=0.002, s_h=0.4, r1=0.3, r2=1/0.3):

        p1 = np.random.uniform(0, 1)
        if p1 < p:
            return img
        else:
            H = img.shape[0]
            W = img.shape[1]
            S = H * W

            while True:
                S_e = S * np.random.uniform(low=s_l, high=s_h)
                r_e = np.random.uniform(low=r1, high=r2)

                H_e = np.sqrt(S_e * r_e)
                W_e = np.sqrt(S_e / r_e)

                x_e = np.random.randint(0, W)
                y_e = np.random.randint(0, H)

                if x_e + W_e <= W and y_e + H_e <= H:
                    img_erased = np.copy(img)
                    img_erased[y_e:int(y_e + H_e + 1), x_e:int(x_e + W_e + 1), :] = np.random.uniform(0, 256)

                    # return single image
                    return img_erased
