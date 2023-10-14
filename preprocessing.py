class StandardScaler:

    def __init__(self):
        self.x_mean = None
        self.x_std = None

    def fit_transform(self, x):
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        # all above 2 should have shape (1, n)

        x = (x - x_mean) / x_std

        self.x_mean = x_mean
        self.x_std = x_std

        return x

    def transform(self, x):

        if self.x_mean is not None and self.x_std is not None:
            x = (x - self.x_mean) / self.x_std
        else:
            raise ValueError('Please train and fit before transform')

        return x


class MinMaxScaler:

    def __init__(self):
        self.x_min = None
        self.x_max = None

    def fit_transform(self, x):
        x_mean = x.mean(axis=0)
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)
        # all above 3 should have shape (1, n)

        x = (x - x_mean) / (x_max - x_min)

        self.x_max = x_max
        self.x_min = x_min

        return x

    def transform(self, x):

        if self.x_min is not None and self.x_max is not None:
            x = (x - self.x_min) / (self.x_max - self.x_min)
        else:
            raise ValueError('Please train and fit before transform')

        return x