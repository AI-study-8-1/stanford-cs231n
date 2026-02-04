from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting   # 테스트 데이터 (10000, 32*32*3)
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.    # knn에서 k 개수
        - num_loops: Determines which implementation to use to compute distances    # 어떤 함수로 계산을 할 지 선정
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the # 테스트 데이터의 예측값 저장
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j] # 10000(행)*50000(열)의 행렬
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]   # X에서 num_test의 개수 -> 10000개
        num_train = self.X_train.shape[0]   # knn.ipynb에서 classifier의 X_train -> 학습 이미지 데이터 수
        dists = np.zeros((num_test, num_train)) # 0으로 초기화된 10000*50000 행렬 생성
        for i in range(num_test):
            for j in range(num_train):
                ##### TODO #####
                dis = X[i] - self.X_train[j]    # X[i]와 X_train[j]의 차(각 값의 차이)
                dists[i, j] = np.sqrt(np.sum(dis**2))   #l2를 시행 -> 각 값을 제곱해서 더하고 루트 씌우기
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            ##### TODO #####
            dis = self.X_train - X[i]   # X_train의 각 값에서 X[i]의 값을 뺌
            dis = np.sqrt(np.sum(dis**2, axis = 1)) # (index, D)에서 D를 없애며 합을 하기 위해 axis를 1로 설정
            dists[i] = dis
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        ##### TODO #####    # x^2 + y^2 -2xy를 하고 sqrt를 하는 방식으로 진행한다
        X_square = np.sum(X**2, axis = 1)   # 각 값의 제곱값을 더해서 x^2을 만듬
        X_train_square = np.sum(self.X_train**2, axis = 1)  #각 값의 제곱값을 더해서 y^2을 만듬
        dists = np.dot(X_square, np.transpose(X_train_square))
        X_dot_X_train = np.dot(X, np.transpose(self.X_train))   # 둘을 내적해서 10000*50000짜리 xy 값들의 합을 만들어줌
        dists = dists - 2 * X_dot_X_train   # x^2 + y^2 -2xy
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]   # test 데이터 수
        y_pred = np.zeros(num_test) # test 데이터 예측값
        for i in range(num_test):
            closest_y_index = np.argsort(dists[i])[:k]  # dists를 정렬해서 낮은 순서대로 index 저장, 앞 k개 슬라이싱
            y_num = [0] * 10
            for a in range(k):
                x = self.y_train[closest_y_index[a]]
                y_num[x] += 1
            y_pred[i] = np.argsort(y_num)[-1]
        return y_pred
