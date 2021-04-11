# 홍익대학교 컴퓨터공학과 2020년 1학기 인공지능 HW1
# 학 번 : B511156
# 이 름 : 이 치 현

import numpy as np

# KNN Class
class sim_knn:
  knn_data = []
  knn_class = []
  knn_names = []
  knn_fnames = []
  knn_nei = 0

  def __init__(self, idata, iclass, inames, ifnames, neigh):
    # print('KNN class is created Successfully!')
    self.knn_data = idata[:][:]
    self.knn_class = iclass
    self.knn_names = inames
    self.knn_fnames = ifnames
    self.knn_nei = neigh

  def cal_distance(self, d_new, d_exist):
    return float(np.sqrt(np.sum(np.power((d_exist - d_new), 2))))

  def get_neighbor(self, d_center):
    # 거리를 계산할 배열
    dis_array = np.zeros(len(self.knn_data))

    # 거리를 계산
    for i in range(0, len(self.knn_data)):
      dis_array[i] = self.cal_distance(d_center, self.knn_data[i])

    ind_array = np.argsort(dis_array)

    # 이웃의 분류를 담을 배열
    near_array = np.zeros(self.knn_nei)

    # 자신이 포함되는 경우 제외하기 위해서 이용
    exclude_flag = 0
    if (self.knn_data[ind_array[0]].all == d_center.all):
      exclude_flag = 1

    # 이웃들의 target 값을 저장
    for i in range(0 + exclude_flag, self.knn_nei + exclude_flag):
      near_array[i - exclude_flag] = self.knn_class[ind_array[i]]

    # 배열을 반환
    # print(near_array)
    return near_array

  def mj_vote(self, d_center):
    # 근접 이웃들을 구해온다
    near_array = self.get_neighbor(d_center)
    # Vote 값을 저장할 배열
    normalvote_array = np.zeros(3)

    # 근접 이웃을의 분류값을 이용해 투표
    for i in range(0, len(near_array)):
      normalvote_array[int(near_array[i])] += 1
    # print(normalvote_array)

    # 투표결과 가장 많은 결과의 반환
    # print(normalvote_array)
    return np.argsort(normalvote_array)[2]

  def w_mj_vote(self, d_center):
    # 근접 이웃들을 구해온다
    dis_array = np.zeros(len(self.knn_data))

    # 거리를 계산
    for i in range(0, len(self.knn_data)):
      dis_array[i] = self.cal_distance(d_center, self.knn_data[i])

    ind_array = np.argsort(dis_array)
    near_array = ind_array[0:self.knn_nei]

    # 이웃들의 가중치를 담은 배열
    w_array = np.zeros(self.knn_nei)

    for i in range(0, len(near_array)):
      d_nei = self.knn_data[int(near_array[i])]
      # exp(-x)로 가중치 계산
      w_array[i] = np.exp(-self.cal_distance(d_center, d_nei))

    # Vote 값을 저장할 배열
    weightvote_array = np.zeros(3)

    # 근접 이웃을의 분류값을 이용해 투표
    for i in range(0, len(near_array)):
      # 각 투표의 중요도는 1 * 가중치로 계산한다.
      weightvote_array[int(self.knn_class[near_array[i]])] += (1 * w_array[i])

    # print(weightvote_array)
    # 투표결과 가장 많은 결과의 반환
    # print(weightvote_array)
    return np.argsort(weightvote_array)[2]

  def __del__(self):
    # print('KNN class is removed Successfully!')
    pass

