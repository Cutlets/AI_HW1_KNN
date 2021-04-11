# 홍익대학교 컴퓨터공학과 2021년 1학기 인공지능 HW1
# 학 번 : B511156
# 이 름 : 이 치 현

import knn as kn
import numpy as np
from sklearn.datasets import load_iris
iris_dataset = load_iris()

# Petal Length와 Petal Width만 이용
i_data = iris_dataset['data'][:, 2:4]
i_class = iris_dataset['target'][:]
i_names = iris_dataset['target_names'][:]
i_fnames = iris_dataset['feature_names'][:]


def cal_knn(k_neighbor, test_num, show_detail):
  train_set = i_data
  train_class = i_class
  #print(train_set)

  # 앞에서부터 지우면 뒤쪽 인덱스가 모두 변하므로 뒤부터 순차로 삭제
  for i in reversed(range(0, len(i_data))):
   if (i + 1) % test_num == 0:
      train_set = np.delete(train_set,i,0)
      train_class = np.delete(train_class,i,0)
  # KNN 클래스 생성
  knn = kn.sim_knn(train_set, train_class, i_names, i_fnames, k_neighbor)

  # 기본 세팅 출력
  print("##\t Neighbor Set Value: " + str(k_neighbor) + "\t\t##")
  print("##\t Test Data Position: " + str(test_num) + "\t\t##")

  # Majority Vote
  print("===== Majority Vote Output =====")

  ## 정답 비율
  correct_ratio = [0, 0]
  for i in range(0, len(i_data)):
    if (i + 1) % test_num != 0:
      continue
    vote_result = i_names[knn.mj_vote(i_data[i])]
    true_result = i_names[i_class[i]]
    if vote_result == true_result:
      match_result = "Correct"
      correct_ratio[0] += 1
    else:
      match_result = "Wrong"
      correct_ratio[1] += 1
    ## 결과 출력
    if show_detail == 1:
      print("Test Data Index: " + str(i) + "\t | Majority vote: " + vote_result + "\t | True class: " + true_result + '\033[96m' + "\t || Result: " + match_result + '\033[0m')

  ##통계 출력
  if show_detail == 0:
    print("Match Result: " + str(correct_ratio[0]) +" of " + str(correct_ratio[0] + correct_ratio[1]) + " (" + str(round((correct_ratio[0]/(correct_ratio[0] + correct_ratio[1])), 3) * 100) + "%)")

  print("================================")
  # Majority Vote End

  #########################

  # Weighted Majority Vote
  print("==== W_Majority Vote Output ====")

  ## 정답 비율
  correct_ratio = [0, 0]
  for i in range(0, len(i_data)):
    if (i + 1) % test_num != 0:
      continue
    vote_result = i_names[knn.w_mj_vote(i_data[i])]
    true_result = i_names[i_class[i]]

    if vote_result == true_result:
      match_result = "Correct"
      correct_ratio[0] += 1
    else:
      match_result = "Wrong"
      correct_ratio[1] += 1
    # 결과 출력
    if show_detail == 1:
      print("Test Data Index: " + str(i) + "\t | Majority vote: " + vote_result + "\t | True class: " + true_result + '\033[96m' + "\t || Result: " + match_result + '\033[0m')

  #통계 출력
  if show_detail == 0:
    print("Match Result: " + str(correct_ratio[0]) +" of " + str(correct_ratio[0] + correct_ratio[1]) + " (" + str(round((correct_ratio[0]/(correct_ratio[0] + correct_ratio[1])), 3) * 100) + "%)")

  print("================================")
  # Weighted Majority Vote End

#############################
## 이곳을 수정하여 세팅 #####
#############################
# 탐색할 이웃 수 설정
k_neighbor = 5

# 매 test data의 위치
test_num = 15

# 세부결과 출력여부
show_detail = 1

# 세탕값으로 결과를 보려면 1 세팅값 순차탐색은 0
set_flag = 1
#############################
#############################

if set_flag == 1:
  cal_knn(k_neighbor, test_num, show_detail)
else:
  for i in range(3, 11):
    for j in range(1,4):
      cal_knn(i, j * 5, 0)
