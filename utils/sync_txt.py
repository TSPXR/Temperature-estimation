from datetime import datetime
import os
import bisect

# 파일 경로 및 파일명 설정
path = './samples/20240215/'
txt_list = ['m1_merge.txt', 'm2_merge.txt', 'm3_merge.txt']
synced_filenames = ['sync_m1.txt', 'sync_m2.txt', 'sync_m3.txt']

# 파일 데이터를 읽어서 타임스탬프와 함께 저장하는 함수
def read_and_parse_file(filename):
    with open(filename, 'r') as file:
        return [(datetime.fromisoformat(line.split(',')[0]), line) for line in file]

# 이진 탐색을 사용하여 가장 가까운 타임스탬프의 레코드를 찾는 함수
def find_closest_index(timestamps, target):
    i = bisect.bisect_left(timestamps, target)
    if i == len(timestamps):
        return i - 1
    elif i == 0:
        return 0
    else:
        if target - timestamps[i-1] <= timestamps[i] - target:
            return i - 1
        else:
            return i

# 각 파일에서 데이터를 읽고 파싱
data = [read_and_parse_file(os.path.join(path, txt)) for txt in txt_list]

# 타임스탬프만 별도로 추출하여 저장
timestamps = [[record[0] for record in dataset] for dataset in data]

# 동기화된 데이터를 저장할 리스트 초기화
synced_data = [[] for _ in range(len(txt_list))]

# 가장 짧은 데이터 세트를 기준으로 모든 레코드에 대해 동기화 수행
base_data_length = min(len(d) for d in data)
base_data_index = min(range(len(data)), key=lambda i: len(data[i]))
base_timestamps = timestamps[base_data_index]

for i, timestamp in enumerate(base_timestamps):
    for j, dataset in enumerate(data):
        closest_index = find_closest_index(timestamps[j], timestamp)
        synced_data[j].append(dataset[closest_index][1])

# 동기화된 데이터를 파일에 기록
for i, filename in enumerate(synced_filenames):
    with open(os.path.join(path, filename), 'w') as file:
        file.writelines(synced_data[i])

print("동기화 완료.")
