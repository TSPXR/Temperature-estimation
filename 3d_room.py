import numpy as np
import matplotlib.pyplot as plt

def get_voxel_pos(x, y, grid_size):
    x = int(np.floor(x / grid_size))
    y = int(np.floor(y / grid_size))
    return x, y

# 복셀 그리드의 크기 계산
space_x = 4.563
space_y = 8.314
grid_size = 1
x_size = int(np.ceil(space_x / grid_size))
y_size = int(np.ceil(space_y / grid_size))

# 3차원 복셀 그리드 생성 (모든 값을 0으로 초기화)
voxel_grid = np.zeros((x_size, y_size))

m1_x, m1_y = get_voxel_pos(4.563, 0.4, grid_size)
m2_x, m2_y = get_voxel_pos(0, 7.492, grid_size)
m3_x, m3_y = get_voxel_pos(4.563, 7.628, grid_size)
window_x, window_y = get_voxel_pos(1.4, 0, grid_size)
heat_x, heat_y = get_voxel_pos(2.81, 3.34, grid_size)
door_x, door_y = get_voxel_pos(0, 6.042, grid_size)

voxel_grid[m1_x,  m1_y] = 1  # m1 position
voxel_grid[m2_x, m2_y] = 1  # m2 position
voxel_grid[m3_x, m3_y] = 1  # m3 position
voxel_grid[window_x, window_y] = 2  # window
voxel_grid[heat_x, heat_y] = 3  # heat
voxel_grid[door_x, door_y] = 4  # heat

# 생성된 복셀 그리드의 크기 확인
print(f"복셀 그리드의 크기: {voxel_grid.shape}")

slice_index = 5 # 예제로 5번 인덱스 슬라이스 선택
plt.imshow(voxel_grid, cmap='viridis')
plt.colorbar()
plt.show()