import matplotlib.pyplot as plt
import psychrolib
psychrolib.SetUnitSystem(psychrolib.SI)

# 실내 온습도 데이터 정의
T_db = 24  # 건구온도(°C)
RH = 50    # 상대습도(%)
Pressure = 101325

# 건구온도와 습도비로부터 습구온도 계산
T_wb = psychrolib.GetTWetBulbFromRelHum(T_db, RH/100.0, Pressure)

# 사이크로메트릭 차트의 컴포트존 정의 및 그리기 (여기서는 간단한 예로, 실제로는 더 복잡한 과정이 필요할 수 있음)
plt.figure(figsize=(10, 6))
plt.plot(T_db, T_wb, 'bo', label='test')
plt.fill_between([22, 26], 40, 60, color='green', alpha=0.2, label='comfort zone')
plt.xlabel('temp (°C)')
plt.ylabel('hud (°^)')
plt.title('chart')
plt.legend()
plt.grid(True)
plt.show()