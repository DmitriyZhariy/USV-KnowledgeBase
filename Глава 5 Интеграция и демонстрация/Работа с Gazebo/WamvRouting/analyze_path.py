#!/usr/bin/env python3

from pathlib import Path
from math import hypot
from rosbags.highlevel import AnyReader
import matplotlib.pyplot as plt

bag_path = Path('praktika_run2')

points = []

with AnyReader([bag_path]) as reader:
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/wamv/debug/path':
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            if msg.poses:
                # Берем только ПОСЛЕДНЮЮ точку из пришедшего пути
                last_pose = msg.poses[-1]
                x = last_pose.pose.position.x
                y = last_pose.pose.position.y
                
                # Защита от дубликатов, если сообщение пришло дважды с теми же данными
                if not points or (points[-1][0] != x or points[-1][1] != y):
                    points.append((x, y))

distance = 0.0
for (x1, y1), (x2, y2) in zip(points, points[1:]):
    distance += hypot(x2 - x1, y2 - y1)

print(f'Всего уникальных точек траектории (сырых): {len(points)}')
print(f'Точек после фильтрации нулевых: {len(points)}')
print(f'Пройденная длина пути (без возвратов в (0,0)), м: {distance:.2f}')
print('Первые 10 точек после фильтрации:')
for i, (x, y) in enumerate(points[:10]):
    print(f'{i:3d}: x={x:.3f}, y={y:.3f}')

# Рисуем и сохраняем траекторию
xs = [p[0] for p in points]
ys = [p[1] for p in points]

plt.figure(figsize=(6, 6))
plt.plot(xs, ys, '-', linewidth=1)
plt.scatter(xs[0], ys[0], c='green', label='start')
plt.scatter(xs[-1], ys[-1], c='red', label='end')
plt.xlabel('x, м (frame map)')
plt.ylabel('y, м (frame map)')
plt.title('Траектория WAM-V по /wamv/debug/path')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('path_xy.png', dpi=200)

print('Картинка траектории сохранена в path_xy.png')