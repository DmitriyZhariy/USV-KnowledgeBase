#!/usr/bin/env python3
"""
WAM-V Waypoint Follower — GPS + IMU, LOS Guidance
==================================================
Архитектура:
  NavSatFix → положение (dx, dy, dist)
  Imu       → реальный курс носа (yaw)
  LOS       → желаемый курс (ψ_d)
  PID       → ошибка курса → дифференциальная тяга

Топики:
  SUB  /wamv/sensors/gps/gps/fix         sensor_msgs/NavSatFix
  SUB  /wamv/sensors/imu/imu/data        sensor_msgs/Imu
  PUB  /wamv/thrusters/left/thrust       std_msgs/Float64
  PUB  /wamv/thrusters/right/thrust      std_msgs/Float64
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64
import math
import time


# ─────────────────────────────────────────────────────────────────────────────
# Координатные утилиты
# ─────────────────────────────────────────────────────────────────────────────

EARTH_RADIUS_M = 6_371_000.0


def gps_to_xy(lat_ref: float, lon_ref: float,
              lat: float,     lon: float) -> tuple[float, float]:
    """
    Плоское приближение GPS → локальный Декарт (Восток=X, Север=Y), метры.
    Погрешность < 0.1 м при расстояниях до 500 м.
    """
    dy = (lat - lat_ref) * math.radians(1.0) * EARTH_RADIUS_M
    dx = (lon - lon_ref) * math.radians(1.0) * EARTH_RADIUS_M * math.cos(math.radians(lat_ref))
    return dx, dy


def normalize(a: float) -> float:
    """Нормализация угла в [-π, +π]."""
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """
    Извлекает yaw (рыскание) из кватерниона.
    Yaw = угол поворота вокруг оси Z (вверх).
    Результат: угол в радианах, 0 = Восток в системе IMU.

    Формула:
      yaw = atan2( 2*(qw*qz + qx*qy), 1 - 2*(qy² + qz²) )
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# ─────────────────────────────────────────────────────────────────────────────
# PID
# ─────────────────────────────────────────────────────────────────────────────

class PID:
    def __init__(self, kp: float, ki: float, kd: float,
                 i_limit: float = 50.0, out_limit: float = 300.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit  = i_limit
        self.out_limit = out_limit
        self._i   = 0.0
        self._prev = 0.0
        self._t:  float | None = None

    def reset(self):
        self._i    = 0.0
        self._prev = 0.0
        self._t    = None

    def compute(self, error: float) -> float:
        now = time.monotonic()
        dt  = (now - self._t) if self._t is not None else 0.05
        dt  = max(1e-3, dt)
        self._t = now

        self._i += error * dt
        self._i  = max(-self.i_limit, min(self.i_limit, self._i))

        d = (error - self._prev) / dt
        self._prev = error

        u = self.kp * error + self.ki * self._i + self.kd * d
        return max(-self.out_limit, min(self.out_limit, u))


# ─────────────────────────────────────────────────────────────────────────────
# Основной узел
# ─────────────────────────────────────────────────────────────────────────────

class WaypointFollowerLOSIMU(Node):

    # ── Маршрут ───────────────────────────────────────────────────────────────
    WAYPOINTS = [
        (-33.722420260545036, 150.67403396621702),  # WP 0
        (-33.722509916956405, 150.67423228791642),  # WP 1
        (-33.722568673622860, 150.67397403791983),  # WP 2
    ]

    # ── Навигация ─────────────────────────────────────────────────────────────
    ARRIVAL_RADIUS_M = 3.0    # принять WP достигнутым, м
    LOS_LOOKAHEAD_M  = 8.0    # Δ — дальность взгляда LOS, м

    # ── Управление ────────────────────────────────────────────────────────────
    THRUST_MAX       = 220.0  # максимальная тяга вперёд, Н
    THRUST_MIN_FWD   = 30.0   # минимальная тяга вперёд в фазе DRIVE, Н

    # ── Пороги FSM (с гистерезисом) ───────────────────────────────────────────
    ROTATE_ENTER_DEG = 20.0   # войти в ROTATE при |err| > N°
    ROTATE_EXIT_DEG  = 8.0    # выйти из ROTATE при |err| < N°

    # ── Смещение IMU → Север ──────────────────────────────────────────────────
    # IMU VRX отдаёт yaw в системе ENU: 0° = Восток, +90° = Север.
    # Наш bearing считается от Севера (как компас).
    # Поправка: yaw_north = yaw_enu + 90° = yaw_enu + π/2
    IMU_TO_NORTH_OFFSET = math.pi / 2.0

    def __init__(self):
        super().__init__('waypoint_follower_los_imu')

        self._pub_l = self.create_publisher(Float64, '/wamv/thrusters/left/thrust',  10)
        self._pub_r = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)

        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix',    self._gps_cb, 10)
        self.create_subscription(Imu,       '/wamv/sensors/imu/imu/data',   self._imu_cb, 10)

        self.create_timer(0.1, self._loop)

        # Состояние
        self._lat: float | None = None
        self._lon: float | None = None
        self._lat_ref: float | None = None
        self._lon_ref: float | None = None
        self._yaw: float | None = None   # курс носа (рад, от Севера, CW)

        self._wp_idx   = 0
        self._rotating = True
        self._pid = PID(kp=140.0, ki=0.4, kd=22.0)

        self.get_logger().info(
            f'LOS+IMU Waypoint Follower запущен | '
            f'{len(self.WAYPOINTS)} точек | Δ={self.LOS_LOOKAHEAD_M} м'
        )

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _gps_cb(self, msg: NavSatFix):
        if msg.status.status < 0:
            return
        self._lat = msg.latitude
        self._lon = msg.longitude
        if self._lat_ref is None:
            self._lat_ref = self._lat
            self._lon_ref = self._lon
            self.get_logger().info(
                f'GPS опора: {self._lat:.7f}, {self._lon:.7f}'
            )

    def _imu_cb(self, msg: Imu):
        q = msg.orientation
        # yaw_enu: 0 = Восток, CCW+
        yaw_enu = quat_to_yaw(q.x, q.y, q.z, q.w)
        # Переводим в навигационный курс: 0 = Север, CW+
        self._yaw = normalize(self.IMU_TO_NORTH_OFFSET - yaw_enu)

    # ── Главный цикл ─────────────────────────────────────────────────────────

    def _loop(self):
        # Ждём данных
        if self._lat is None or self._yaw is None:
            return
        if self._wp_idx >= len(self.WAYPOINTS):
            return

        # Текущее положение в локальном Декарте
        cur_x, cur_y = gps_to_xy(
            self._lat_ref, self._lon_ref, self._lat, self._lon
        )

        # Целевая точка
        wp_lat, wp_lon = self.WAYPOINTS[self._wp_idx]
        wp_x, wp_y = gps_to_xy(self._lat_ref, self._lon_ref, wp_lat, wp_lon)

        dist = math.hypot(wp_x - cur_x, wp_y - cur_y)

        # ── Прибытие ─────────────────────────────────────────────────────────
        if dist < self.ARRIVAL_RADIUS_M:
            self.get_logger().info(
                f'✓ WP {self._wp_idx} достигнут! (dist={dist:.1f} м)'
            )
            self._wp_idx  += 1
            self._rotating = True
            self._pid.reset()
            if self._wp_idx >= len(self.WAYPOINTS):
                self._thrust(0.0, 0.0)
                self.get_logger().info('🏁 Маршрут завершён!')
            return

        # ── LOS Guidance → желаемый курс ─────────────────────────────────────
        psi_d = self._los(cur_x, cur_y, wp_x, wp_y)

        # ── Ошибка курса ─────────────────────────────────────────────────────
        err = normalize(psi_d - self._yaw)

        # ── FSM с гистерезисом ───────────────────────────────────────────────
        abs_err = abs(err)
        if not self._rotating and abs_err > math.radians(self.ROTATE_ENTER_DEG):
            self._rotating = True
            self._pid.reset()
            self.get_logger().info(
                f'→ ROTATE (err={math.degrees(err):+.1f}°)'
            )
        elif self._rotating and abs_err < math.radians(self.ROTATE_EXIT_DEG):
            self._rotating = False
            self.get_logger().info(
                f'→ DRIVE  (err={math.degrees(err):+.1f}°)'
            )

        # ── PID ──────────────────────────────────────────────────────────────
        u = self._pid.compute(err)

        # ── Смешивание тяг ───────────────────────────────────────────────────
        if self._rotating:
            # Поворот на месте — IMU обновляется без движения, COG не нужен
            left  =  u
            right = -u
            phase = 'ROTATE'
        else:
            # Тяга вперёд пропорциональна cos(err): при большой ошибке —
            # замедляемся, но не останавливаемся полностью (min = THRUST_MIN_FWD)
            t_fwd = max(
                self.THRUST_MIN_FWD,
                self.THRUST_MAX * math.cos(err)
            )
            left  = t_fwd + u
            right = t_fwd - u
            phase = 'DRIVE '

        self._thrust(left, right)

        self.get_logger().info(
            f'[{phase}] WP{self._wp_idx} | '
            f'dist:{dist:.1f}m | '
            f'ψ_d:{math.degrees(psi_d):+.1f}° | '
            f'yaw:{math.degrees(self._yaw):+.1f}° | '
            f'err:{math.degrees(err):+.1f}° | '
            f'L:{left:.0f} R:{right:.0f}'
        )

    # ── LOS Guidance Law ─────────────────────────────────────────────────────

    def _los(self, cx: float, cy: float,
             wx: float, wy: float) -> float:
        """
        Line-of-Sight guidance.

        Сегмент пути: от предыдущего WP (или текущей позиции) до текущего WP.

          α   = atan2(dx_path, dy_path)          — угол пути от Севера
          e   = dx_err·cos(α) − dy_err·sin(α)    — поперечная ошибка
          ψ_d = α + atan2(−e, Δ)                 — желаемый курс
        """
        if self._wp_idx == 0:
            # Первый сегмент: нет предыдущей точки — простой bearing
            return math.atan2(wx - cx, wy - cy)

        prev_lat, prev_lon = self.WAYPOINTS[self._wp_idx - 1]
        px, py = gps_to_xy(self._lat_ref, self._lon_ref, prev_lat, prev_lon)

        dx_path = wx - px
        dy_path = wy - py
        if math.hypot(dx_path, dy_path) < 0.1:
            return math.atan2(wx - cx, wy - cy)

        alpha = math.atan2(dx_path, dy_path)   # угол пути

        # Вектор от начала сегмента до текущей позиции
        dx_e = cx - px
        dy_e = cy - py

        # Поперечная ошибка (> 0 → катер правее пути)
        e = dx_e * math.cos(alpha) - dy_e * math.sin(alpha)

        return alpha + math.atan2(-e, self.LOS_LOOKAHEAD_M)

    # ── Публикация тяги ───────────────────────────────────────────────────────

    def _thrust(self, left: float, right: float):
        lim = 300.0
        lm = Float64(); lm.data = max(-lim, min(lim, left))
        rm = Float64(); rm.data = max(-lim, min(lim, right))
        self._pub_l.publish(lm)
        self._pub_r.publish(rm)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = WaypointFollowerLOSIMU()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._thrust(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
