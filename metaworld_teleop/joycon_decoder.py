"""
JoyCon HID Report Decoder for Web Teleoperation.

Decodes raw 0x30 input reports (received as bytes from WebHID via WebSocket)
using the same button/stick/IMU layout as third_party/joycon-robotics.

The key difference from the third-party library:
- No direct HID/Bluetooth connection needed on the server.
- Raw bytes arrive from the browser via WebSocket (base64-encoded).
- This module is a pure-Python stateful decoder — no `hid`, `glm` or hardware deps.

Button mapping (Right JoyCon):
  Stick vertical  → forward/backward (Y axis)
  Stick horizontal→ left/right (X axis)
  R button        → up (+Z)
  R-stick press   → down (-Z)
  ZR (trigger)    → toggle gripper open/close
  A               → start recording
  B               → stop recording
  X               → next scene (task)
  Y               → previous scene (task)
  Home            → env reset
  IMU gyro+accel  → end-effector orientation (roll/pitch/yaw via complementary filter)

NOTE: Metaworld uses a 4-DOF action space [dx, dy, dz, gripper].
  End-effector orientation (from IMU) is applied experimentally via MuJoCo
  mocap body manipulation, but is NOT included in the recorded action data.
"""

import math
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np

logger = logging.getLogger("joycon_decoder")

# ---- Constants (from third_party/joycon-robotics/joyconrobotics/constants.py) ----
JOYCON_L_PRODUCT_ID = 0x2006
JOYCON_R_PRODUCT_ID = 0x2007


# ---------------------------------------------------------------------------
# Low-pass filter (mirrors joyconrobotics.LowPassFilter)
# ---------------------------------------------------------------------------
class LowPassFilter:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.prev_value = 0.0

    def update(self, new_value: float) -> float:
        self.prev_value = self.alpha * new_value + (1 - self.alpha) * self.prev_value
        return self.prev_value


# ---------------------------------------------------------------------------
# Attitude estimator (mirrors joyconrobotics.AttitudeEstimator)
# ---------------------------------------------------------------------------
class AttitudeEstimator:
    """Complementary-filter attitude from accel + gyro, matching JoyconRobotics."""

    def __init__(self, common_rad: bool = True, lowpassfilter_alpha_rate: float = 0.05):
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.dt = 0.01
        self.alpha = 0.55  # complementary filter weight

        self.yaw_diff = 0.0
        self.common_rad = common_rad

        # Direction vectors for yaw tracking (unit vectors, rotated by gyro)
        self._dir_x = [1.0, 0.0, 0.0]
        self._dir_y = [0.0, 1.0, 0.0]
        self._dir_z = [0.0, 0.0, 1.0]

        lp_alpha = 0.05 * lowpassfilter_alpha_rate
        self.lpf_roll = LowPassFilter(alpha=lp_alpha)
        self.lpf_pitch = LowPassFilter(alpha=lp_alpha)

    def reset_yaw(self):
        self._dir_x = [1.0, 0.0, 0.0]
        self._dir_y = [0.0, 1.0, 0.0]
        self._dir_z = [0.0, 0.0, 1.0]

    def set_yaw_diff(self, data: float):
        self.yaw_diff = data

    # -- tiny rotation helpers (avoid glm dependency) --
    @staticmethod
    def _axis_angle_rotate(vec: List[float], axis: List[float], angle: float) -> List[float]:
        """Rodrigues' rotation of *vec* around *axis* by *angle* (radians)."""
        c = math.cos(angle)
        s = math.sin(angle)
        dot = sum(a * b for a, b in zip(vec, axis))
        cross = [
            axis[1] * vec[2] - axis[2] * vec[1],
            axis[2] * vec[0] - axis[0] * vec[2],
            axis[0] * vec[1] - axis[1] * vec[0],
        ]
        return [
            c * vec[i] + s * cross[i] + (1 - c) * dot * axis[i]
            for i in range(3)
        ]

    def update(self, gyro_in_rad: Tuple[float, float, float],
               accel_in_g: Tuple[float, float, float]) -> List[float]:
        """Return [roll, pitch, yaw] in radians, matching JoyconRobotics convention."""
        self.pitch = 0.0
        self.roll = 0.0

        ax, ay, az = accel_in_g
        ax *= math.pi
        ay *= math.pi
        az *= math.pi

        gx, gy, gz = gyro_in_rad

        # Accelerometer-derived angles
        roll_acc = math.atan2(ay, -az)
        pitch_acc = math.atan2(ax, math.sqrt(ay ** 2 + az ** 2))

        # Gyro integration
        self.pitch += gy * self.dt
        self.roll -= gx * self.dt

        # Complementary filter
        self.pitch = self.alpha * self.pitch + (1 - self.alpha) * pitch_acc
        self.roll = self.alpha * self.roll + (1 - self.alpha) * roll_acc

        # Low-pass
        self.pitch = self.lpf_pitch.update(self.pitch)
        self.roll = self.lpf_roll.update(self.roll)

        # Yaw: integrate gyro to rotate direction vectors (same factor as library)
        factor = -1.0 / 86.0
        for axis, angle in [(self._dir_x, gx * factor),
                            (self._dir_y, gy * factor),
                            (self._dir_z, gz * factor)]:
            self._dir_x = self._axis_angle_rotate(self._dir_x, axis, angle)
            self._dir_y = self._axis_angle_rotate(self._dir_y, axis, angle)
            self._dir_z = self._axis_angle_rotate(self._dir_z, axis, angle)

        self.yaw = self._dir_x[1]  # same as library

        if self.common_rad:
            self.roll = self.roll * math.pi / 1.5
            self.pitch = self.pitch * math.pi / 1.5
            self.yaw = -self.yaw * math.pi / 1.5

        self.yaw -= self.yaw_diff
        return [self.roll, self.pitch, self.yaw]


# ---------------------------------------------------------------------------
# Raw report parser (pure bit-twiddling, mirrors joycon.py)
# ---------------------------------------------------------------------------
@dataclass
class JoyConReport:
    """Parsed fields from a single 0x30 input report."""
    # Buttons — right JoyCon
    btn_y: bool = False
    btn_x: bool = False
    btn_b: bool = False
    btn_a: bool = False
    btn_sr_r: bool = False
    btn_sl_r: bool = False
    btn_r: bool = False
    btn_zr: bool = False
    # Shared buttons
    btn_minus: bool = False
    btn_plus: bool = False
    btn_r_stick: bool = False
    btn_l_stick: bool = False
    btn_home: bool = False
    btn_capture: bool = False
    # Buttons — left JoyCon
    btn_down: bool = False
    btn_up: bool = False
    btn_right: bool = False
    btn_left: bool = False
    btn_sr_l: bool = False
    btn_sl_l: bool = False
    btn_l: bool = False
    btn_zl: bool = False
    # Sticks (12-bit, 0-4095, center ~2048)
    stick_lh: int = 0
    stick_lv: int = 0
    stick_rh: int = 0
    stick_rv: int = 0
    # IMU first sample (raw int16)
    accel_x: int = 0
    accel_y: int = 0
    accel_z: int = 0
    gyro_x: int = 0
    gyro_y: int = 0
    gyro_z: int = 0


def _bit(byte: int, offset: int) -> bool:
    return bool((byte >> offset) & 1)


def _int16le(hi: int, lo: int) -> int:
    """Convert two bytes to signed 16-bit little-endian (same as joycon.py)."""
    val = (lo << 8) | hi
    return val if val < 32768 else val - 65536


def parse_report(data: bytes) -> JoyConReport:
    """Parse a 0x30 input report payload.

    The *data* here is the payload **after** the reportId byte.
    WebHID strips the reportId, so data[0] corresponds to joycon.py byte index 1
    (i.e. timer byte).  Mapping:
        WebHID data[n]  ←→  joycon.py  self._input_report[n+1]

    In joycon.py the full 49-byte report is indexed [0..48] where [0]=reportId=0x30.
    Button bytes are at indices 3,4,5 → data offsets 2,3,4.
    """
    r = JoyConReport()

    if len(data) < 12:
        return r  # too short

    b3 = data[2]  # Right buttons
    b4 = data[3]  # Shared buttons
    b5 = data[4]  # Left buttons

    # Right JoyCon buttons (byte 3)
    r.btn_y = _bit(b3, 0)
    r.btn_x = _bit(b3, 1)
    r.btn_b = _bit(b3, 2)
    r.btn_a = _bit(b3, 3)
    r.btn_sr_r = _bit(b3, 4)
    r.btn_sl_r = _bit(b3, 5)
    r.btn_r = _bit(b3, 6)
    r.btn_zr = _bit(b3, 7)

    # Shared buttons (byte 4)
    r.btn_minus = _bit(b4, 0)
    r.btn_plus = _bit(b4, 1)
    r.btn_r_stick = _bit(b4, 2)
    r.btn_l_stick = _bit(b4, 3)
    r.btn_home = _bit(b4, 4)
    r.btn_capture = _bit(b4, 5)

    # Left JoyCon buttons (byte 5)
    r.btn_down = _bit(b5, 0)
    r.btn_up = _bit(b5, 1)
    r.btn_right = _bit(b5, 2)
    r.btn_left = _bit(b5, 3)
    r.btn_sr_l = _bit(b5, 4)
    r.btn_sl_l = _bit(b5, 5)
    r.btn_l = _bit(b5, 6)
    r.btn_zl = _bit(b5, 7)

    # Left stick (bytes 6,7,8 → data offsets 5,6,7)
    if len(data) > 7:
        r.stick_lh = data[5] | ((data[6] & 0x0F) << 8)
        r.stick_lv = (data[6] >> 4) | (data[7] << 4)

    # Right stick (bytes 9,10,11 → data offsets 8,9,10)
    if len(data) > 10:
        r.stick_rh = data[8] | ((data[9] & 0x0F) << 8)
        r.stick_rv = (data[9] >> 4) | (data[10] << 4)

    # IMU first sample (bytes 13-24 → data offsets 12-23)
    if len(data) > 23:
        r.accel_x = _int16le(data[12], data[13])
        r.accel_y = _int16le(data[14], data[15])
        r.accel_z = _int16le(data[16], data[17])
        r.gyro_x = _int16le(data[18], data[19])
        r.gyro_y = _int16le(data[20], data[21])
        r.gyro_z = _int16le(data[22], data[23])

    return r


# ---------------------------------------------------------------------------
# Stateful decoder → action mapping  (mirrors JoyconRobotics.common_update)
# ---------------------------------------------------------------------------
@dataclass
class JoyConState:
    """Persistent state for JoyCon-to-robot action mapping."""
    # Which controller
    is_right: bool = True

    # Position (delta-based, mapped to Metaworld 4-DOF action)
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    speed: float = 0.1  # Will be synced from TeleopState.speed

    # Gripper
    gripper_open_val: float = -1.0   # Metaworld convention: -1 = open
    gripper_close_val: float = 1.0   # +1 = closed
    gripper_state: float = -1.0      # current
    _prev_zr: bool = False           # for edge detection

    # Button control
    button_control: int = 0  # 0=none, 1=record_start, -1=record_stop
    scene_control: int = 0   # 0=none, 1=next_scene, -1=prev_scene
    reset_requested: bool = False
    _prev_home: bool = False
    _prev_a: bool = False
    _prev_b: bool = False
    _prev_x: bool = False
    _prev_y: bool = False

    # Orientation estimator
    attitude: AttitudeEstimator = field(default_factory=AttitudeEstimator)
    orientation_rad: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    _prev_orientation_rad: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Direction vectors derived from orientation (for stick→world mapping)
    direction_vector: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    direction_vector_right: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])

    # Config flags matching JoyconRobotics defaults
    pure_z: bool = True

    # Scaling: JoyconRobotics uses 0.001 per tick at ~100 Hz.
    # Our WebHID reports arrive at 60 Hz (0x30 mode).
    # We map to Metaworld action space [-1,1] directly, so scale differently.
    _STICK_SPEED: float = 1.0   # normalized; will be multiplied by state.speed later
    _BUTTON_SPEED: float = 1.0

    def reset(self):
        """Reset mutable state (gripper, buttons, orientation) on episode reset."""
        self.gripper_state = self.gripper_open_val
        self._prev_zr = False
        self.button_control = 0
        self.scene_control = 0
        self.reset_requested = False
        self._prev_home = False
        self._prev_a = False
        self._prev_b = False
        self._prev_x = False
        self._prev_y = False
        self.orientation_rad = [0.0, 0.0, 0.0]
        self._prev_orientation_rad = [0.0, 0.0, 0.0]
        self.attitude.reset_yaw()

    def process_report(self, report: JoyConReport) -> dict:
        """Process one parsed report. Returns dict with action info.

        Returns:
            {
                'dx': float, 'dy': float, 'dz': float,
                'gripper': float,
                'orientation_delta': [roll, pitch, yaw],
                'button_control': int,   # 0=none, 1=record_start, -1=record_stop
                'scene_control': int,    # 0=none, 1=next_scene, -1=prev_scene
                'reset': bool,
            }
        """
        # --- Orientation from IMU ---
        # Convert raw IMU to physical units (same constants as wrappers.py)
        ACCEL_G = 4.0 / 0x4000
        GYRO_RAD = 0.0001694 * math.pi

        accel_in_g = (
            report.accel_x * ACCEL_G,
            report.accel_y * ACCEL_G,
            report.accel_z * ACCEL_G,
        )
        gyro_in_rad = (
            report.gyro_x * GYRO_RAD,
            report.gyro_y * GYRO_RAD,
            report.gyro_z * GYRO_RAD,
        )

        self._prev_orientation_rad = list(self.orientation_rad)
        self.orientation_rad = self.attitude.update(gyro_in_rad, accel_in_g)
        roll, pitch, yaw = self.orientation_rad

        # Direction vectors for stick-to-world mapping
        self.direction_vector = [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ]
        self.direction_vector_right = [
            math.cos(roll) * math.sin(-yaw),
            math.cos(roll) * math.cos(-yaw),
            math.sin(-roll),
        ]

        # --- Stick → position delta ---
        dx, dy, dz = 0.0, 0.0, 0.0

        # Pick the correct stick depending on L/R
        if self.is_right:
            stick_v = report.stick_rv
            stick_h = report.stick_rh
        else:
            stick_v = report.stick_lv
            stick_h = report.stick_lh

        # Forward/backward (stick vertical, deadzone: center±~1000)
        # NOTE: signs inverted — stick-up = move backward, stick-down = move forward
        if stick_v > 3000:
            fwd = (stick_v - 2048) / 2048.0  # 0..~1
            dx -= fwd * self.direction_vector[0] * self._STICK_SPEED
            dy -= fwd * self.direction_vector[1] * self._STICK_SPEED
            if self.pure_z:
                pass  # Z controlled by buttons
            else:
                dz -= fwd * self.direction_vector[2] * self._STICK_SPEED
        elif stick_v < 1000:
            bwd = (2048 - stick_v) / 2048.0
            dx += bwd * self.direction_vector[0] * self._STICK_SPEED
            dy += bwd * self.direction_vector[1] * self._STICK_SPEED
            if not self.pure_z:
                dz += bwd * self.direction_vector[2] * self._STICK_SPEED

        # Left/right (stick horizontal → sideways via direction_vector_right)
        # NOTE: signs inverted — stick-left = move right, stick-right = move left
        if stick_h > 3000:
            side = (stick_h - 2048) / 2048.0
            dx += side * self.direction_vector_right[0] * self._STICK_SPEED
            dy += side * self.direction_vector_right[1] * self._STICK_SPEED
        elif stick_h < 1000:
            side = (2048 - stick_h) / 2048.0
            dx -= side * self.direction_vector_right[0] * self._STICK_SPEED
            dy -= side * self.direction_vector_right[1] * self._STICK_SPEED

        # Up/down via buttons (R/L = up, R-stick/L-stick = down)
        btn_up = report.btn_r if self.is_right else report.btn_l
        btn_down = report.btn_r_stick if self.is_right else report.btn_l_stick
        if btn_up:
            dz += self._BUTTON_SPEED
        if btn_down:
            dz -= self._BUTTON_SPEED

        # --- Gripper toggle (ZR/ZL edge detection) ---
        btn_zr_now = report.btn_zr if self.is_right else report.btn_zl
        if btn_zr_now and not self._prev_zr:
            # Toggle
            if self.gripper_state == self.gripper_open_val:
                self.gripper_state = self.gripper_close_val
            else:
                self.gripper_state = self.gripper_open_val
        self._prev_zr = btn_zr_now

        # --- Button control ---
        self.button_control = 0
        self.scene_control = 0
        self.reset_requested = False

        # Home → env reset (edge detection)
        btn_home_now = report.btn_home if self.is_right else report.btn_capture
        if btn_home_now and not self._prev_home:
            self.reset_requested = True
            # Also reset attitude
            self.attitude.reset_yaw()
        self._prev_home = btn_home_now

        # A → start recording (edge)
        if report.btn_a and not self._prev_a:
            self.button_control = 1  # record_start
        self._prev_a = report.btn_a

        # B → stop recording (edge)
        if report.btn_b and not self._prev_b:
            self.button_control = -1  # record_stop
        self._prev_b = report.btn_b

        # X → next scene/task (edge)
        if report.btn_x and not self._prev_x:
            self.scene_control = 1  # next_scene
        self._prev_x = report.btn_x

        # Y → previous scene/task (edge)
        if report.btn_y and not self._prev_y:
            self.scene_control = -1  # prev_scene
        self._prev_y = report.btn_y

        # Compute orientation delta (frame-to-frame change) for incremental
        # application in apply_orientation().
        ori_delta = [
            self.orientation_rad[0] - self._prev_orientation_rad[0],
            self.orientation_rad[1] - self._prev_orientation_rad[1],
            self.orientation_rad[2] - self._prev_orientation_rad[2],
        ]

        return {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "gripper": self.gripper_state,
            "orientation_delta": ori_delta,
            "button_control": self.button_control,
            "scene_control": self.scene_control,
            "reset": self.reset_requested,
        }
