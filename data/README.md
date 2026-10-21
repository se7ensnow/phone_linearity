
# Accelerometer Data

Measurements were recorded using the phyphox app in **Acceleration with g** mode.

**Device:** [Phone model]

**Measurement protocol:**
- Two sessions of six static orientations: screen up/down, top/bottom/left/right edge down.
- Four tilt measurements using a raised surface:
  - h = 6 cm, L = 16 cm, angle = 22°
  - h = 7 cm, L = 15 cm, angle = 27.5°
  - h = 8 cm, L = 14.5 cm, angle = 33.5°
  - h = 9 cm, L = 14 cm, angle = 40°
- One additional recording with strong shaking to investigate the model's limitations.
- Approximately 20 seconds per recording.

**CSV columns:**
- `Time (s)` — time in seconds.
- `Acceleration x (m/s^2)` — X-axis acceleration.
- `Acceleration y (m/s^2)` — Y-axis acceleration.
- `Acceleration z (m/s^2)` — Z-axis acceleration.
- `Absolute acceleration (m/s^2)` — acceleration magnitude.

All acceleration values are in m/s² and include gravity.

Original recordings are stored unchanged in `data/raw/`. Recording metadata is stored in `data/recordings.csv`.

The first static session is used for calibration, while the second session and tilt measurements are used for validation.
