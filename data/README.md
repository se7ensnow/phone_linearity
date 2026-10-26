# Accelerometer Data

Measurements were recorded with the phyphox app, experiment **Acceleration with g** (gravity included).

**Device:** Xiaomi Redmi Note 13 Pro+ 5G (Android). The same phone, **with its case on**, was used for all recordings.

## Recordings (17 files, all recorded on 2026-10-20)

- **12 static poses**: two sessions of six orientations (screen up, screen down, and left, right, bottom and top edge down). About 20 s per recording, phone not touched while recording.
- **4 tilt recordings**: the phone leaned against a wall, screen facing up, top edge always raised. h = height on the wall and L = distance along the floor, both measured with a tape (cm). The tilt angle is computed from tan θ = h/L.
- **1 limit experiment**: random shaking by hand, about 21 s (`Random.csv`).

| File | h (cm) | L (cm) | Angle from h and L |
|---|---|---|---|
| `h=6,l=16,angle=22.csv` | 6 | 16 | 20.6° |
| `h=7,l=15,angle=27.5.csv` | 7 | 15 | 25.0° |
| `h=8,l=14.5,angle=33.5.csv` | 8 | 14.5 | 28.9° |
| `h=9,l=14,angle=40.csv` | 9 | 14 | 32.7° |

The angles typed in these file names do not match h and L. The file names were left unchanged because raw data must not be edited. **The angle computed from h and L is the one used in the analysis.** The tape measurement is uncertain by about ±2.2° to ±2.4°.

## File names

Files are named `<pose>.csv` for session 1 and `<pose>_2.csv` for session 2 (for example `screen_up.csv` and `screen_up_2.csv`). This is not the `team_pose_session` pattern suggested in the project description.

## CSV columns

- `Time (s)`: time in seconds.
- `Acceleration x (m/s^2)`: x-axis acceleration.
- `Acceleration y (m/s^2)`: y-axis acceleration.
- `Acceleration z (m/s^2)`: z-axis acceleration.
- `Absolute acceleration (m/s^2)`: magnitude, computed by phyphox. It is not used for fitting.

All values are in m/s² and include gravity.

## Axis convention

With the phone screen up: x points to the right of the screen, y toward the top edge, z out of the screen. At rest, the phone reads +g along the direction that points up. So screen up gives a = (0, 0, +g).

## Use of the data

Session 1 is used to fit the model. Session 2 and the tilt recordings are used for validation. The shaking recording is used to show where the model fails.

Original recordings are stored unchanged in `data/raw/`. Metadata for each file is in `data/recordings.csv`.