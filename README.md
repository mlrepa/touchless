# touchless
"Touchless" is a Python library for creating spatial interfaces without physical contact, using gestures, eye tracking, and voice commands. It enables developers to build immersive 3D UIs for VR/AR and beyond, with advanced gesture and voice recognition technology.


## Environment Setup
```bash
python3 -m venv .venv
echo "export PYTHONPATH=$PWD" >> .venv/bin/activate
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Demo scripts

### Demo for all gestures

```bash
python examples/demo.py
```

Description: print (in OpenCV window) all detected gestures


### Turn on/off video stream output and blur

```bash
python examples/turn_on_off_and_blur.py
```

Recognized hand gestures:

| Gesture | Action |
| :----: | :----: |
| Hand down | Turn off video output |
| Hand up | Turn on video output |
| Three fingers | Turn on blur |
| Two fingers | Turn off blur |


### Select shape

```bash
python examples/select_shape.py
```

Description:
- *pointer* is the index finger TIP
- if the *pointer* is over a shape then the shape will be selected


### Select shape with pointer mapping from the specific region

```bash
python examples/select_shape_map_pointer.py
```

Description:
- *pointer* is the index finger TIP
- if the *pointer* inside the center box then it will be mapped to pointer which can select shapes


### Calculate angle

```bash
python examples/calc_angle.py [-x <x-coord>, [-y <y-coord>]]
```

where: *-x* and *-y* are user defined coordinates of an angle vertex; by default *x*, *y* = frame_width // 2, 0.

Description:
- *pointer* is the index finger TIP
- the first side of an *angle* is a vertical line - from vertex to bottom
- the second side is a line connected *angle* vertex with the *pointer*
- if the *pointer* detected then:
    - the second *angle* side will be drawn
    - the *angle* value (in degrees) will be calculated


### Triangle by points of wrist and index and pinky fingers MCP

```bash
python examples/triangle_0_5_17_area.py [--width <frame_width> [--height <frame_height>]]
```

Calculates:
- the triangle area (in pixels)
- a distance to a camera (approximately)
- angle between the triangle and camera by z-axis

**Note**: the distance to a camera is calculated with assuming a palm (the triangle) is parallel (as well as possible) to the camera; for now the angle by z-axis is not considered.


### Multiple hands using GestureProvider

```bash
python examples/multiple_hands.py
```

Description: detects all hands and gestures by its point, outputs detected gesture for each hand.
