# Invisible Man — Python + OpenCV

A real-time webcam effect that replaces a coloured cloth with a previously captured background, creating an "invisibility cloak" illusion.

## Requirements

- Python 3.10 or newer
- A webcam
- A single-colour cloth (bright red, blue, or green works best)
- Even lighting and a stationary camera

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python invisible_man.py
```

If the wrong camera opens, choose another camera index:

```powershell
python invisible_man.py --camera 1
```

## How to use

1. Keep the camera still and press **B**.
2. Step completely out of view while the background is captured.
3. Return holding or wearing a single-colour cloth.
4. Click a well-lit area in the middle of the cloth to select its colour.
5. Adjust the three sliders until the entire cloth disappears cleanly.

### Controls

| Control | Action |
| --- | --- |
| Left click | Select the cloak colour |
| `B` | Recapture the empty background |
| `M` | Show/hide the detected-colour mask |
| `Q` or `Esc` | Quit |

The **Hue tolerance** slider expands the selected colour range. **Min saturation** rejects grey/white regions, while **Min brightness** rejects dark shadows.

## Tips for a better effect

- Use a cloth colour that does not appear on your skin or clothes.
- Avoid wrinkles and strong shadows on the cloth.
- Use soft, even lighting.
- Do not move the webcam after capturing the background.
- Use the mask view (`M`) while tuning the sliders.

## Privacy

All video processing happens locally. The application does not record or upload camera frames.
