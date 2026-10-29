# Invisible Man — Python + OpenCV

A real-time webcam effect that automatically captures the empty background, detects a coloured cloth, and replaces it with the background to create an "invisibility cloak" illusion.

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

1. Start the program and immediately step completely out of view.
2. The empty background is captured automatically during the three-second countdown.
3. Return holding or wearing a bright, single-colour cloth.
4. Hold it clearly in view for a moment. The program detects its dominant colour and starts the effect automatically.
5. No click or key press is required. The sliders are available only for fine-tuning difficult lighting.

### Controls

| Control | Action |
| --- | --- |
| Left click | Optional manual colour override |
| `R` or `B` | Recapture the empty background |
| `A` | Return to automatic colour detection |
| `M` | Show/hide the detected-colour mask |
| `Q` or `Esc` | Quit |

The **Hue tolerance** slider expands the detected colour range. **Min saturation** rejects grey/white regions, **Min brightness** rejects dark shadows, and **Scene difference** controls how different an object must be from the empty background.

## Tips for a better effect

- Use a cloth colour that does not appear on your skin or clothes.
- Avoid wrinkles and strong shadows on the cloth.
- Use soft, even lighting.
- Do not move the webcam after capturing the background.
- Use the mask view (`M`) while tuning the sliders.

## Privacy

All video processing happens locally. The application does not record or upload camera frames.
