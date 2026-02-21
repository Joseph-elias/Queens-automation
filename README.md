# Queens Automation Solving

Live assistant for the Queens puzzle on Windows.

`queens_live_solver.py` detects the board from your live screen, solves it, and clicks the solution cells on the game board.

## Features

- Live screen preview (`mss`)
- Automatic board detection and perspective warp
- Automatic grid size detection (`N`)
- Region extraction from board colors
- Queens solver with constraints:
  - 1 queen per row
  - 1 queen per column
  - 1 queen per region
  - no touching queens (8-neighborhood)
- Detects pre-placed queen(s) and respects them
- Auto-clicks solved cells (double-click by default)
- Optional debug window

## Requirements

- Windows
- Python 3.10+ (3.11 recommended)

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python queens_live_solver.py
```

## Controls

- `S`: solve and click immediately
- `R`: re-run solve and click
- `C`: re-click last solved points
- `Q`: quit

## Useful Options

```powershell
python queens_live_solver.py --debug
python queens_live_solver.py --fallback-n 8
python queens_live_solver.py --dbscan-eps 12
python queens_live_solver.py --clicks-per-cell 2 --click-delay 0 --click-countdown 0
```

## Notes

- Keep the puzzle fully visible on your primary monitor.
- If clicks are slightly off, keep Windows display scaling consistent (100%/125%) and rerun.
- `pyautogui` failsafe is enabled: moving the mouse to the top-left corner can abort rapid clicking.

## Project Files

- `queens_live_solver.py`: main script
- `requirements.txt`: Python dependencies

