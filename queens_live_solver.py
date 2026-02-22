#!/usr/bin/env python3
"""
Install:
  pip install mss opencv-python numpy scikit-learn pyautogui
Run:
  python queens_live_solver.py
"""

from __future__ import annotations

import argparse
import time
import ctypes
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
from mss import mss

try:
    from sklearn.cluster import DBSCAN
except Exception:
    DBSCAN = None  # type: ignore

try:
    import pyautogui
except Exception:
    pyautogui = None  # type: ignore


@dataclass
class SolveResult:
    screen_bgr: np.ndarray
    monitor_left: int
    monitor_top: int
    board_quad: np.ndarray
    warped: np.ndarray
    n: int
    region_grid: np.ndarray
    fixed_queens: list[tuple[int, int]]
    placements: Optional[list[tuple[int, int]]]
    screen_points: Optional[list[tuple[float, float]]]
    debug_images: dict[str, np.ndarray]


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Expected four 2D points")

    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def set_dpi_awareness() -> None:
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
    except Exception:
        pass


def capture_primary_monitor(sct: mss) -> tuple[np.ndarray, int, int]:
    monitor = sct.monitors[1]
    raw = np.array(sct.grab(monitor), dtype=np.uint8)
    bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    return bgr, int(monitor.get("left", 0)), int(monitor.get("top", 0))


def detect_board_quad(screen_bgr: np.ndarray) -> tuple[Optional[np.ndarray], dict[str, np.ndarray]]:
    gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    _, dark_mask = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)
    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    min_area = 0.02 * (h * w)

    best_score = -1.0
    best_quad: Optional[np.ndarray] = None

    dbg = screen_bgr.copy()

    # Stage 1: connected dark-line component, robust for thin board lines.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    for label_id in range(1, n_labels):
        x, y, bw, bh, px_area = stats[label_id]
        if bw <= 0 or bh <= 0:
            continue

        box_area = float(bw * bh)
        if box_area < min_area:
            continue

        ratio = bw / float(bh)
        if ratio < 0.75 or ratio > 1.25:
            continue

        comp = (labels[y : y + bh, x : x + bw] == label_id).astype(np.uint8) * 255
        line_density = float(px_area) / max(1.0, box_area)
        # Prefer big square-ish components with moderate dark-line density.
        density_score = max(0.05, min(0.45, line_density))
        score = box_area * (1.0 - abs(1.0 - ratio)) * density_score

        if score > best_score:
            best_score = score
            best_quad = np.array(
                [[x, y], [x + bw - 1, y], [x + bw - 1, y + bh - 1], [x, y + bh - 1]],
                dtype=np.float32,
            )

    # Stage 2 fallback: contour-based candidates.
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        ratio = bw / float(bh)
        if ratio < 0.75 or ratio > 1.25:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(cnt)
            quad = cv2.boxPoints(rect).astype(np.float32)

        border_mask = np.zeros_like(gray)
        cv2.drawContours(border_mask, [quad.astype(np.int32)], -1, 255, thickness=8)
        mean_border = cv2.mean(gray, mask=border_mask)[0]
        border_darkness = (255.0 - mean_border) / 255.0

        score = area * max(0.1, border_darkness)
        if score > best_score:
            best_score = score
            best_quad = quad

    if best_quad is not None:
        cv2.polylines(dbg, [best_quad.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)

    return best_quad, {"edges": edges, "dark_mask": dark_mask, "board_detect": dbg}


def warp_board(image: np.ndarray, quad: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = order_points(quad)
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(src, dst)
    minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, m, (size, size), flags=cv2.INTER_LINEAR)
    return warped, m, minv


def _group_peak_indices(indices: np.ndarray, max_gap: int = 2) -> list[int]:
    if indices.size == 0:
        return []
    groups: list[list[int]] = [[int(indices[0])]]
    for idx in indices[1:]:
        i = int(idx)
        if i - groups[-1][-1] <= max_gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [int(round(float(np.mean(g)))) for g in groups]


def detect_grid_size(warped: np.ndarray, min_n: int = 4, max_n: int = 20) -> tuple[Optional[int], dict[str, Any]]:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, dark_mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_mask_fix = cv2.inRange(gray, 0, 110)
    dark_mask = cv2.bitwise_or(dark_mask_otsu, dark_mask_fix)

    h, w = dark_mask.shape
    vk = max(8, h // 45)
    hk = max(8, w // 45)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))

    v_lines = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, v_kernel)
    h_lines = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, h_kernel)

    proj_x = v_lines.sum(axis=0).astype(np.float32)
    proj_y = h_lines.sum(axis=1).astype(np.float32)

    def line_positions(proj: np.ndarray, length: int) -> list[int]:
        if float(np.max(proj)) <= 0:
            return []
        threshold = 0.28 * float(np.max(proj))
        idx = np.where(proj >= threshold)[0]
        peaks = _group_peak_indices(idx, max_gap=3)
        if len(peaks) < 2:
            return peaks

        min_spacing = max(8, int(length / (max_n + 2)))
        filtered = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered[-1] >= min_spacing:
                filtered.append(p)
        return filtered

    x_lines = line_positions(proj_x, w)
    y_lines = line_positions(proj_y, h)

    nx = len(x_lines) - 1
    ny = len(y_lines) - 1

    n_candidates = [v for v in [nx, ny] if min_n <= v <= max_n]
    n: Optional[int] = None

    if n_candidates and abs(nx - ny) <= 1:
        n = int(round((nx + ny) / 2.0))
    elif n_candidates:
        n = int(sorted(n_candidates)[len(n_candidates) // 2])

    proj_vis = np.zeros((400, 400, 3), dtype=np.uint8)
    if proj_x.size > 0 and np.max(proj_x) > 0:
        xs = np.linspace(0, 399, proj_x.size).astype(np.int32)
        ys = 399 - (proj_x / np.max(proj_x) * 380).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(proj_vis, [pts], False, (0, 220, 255), 1, cv2.LINE_AA)
    if proj_y.size > 0 and np.max(proj_y) > 0:
        ys2 = np.linspace(0, 399, proj_y.size).astype(np.int32)
        xs2 = (proj_y / np.max(proj_y) * 380).astype(np.int32)
        pts2 = np.stack([xs2, ys2], axis=1).reshape(-1, 1, 2)
        cv2.polylines(proj_vis, [pts2], False, (255, 180, 0), 1, cv2.LINE_AA)
    cv2.putText(proj_vis, f"x-lines={len(x_lines)}, y-lines={len(y_lines)}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    debug = {
        "dark_mask": dark_mask,
        "v_lines": v_lines,
        "h_lines": h_lines,
        "proj_vis": proj_vis,
        "x_lines": x_lines,
        "y_lines": y_lines,
    }
    return n, debug


def cluster_region_colors(features: np.ndarray, target_n: int, eps: float) -> tuple[np.ndarray, int]:
    if DBSCAN is None:
        rounded = np.round(features / 6.0).astype(np.int32)
        _, labels = np.unique(rounded, axis=0, return_inverse=True)
        return labels.astype(np.int32), int(np.unique(labels).size)

    eps_candidates = [eps, eps * 0.8, eps * 1.2, eps + 4.0, max(3.0, eps - 4.0), 8.0, 16.0]
    best_labels: Optional[np.ndarray] = None
    best_k = 10**9
    best_dist = 10**9

    for e in eps_candidates:
        model = DBSCAN(eps=float(e), min_samples=1)
        labels = model.fit_predict(features)
        unique = np.unique(labels)
        k = int(unique.size)
        dist = abs(k - target_n)
        if dist < best_dist:
            best_dist = dist
            best_k = k
            best_labels = labels
            if dist == 0:
                break

    if best_labels is None:
        raise RuntimeError("DBSCAN clustering failed")

    return best_labels.astype(np.int32), best_k


def merge_labels_to_target(labels: np.ndarray, features: np.ndarray, target_n: int) -> tuple[np.ndarray, int]:
    labels = labels.astype(np.int32).copy()
    unique = np.unique(labels)
    if unique.size <= target_n:
        return labels, int(unique.size)

    while True:
        unique = np.unique(labels)
        k = int(unique.size)
        if k <= target_n:
            break

        counts = {int(u): int(np.sum(labels == u)) for u in unique}
        centroids = {}
        for u in unique:
            mask = labels == u
            centroids[int(u)] = features[mask].mean(axis=0)

        src = min(unique.tolist(), key=lambda u: counts[int(u)])
        src_c = centroids[int(src)]

        best_dst = None
        best_dist = float("inf")
        for u in unique:
            if int(u) == int(src):
                continue
            d = float(np.linalg.norm(src_c - centroids[int(u)]))
            if d < best_dist:
                best_dist = d
                best_dst = int(u)

        if best_dst is None:
            break
        labels[labels == int(src)] = int(best_dst)

    remap_vals = np.unique(labels)
    remap = {int(v): i for i, v in enumerate(remap_vals.tolist())}
    compact = np.array([remap[int(v)] for v in labels], dtype=np.int32)
    return compact, int(np.unique(compact).size)


def build_region_grid(warped: np.ndarray, n: int, dbscan_eps: float) -> tuple[np.ndarray, np.ndarray, int]:
    h, w = warped.shape[:2]
    cell_h = h / float(n)
    cell_w = w / float(n)

    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)

    feats: list[np.ndarray] = []
    for r in range(n):
        for c in range(n):
            y0 = int(round(r * cell_h))
            y1 = int(round((r + 1) * cell_h))
            x0 = int(round(c * cell_w))
            x1 = int(round((c + 1) * cell_w))

            cy = (y0 + y1) // 2
            cx = (x0 + x1) // 2
            patch_r = max(2, int(min(y1 - y0, x1 - x0) * 0.22))

            py0 = max(y0, cy - patch_r)
            py1 = min(y1, cy + patch_r)
            px0 = max(x0, cx - patch_r)
            px1 = min(x1, cx + patch_r)

            patch = lab[py0:py1, px0:px1]
            if patch.size == 0:
                patch = lab[y0:y1, x0:x1]
            pix = patch.reshape(-1, 3).astype(np.float32)
            # Ignore dark icon pixels (queen glyph) so region color remains stable.
            keep = pix[:, 0] > 70
            if int(np.sum(keep)) >= max(10, int(0.25 * pix.shape[0])):
                pix = pix[keep]
            feats.append(np.median(pix, axis=0))

    features = np.array(feats, dtype=np.float32)
    labels, region_count = cluster_region_colors(features, target_n=n, eps=dbscan_eps)
    if region_count > n:
        labels, region_count = merge_labels_to_target(labels, features, target_n=n)

    unique = np.unique(labels)
    remap = {int(v): i for i, v in enumerate(unique.tolist())}
    compact = np.array([remap[int(v)] for v in labels], dtype=np.int32)
    region_grid = compact.reshape(n, n)

    region_vis = np.zeros((n, n, 3), dtype=np.uint8)
    rng = np.random.default_rng(12345)
    palette = rng.integers(30, 255, size=(len(unique), 3), dtype=np.uint8)
    for r in range(n):
        for c in range(n):
            region_vis[r, c] = palette[region_grid[r, c] % len(palette)]

    region_vis = cv2.resize(region_vis, warped.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    return region_grid, region_vis, region_count


def solve_queens(region_grid: np.ndarray) -> Optional[list[tuple[int, int]]]:
    return solve_queens_with_fixed(region_grid, fixed_queens=[])


def solve_queens_with_fixed(
    region_grid: np.ndarray,
    fixed_queens: list[tuple[int, int]],
) -> Optional[list[tuple[int, int]]]:
    n = int(region_grid.shape[0])
    region_count = int(np.unique(region_grid).size)
    if region_count != n:
        return None

    all_rows = set(range(n))
    used_cols: set[int] = set()
    used_regions: set[int] = set()
    forbidden: set[tuple[int, int]] = set()
    placements: dict[int, int] = {}

    def add_placement(r: int, c: int) -> bool:
        if r in placements:
            return False
        if c in used_cols:
            return False
        rid = int(region_grid[r, c])
        if rid in used_regions:
            return False
        if (r, c) in forbidden:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    forbidden.add((rr, cc))
        placements[r] = c
        used_cols.add(c)
        used_regions.add(rid)
        return True

    for r, c in fixed_queens:
        if not (0 <= r < n and 0 <= c < n):
            return None
        if not add_placement(r, c):
            return None

    def candidates_for_row(r: int) -> list[int]:
        out: list[int] = []
        for c in range(n):
            if c in used_cols:
                continue
            rid = int(region_grid[r, c])
            if rid in used_regions:
                continue
            if (r, c) in forbidden:
                continue
            out.append(c)
        return out

    def backtrack() -> bool:
        if len(placements) == n:
            return True

        unassigned = list(all_rows - set(placements.keys()))
        best_row = -1
        best_cands: Optional[list[int]] = None

        for r in unassigned:
            cands = candidates_for_row(r)
            if not cands:
                return False
            if best_cands is None or len(cands) < len(best_cands):
                best_row = r
                best_cands = cands

        assert best_cands is not None

        for c in best_cands:
            rid = int(region_grid[best_row, c])
            added_forbidden: set[tuple[int, int]] = set()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = best_row + dr
                    cc = c + dc
                    if 0 <= rr < n and 0 <= cc < n and (rr, cc) not in forbidden:
                        forbidden.add((rr, cc))
                        added_forbidden.add((rr, cc))

            placements[best_row] = c
            used_cols.add(c)
            used_regions.add(rid)

            if backtrack():
                return True

            del placements[best_row]
            used_cols.remove(c)
            used_regions.remove(rid)
            for pos in added_forbidden:
                forbidden.remove(pos)

        return False

    ok = backtrack()
    if not ok:
        return None
    return sorted([(r, c) for r, c in placements.items()], key=lambda x: x[0])


def detect_fixed_queens(warped: np.ndarray, n: int) -> list[tuple[int, int]]:
    h, w = warped.shape[:2]
    cell_h = h / float(n)
    cell_w = w / float(n)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    scores: list[tuple[float, int, int]] = []
    for r in range(n):
        for c in range(n):
            y0 = int(round(r * cell_h))
            y1 = int(round((r + 1) * cell_h))
            x0 = int(round(c * cell_w))
            x1 = int(round((c + 1) * cell_w))

            ph = max(3, y1 - y0)
            pw = max(3, x1 - x0)
            # Crown icon is usually near the upper-middle of the cell.
            py0 = y0 + int(0.10 * ph)
            py1 = y0 + int(0.60 * ph)
            px0 = x0 + int(0.18 * pw)
            px1 = x0 + int(0.82 * pw)
            patch = gray[max(y0, py0) : min(y1, py1), max(x0, px0) : min(x1, px1)]
            if patch.size == 0:
                continue

            dark_ratio = float(np.mean(patch < 70))
            very_dark_ratio = float(np.mean(patch < 45))
            score = 0.6 * dark_ratio + 0.4 * very_dark_ratio
            scores.append((score, r, c))

    if not scores:
        return []

    scores.sort(reverse=True, key=lambda t: t[0])
    best_score = scores[0][0]
    if best_score < 0.02:
        return []

    fixed: list[tuple[int, int]] = []
    threshold = max(0.03, best_score * 0.55)
    for sc, r, c in scores[: min(3, len(scores))]:
        if sc >= threshold:
            fixed.append((r, c))

    # Keep at most one fixed queen per row, highest score wins.
    by_row: dict[int, tuple[float, int]] = {}
    for sc, r, c in [(s, rr, cc) for s, rr, cc in scores if (rr, cc) in fixed]:
        if r not in by_row or sc > by_row[r][0]:
            by_row[r] = (sc, c)
    return sorted([(r, c) for r, (_, c) in by_row.items()], key=lambda x: x[0])


def annotate_solution(warped: np.ndarray, placements: list[tuple[int, int]], n: int) -> np.ndarray:
    out = warped.copy()
    h, w = out.shape[:2]
    cell_h = h / float(n)
    cell_w = w / float(n)

    for i in range(n + 1):
        x = int(round(i * cell_w))
        y = int(round(i * cell_h))
        cv2.line(out, (x, 0), (x, h - 1), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(out, (0, y), (w - 1, y), (255, 255, 255), 1, cv2.LINE_AA)

    for r, c in placements:
        cx = int(round((c + 0.5) * cell_w))
        cy = int(round((r + 0.5) * cell_h))
        radius = max(10, int(min(cell_h, cell_w) * 0.32))
        cv2.circle(out, (cx, cy), radius, (0, 220, 0), 3, cv2.LINE_AA)
        cv2.putText(
            out,
            "Q",
            (cx - radius // 2, cy + radius // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.8, min(2.5, min(cell_h, cell_w) / 28.0)),
            (0, 100, 0),
            3,
            cv2.LINE_AA,
        )

    return out


def queen_click_points_warped(placements: list[tuple[int, int]], n: int, size: int) -> list[tuple[float, float]]:
    step = size / float(n)
    points: list[tuple[float, float]] = []
    for r, c in placements:
        # Click slightly below center to avoid borderline misses on top-row cells.
        x_frac = 0.52
        y_frac = 0.58
        # Add a bit more inset on outer borders where perspective and line thickness are strongest.
        if c == 0:
            x_frac = max(x_frac, 0.60)
        elif c == n - 1:
            x_frac = min(x_frac, 0.40)
        if r == 0:
            y_frac = max(y_frac, 0.65)
        elif r == n - 1:
            y_frac = min(y_frac, 0.45)
        points.append(((c + x_frac) * step, (r + y_frac) * step))
    return points


def map_points_to_screen(points: list[tuple[float, float]], minv: np.ndarray, left: int = 0, top: int = 0) -> list[tuple[float, float]]:
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, minv).reshape(-1, 2)
    return [(float(x) + float(left), float(y) + float(top)) for x, y in mapped]


def run_pipeline(screen_bgr: np.ndarray, args: argparse.Namespace, monitor_left: int = 0, monitor_top: int = 0) -> SolveResult:
    quad, board_dbg = detect_board_quad(screen_bgr)
    if quad is None:
        raise RuntimeError("Board not found on screen")

    warped, _, minv = warp_board(screen_bgr, quad, args.warp_size)

    n, grid_dbg = detect_grid_size(warped, min_n=4, max_n=20)
    used_fallback = False
    if n is None:
        n = int(args.fallback_n)
        used_fallback = True

    region_grid, region_vis, region_count = build_region_grid(warped, n=n, dbscan_eps=args.dbscan_eps)
    fixed_queens = detect_fixed_queens(warped, n=n)
    placements = solve_queens_with_fixed(region_grid, fixed_queens=fixed_queens)

    screen_points = None
    if placements:
        warped_pts = queen_click_points_warped(placements, n=n, size=args.warp_size)
        screen_points = map_points_to_screen(warped_pts, minv, left=monitor_left, top=monitor_top)

    dbg: dict[str, np.ndarray] = {}
    dbg.update(board_dbg)
    dbg["dark_mask"] = grid_dbg["dark_mask"]
    dbg["v_lines"] = grid_dbg["v_lines"]
    dbg["h_lines"] = grid_dbg["h_lines"]
    dbg["proj_vis"] = grid_dbg["proj_vis"]
    dbg["region_map"] = region_vis

    print("[INFO] Board detected.")
    if used_fallback:
        print(f"[WARN] Grid size detection failed, using fallback N={n}.")
    else:
        print(f"[INFO] Detected N={n}.")
    print(f"[INFO] Region clusters found: {region_count}.")
    if region_count != n:
        print(f"[WARN] Region count ({region_count}) != N ({n}); puzzle may be unsatisfiable.")

    if placements is None:
        print("[WARN] No solution found.")
    else:
        print(f"[INFO] Solution found with {len(placements)} queens.")
    if fixed_queens:
        print(f"[INFO] Detected fixed queen(s): {fixed_queens}")

    return SolveResult(
        screen_bgr=screen_bgr,
        monitor_left=monitor_left,
        monitor_top=monitor_top,
        board_quad=quad,
        warped=warped,
        n=n,
        region_grid=region_grid,
        fixed_queens=fixed_queens,
        placements=placements,
        screen_points=screen_points,
        debug_images=dbg,
    )


def make_debug_view(result: SolveResult) -> np.ndarray:
    board = result.debug_images["board_detect"]
    warped_shape = result.warped.shape[:2][::-1]

    dark = cv2.cvtColor(result.debug_images["dark_mask"], cv2.COLOR_GRAY2BGR)
    v_lines = cv2.cvtColor(result.debug_images["v_lines"], cv2.COLOR_GRAY2BGR)
    h_lines = cv2.cvtColor(result.debug_images["h_lines"], cv2.COLOR_GRAY2BGR)
    proj = result.debug_images["proj_vis"]
    region = result.debug_images["region_map"]

    dark = cv2.resize(dark, warped_shape)
    v_lines = cv2.resize(v_lines, warped_shape)
    h_lines = cv2.resize(h_lines, warped_shape)
    proj = cv2.resize(proj, warped_shape)
    board_small = cv2.resize(board, warped_shape)

    top = np.hstack([board_small, dark, v_lines])
    bottom = np.hstack([h_lines, proj, region])
    debug = np.vstack([top, bottom])
    cv2.putText(debug, "Debug", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Queens puzzle detector and solver with live auto-click on solve.")
    parser.add_argument("--fallback-n", type=int, default=10, help="Fallback grid size if auto-detection fails.")
    parser.add_argument("--warp-size", type=int, default=700, help="Square size (pixels) for perspective-warped board.")
    parser.add_argument("--dbscan-eps", type=float, default=12.0, help="DBSCAN eps in LAB color space.")
    parser.add_argument("--debug", action="store_true", help="Show optional debug visualizations.")
    parser.add_argument("--click-delay", type=float, default=0.0, help="Delay in seconds between clicks.")
    parser.add_argument("--click-countdown", type=float, default=0.0, help="Countdown before click sequence starts.")
    parser.add_argument("--clicks-per-cell", type=int, default=2, help="Click count per solved cell (default 2).")
    return parser.parse_args()


def click_screen_points(points: list[tuple[float, float]], countdown: float, delay: float, clicks_per_cell: int) -> None:
    if pyautogui is None:
        raise RuntimeError("pyautogui is not installed. Install it with: pip install pyautogui")
    if not points:
        return

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0
    print(f"[INFO] Clicking will start in {countdown:.1f}s. Move mouse to top-left corner to abort (failsafe).")
    if countdown > 0:
        time.sleep(countdown)

    clicks_per_cell = max(1, int(clicks_per_cell))
    for i, (x, y) in enumerate(points, start=1):
        xi, yi = int(round(x)), int(round(y))
        print(f"[INFO] Clicking Q{i} at screen ({xi}, {yi}) x{clicks_per_cell}")
        pyautogui.moveTo(xi, yi, duration=0.0)
        for _ in range(clicks_per_cell):
            pyautogui.click(x=xi, y=yi)
            time.sleep(0.01)
        if delay > 0 and i < len(points):
            time.sleep(delay)


def main() -> None:
    set_dpi_awareness()
    args = parse_args()
    cv2.namedWindow("Screen", cv2.WINDOW_NORMAL)

    print("[INFO] Live preview started. Press S to solve+click, R to re-solve+click, C to re-click last solution, Q to quit.")

    with mss() as sct:
        last_result: Optional[SolveResult] = None

        while True:
            screen, monitor_left, monitor_top = capture_primary_monitor(sct)
            preview = screen.copy()

            if last_result is not None:
                cv2.polylines(preview, [last_result.board_quad.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(preview, "S: solve+click  R: re-run  C: re-click  Q: quit", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2, cv2.LINE_AA)
            cv2.imshow("Screen", preview)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                try:
                    if last_result is None or not last_result.screen_points:
                        print("[WARN] No solved screen points available to click.")
                    else:
                        click_screen_points(last_result.screen_points, args.click_countdown, args.click_delay, args.clicks_per_cell)
                except Exception as exc:
                    print(f"[ERROR] Clicking failed: {exc}")
            if key in (ord("s"), ord("S"), ord("r"), ord("R")):
                try:
                    result = run_pipeline(screen, args, monitor_left=monitor_left, monitor_top=monitor_top)
                    last_result = result

                    if result.placements:
                        annotated = annotate_solution(result.warped, result.placements, result.n)
                        cv2.imshow("Warped Board (Solution)", annotated)

                        warped_pts = queen_click_points_warped(result.placements, result.n, args.warp_size)
                        print("[INFO] Queen centers in warped space:")
                        for i, (x, y) in enumerate(warped_pts, start=1):
                            print(f"  Q{i}: ({x:.1f}, {y:.1f})")

                        if result.screen_points is not None:
                            print("[INFO] Approx queen centers in screen space:")
                            for i, (x, y) in enumerate(result.screen_points, start=1):
                                print(f"  Q{i}: ({x:.1f}, {y:.1f})")
                            click_screen_points(result.screen_points, args.click_countdown, args.click_delay, args.clicks_per_cell)
                    else:
                        msg = result.warped.copy()
                        cv2.putText(msg, "No solution", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
                        cv2.imshow("Warped Board (Solution)", msg)

                    if args.debug:
                        cv2.imshow("Debug (Optional)", make_debug_view(result))

                except Exception as exc:
                    print(f"[ERROR] {exc}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
