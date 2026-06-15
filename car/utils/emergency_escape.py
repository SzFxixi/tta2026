import math
from utils.config_loader import cfg


def danger_score(px, py, obstacles, zones):
    s = 0.0
    for o in obstacles:
        s += 5.0 / max(math.hypot(px - o['x'], py - o['y']), 0.01)
    if zones:
        for zxmin, zxmax, zymin, zymax in zones:
            dx = max(zxmin - px, 0, px - zxmax)
            dy = max(zymin - py, 0, py - zymax)
            s += 1.0 / max(math.hypot(dx, dy), 0.01)
    return s


def escape_danger_zone(cx, cy, obstacles, zones, cli,
                        max_rounds=None, min_safe_dist=None,
                        car_position_fn=None):
    if max_rounds is None:
        max_rounds = cfg.client.max_probe_rounds
    if min_safe_dist is None:
        min_safe_dist = cfg.path_planning.expansion_radius_3

    last_dir = None
    same_dir_count = 0

    for probe in range(max_rounds):
        # 检查是否已离开危险区
        min_obs = min((math.hypot(cx - o['x'], cy - o['y']) for o in obstacles), default=10.0)
        in_zone = any(z[0] <= cx <= z[1] and z[2] <= cy <= z[3] for z in zones) if zones else False
        if min_obs >= min_safe_dist and not in_zone:
            break

        # 四方向打分，选最安全
        best_dx, best_dy, best_score = 0, 0, danger_score(cx, cy, obstacles, zones)
        for dx, dy in [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]:
            s = danger_score(cx + dx, cy + dy, obstacles, zones)
            if s < best_score:
                best_dx, best_dy, best_score = dx, dy, s
        if best_dx == 0 and best_dy == 0:
            break

        # 同方向连续则加速
        cur_dir = (best_dx, best_dy)
        step = 0.1 * (same_dir_count + 1) if cur_dir == last_dir else 0.1
        same_dir_count = same_dir_count + 1 if cur_dir == last_dir else 0
        last_dir = cur_dir

        target_x = cx + best_dx * step / 0.1
        target_y = cy + best_dy * step / 0.1
        print(f"  [避险] 离开危险区{probe+1}: ({cx:.3f},{cy:.3f}) → ({target_x:.3f},{target_y:.3f})  obs={min_obs:.3f}m step={step:.1f}m")

        ok, resp = cli.move_relative(-(target_x - cx), -(target_y - cy))
        if not ok:
            print(f"  ✗ 探路失败: {resp.get('errorMessage', resp.get('error', '?'))}")
            break

        # 读实际位置
        if car_position_fn:
            actual_x, actual_y = car_position_fn()
            if actual_x is not None:
                cx, cy = actual_x, actual_y
        else:
            cx, cy = target_x, target_y

    return cx, cy
