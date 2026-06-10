import numpy as np


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

BASKET_X = 89.25
BASKET_Y = 25.0

NUM_DEFENDERS = 5
NUM_ATTACKERS = 4
NUM_FRAMES = 6

# ------------------------------------------------------------
# Basic geometry helpers
# ------------------------------------------------------------

def euclidean_distance(x1, y1, x2, y2):
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


def distance_point_to_segment(px, py, ax, ay, bx, by):
    """
    Distance from point P to line segment A-B.
    Also returns projection fraction t:
        t = 0 at A
        t = 1 at B
    """
    abx = bx - ax
    aby = by - ay

    apx = px - ax
    apy = py - ay

    denom = abx ** 2 + aby ** 2

    if denom == 0:
        return euclidean_distance(px, py, ax, ay), 0.0

    t = (apx * abx + apy * aby) / denom
    t_clipped = np.clip(t, 0.0, 1.0)

    closest_x = ax + t_clipped * abx
    closest_y = ay + t_clipped * aby

    dist = euclidean_distance(px, py, closest_x, closest_y)

    return dist, float(t)


def point_projection_fraction(px, py, ax, ay, bx, by):
    """
    Projection fraction of point P onto infinite line A-B.
    Returns:
        0 at A
        1 at B
    """
    abx = bx - ax
    aby = by - ay

    denom = abx ** 2 + aby ** 2

    if denom == 0:
        return 0.0

    return float(((px - ax) * abx + (py - ay) * aby) / denom)


def perpendicular_distance_to_line(px, py, ax, ay, bx, by):
    """
    Perpendicular distance from point P to infinite line A-B.
    """
    abx = bx - ax
    aby = by - ay

    denom = np.sqrt(abx ** 2 + aby ** 2)

    if denom == 0:
        return euclidean_distance(px, py, ax, ay)

    return float(abs(abx * (ay - py) - (ax - px) * aby) / denom)


# ------------------------------------------------------------
# Extract absolute positions from canonical row
# ------------------------------------------------------------

def get_positions_from_row(row, frame=0):
    """
    Converts your canonical relative representation into absolute positions.

    Output:
        {
            "shooter": (x, y),
            "defender1": (x, y),
            ...
            "attacker1": (x, y),
            ...
        }
    """
    positions = {}

    sx = float(row[f"shooter_x_t{frame}"])
    sy = float(row[f"shooter_y_t{frame}"])

    positions["shooter"] = (sx, sy)

    for i in range(1, NUM_DEFENDERS + 1):
        dx_col = f"defender{i}_dx_t{frame}"
        dy_col = f"defender{i}_dy_t{frame}"

        if dx_col in row.index and dy_col in row.index:
            positions[f"defender{i}"] = (
                sx + float(row[dx_col]),
                sy + float(row[dy_col]),
            )

    for i in range(1, NUM_ATTACKERS + 1):
        dx_col = f"attacker{i}_dx_t{frame}"
        dy_col = f"attacker{i}_dy_t{frame}"

        if dx_col in row.index and dy_col in row.index:
            positions[f"attacker{i}"] = (
                sx + float(row[dx_col]),
                sy + float(row[dy_col]),
            )

    return positions


def get_defender_positions(row, frame=0):
    positions = get_positions_from_row(row)

    return [
        positions[f"defender{i}"]
        for i in range(1, NUM_DEFENDERS + 1)
        if f"defender{i}" in positions
    ]


def get_attacker_positions(row, frame=0):
    positions = get_positions_from_row(row)

    return [
        positions[f"attacker{i}"]
        for i in range(1, NUM_ATTACKERS + 1)
        if f"attacker{i}" in positions
    ]


# ------------------------------------------------------------
# Feature recomputation
# ------------------------------------------------------------

def compute_shot_angle(row, frame=0):

    sx = float(row[f"shooter_x_t{frame}"])
    sy = float(row[f"shooter_y_t{frame}"])

    dx = BASKET_X - sx
    dy = sy - BASKET_Y

    angle = np.degrees(np.arctan2(abs(dy), abs(dx)))

    return float(angle)


def compute_distance_to_basket(row, frame=0):
    sx = float(row[f"shooter_x_t{frame}"])
    sy = float(row[f"shooter_y_t{frame}"])

    return euclidean_distance(sx, sy, BASKET_X, BASKET_Y)


def compute_defender_distances(row, frame=0):
    positions = get_positions_from_row(row)

    sx, sy = positions["shooter"]

    distances = []

    for i in range(1, NUM_DEFENDERS + 1):
        key = f"defender{i}"

        if key not in positions:
            continue

        dx, dy = positions[key]
        distances.append(euclidean_distance(sx, sy, dx, dy))

    return np.array(distances, dtype=float)


def compute_nearest_defender_dist(row, frame=0):
    dists = compute_defender_distances(row)

    if len(dists) == 0:
        return np.nan

    return float(np.min(dists))


def compute_avg_defender_dist(row, frame=0):
    dists = compute_defender_distances(row)

    if len(dists) == 0:
        return np.nan

    return float(np.mean(dists))


def compute_defenders_within(row, radius_ft, frame=0):
    dists = compute_defender_distances(row)

    if len(dists) == 0:
        return 0

    return int(np.sum(dists <= radius_ft))


def compute_nearest_teammate_distance(row, frame=0):
    positions = get_positions_from_row(row)

    sx, sy = positions["shooter"]

    distances = []

    for i in range(1, NUM_ATTACKERS + 1):
        key = f"attacker{i}"

        if key not in positions:
            continue

        ax, ay = positions[key]
        distances.append(euclidean_distance(sx, sy, ax, ay))

    if len(distances) == 0:
        return np.nan

    return float(np.min(distances))


def compute_defenders_between(row, frame=0, basket_x=BASKET_X, basket_y=BASKET_Y):
    """
    Counts defenders between shooter and basket.
    """
    positions = get_positions_from_row(row)

    sx, sy = positions["shooter"]

    count = 0

    for i in range(1, NUM_DEFENDERS + 1):
        key = f"defender{i}"

        if key not in positions:
            continue

        dx, dy = positions[key]

        proj = point_projection_fraction(
            dx, dy,
            sx, sy,
            basket_x, basket_y
        )

        perp_dist = perpendicular_distance_to_line(
            dx, dy,
            sx, sy,
            basket_x, basket_y
        )

        if 0.0 < proj < 1.0 and perp_dist < 3.0:
            count += 1

    return int(count)


def compute_has_screen(row, frame=0):
    """
    Detects whether an offensive teammate is between the shooter
    and the nearest defender, close enough to plausibly screen.
    """
    positions = get_positions_from_row(row)

    sx, sy = positions["shooter"]

    defender_dists = {}

    for i in range(1, NUM_DEFENDERS + 1):
        key = f"defender{i}"

        if key not in positions:
            continue

        dx, dy = positions[key]
        defender_dists[key] = euclidean_distance(sx, sy, dx, dy)

    if len(defender_dists) == 0:
        return 0

    nearest_defender_key = min(defender_dists, key=defender_dists.get)
    ndx, ndy = positions[nearest_defender_key]

    for i in range(1, NUM_ATTACKERS + 1):
        key = f"attacker{i}"

        if key not in positions:
            continue

        ax, ay = positions[key]

        proj = point_projection_fraction(
            ax, ay,
            sx, sy,
            ndx, ndy
        )

        perp_dist = perpendicular_distance_to_line(
            ax, ay,
            sx, sy,
            ndx, ndy
        )

        dist_to_def = euclidean_distance(ax, ay, ndx, ndy)

        if 0.0 < proj < 1.0 and perp_dist < 2.0 and dist_to_def < 5.0:
            return 1

    return 0


def compute_teammate_between_defender(row, frame=0):
    """
    Checks whether a teammate is between shooter and nearest defender.
    """
    positions = get_positions_from_row(row)

    sx, sy = positions["shooter"]

    defender_dists = {}

    for i in range(1, NUM_DEFENDERS + 1):
        key = f"defender{i}"

        if key not in positions:
            continue

        dx, dy = positions[key]
        defender_dists[key] = euclidean_distance(sx, sy, dx, dy)

    if len(defender_dists) == 0:
        return 0

    nearest_defender_key = min(defender_dists, key=defender_dists.get)
    ndx, ndy = positions[nearest_defender_key]

    for i in range(1, NUM_ATTACKERS + 1):
        key = f"attacker{i}"

        if key not in positions:
            continue

        ax, ay = positions[key]

        proj = point_projection_fraction(
            ax, ay,
            sx, sy,
            ndx, ndy
        )

        perp_dist = perpendicular_distance_to_line(
            ax, ay,
            sx, sy,
            ndx, ndy
        )

        if 0.0 < proj < 1.0 and perp_dist < 5.0:
            return 1

    return 0


def compute_players_in_paint(row, frame=0):
    """
    Counts all 10 players in the paint.
    """
    positions = get_positions_from_row(row)

    count = 0

    for _, (x, y) in positions.items():
        if (x > 75.0) and (x <= 94.0) and (y > 17.0) and (y < 33.0):
            count += 1

    return int(count)


def compute_shooter_speed(row, frame_start=5, frame_end=0,
                          frame_step_seconds=0.2):
    
    x0 = float(row[f"shooter_x_t{frame_start}"])
    y0 = float(row[f"shooter_y_t{frame_start}"])

    x1 = float(row[f"shooter_x_t{frame_end}"])
    y1 = float(row[f"shooter_y_t{frame_end}"])

    dt = abs(frame_start - frame_end) * frame_step_seconds

    if dt == 0:
        return 0.0

    return euclidean_distance(x0, y0, x1, y1) / dt


def compute_defender_closing_speed(row, frame_start=5, frame_end=0,
                                   frame_step_seconds=0.2):

    start_col = f"defender1_dist_t{frame_start}"
    end_col = f"defender1_dist_t{frame_end}"

    if start_col not in row.index or end_col not in row.index:
        return np.nan

    dist_start = float(row[start_col])
    dist_end = float(row[end_col])

    dt = abs(frame_start - frame_end) * frame_step_seconds

    if dt == 0:
        return 0.0

    return float((dist_start - dist_end) / dt)


# ------------------------------------------------------------
# Main function to update all additional features
# ------------------------------------------------------------

def recompute_additional_features(row, frame=0):
    """
    Recomputes all engineered features used by the model.
    """
    row = row.copy()

    row["shot_angle"] = compute_shot_angle(row)
    row["distance_to_basket_tracking"] = compute_distance_to_basket(row)

    # Defender distances
    row["nearest_defender_dist"] = compute_nearest_defender_dist(row)
    row["avg_defender_dist"] = compute_avg_defender_dist(row)
    row["defenders_within_3ft"] = compute_defenders_within(row, radius_ft=3.0)
    row["defenders_within_5ft"] = compute_defenders_within(row, radius_ft=5.0)
    row["defenders_within_7ft"] = compute_defenders_within(row, radius_ft=7.0)

    # Spatial interaction
    row["defenders_between"] = compute_defenders_between(row)
    row["has_screen"] = compute_has_screen(row)
    row["teammate_between_defender"] = compute_teammate_between_defender(row)
    row["players_in_paint"] = compute_players_in_paint(row)

    # Teammate spacing
    row["nearest_teammate_distance"] = compute_nearest_teammate_distance(row)

    # Movement features
    row["shooter_speed"] = compute_shooter_speed(row, frame_start=5, frame_end=0)

    row["defender_closing_speed"] = compute_defender_closing_speed(row, frame_start=5, frame_end=0)

    return row