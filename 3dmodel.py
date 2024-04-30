import cv2
import numpy as np
import pyvista as pv


def process_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    edges = cv2.Canny(gray_image, 50, 500)

    # corner Detection
    gray = np.float32(gray_image)
    dst = cv2.cornerHarris(gray, blockSize=4, ksize=5, k=0.04)

    dst = cv2.dilate(dst, None)
    corners = np.where(dst > 0.01 * dst.max())
    corners = list(zip(*corners[::-1]))
    return edges, corners


def is_on_ray(lx1, ly1, lx2, ly2, cx, cy):
    line_dir = np.array([lx2 - lx1, ly2 - ly1])
    to_corner = np.array([cx - lx1, cy - ly1])
    if np.cross(line_dir, to_corner) == 0:
        t = np.dot(to_corner, line_dir) / np.dot(line_dir, line_dir)
        return t >= 0
    else:
        return False


def refine_lines(lines, corners, max_distance=30):
    if lines is None:
        return lines

    refined_lines = []
    lines = [line[0] for line in lines]
    while lines:
        l = lines.pop(0)
        x1, y1, x2, y2 = l

        # Find nearest corners on the ray for both endpoints
        possible_starts = [(c[0], c[1]) for c in corners if is_on_ray(x1, y1, x2, y2, c[0], c[1])]
        possible_ends = [(c[0], c[1]) for c in corners if is_on_ray(x2, y2, x1, y1, c[0], c[1])]

        if possible_starts:
            start_point = min(possible_starts, key=lambda c: np.hypot(c[0] - x1, c[1] - y1))
            if np.hypot(start_point[0] - x1, start_point[1] - y1) < max_distance:
                x1, y1 = start_point

        if possible_ends:
            end_point = min(possible_ends, key=lambda c: np.hypot(c[0] - x2, c[1] - y2))
            if np.hypot(end_point[0] - x2, end_point[1] - y2) < max_distance:
                x2, y2 = end_point

        refined_lines.append([x1, y1, x2, y2])

    return np.array(refined_lines).reshape(-1, 1, 4)


def connect_lines(lines, max_distance=10, max_angle=1):
    if lines is None:
        return lines

    # Convert angles to radians for calculation
    max_angle = np.deg2rad(max_angle)

    # Create a list of lines from the array for easier manipulation
    lines = [line[0] for line in lines]

    def line_magnitude(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def distance_point_to_line(px, py, x1, y1, x2, y2):
        return np.abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / line_magnitude(x1, y1, x2, y2)

    merged_lines = []
    while lines:
        l = lines.pop(0)
        x1, y1, x2, y2 = l
        line_to_merge = []
        for idx, (x3, y3, x4, y4) in enumerate(lines):
            # Check if current lines are close enough and have a similar slope
            if distance_point_to_line(x3, y3, x1, y1, x2, y2) < max_distance or distance_point_to_line(x4, y4, x1, y1,x2,y2) < max_distance:
                # Check angle between lines
                angle1 = np.arctan2(y2 - y1, x2 - x1)
                angle2 = np.arctan2(y4 - y3, x4 - x3)
                if abs(angle1 - angle2) < max_angle:
                    line_to_merge.append(idx)

        # Merge lines
        for idx in sorted(line_to_merge, reverse=True):
            x3, y3, x4, y4 = lines.pop(idx)
            # Assuming we simply connect the endpoints of the closest segments
            all_x = [x1, x2, x3, x4]
            all_y = [y1, y2, y3, y4]
            # New line endpoints
            x1, y1 = min(all_x), min(all_y)
            x2, y2 = max(all_x), max(all_y)

        merged_lines.append([x1, y1, x2, y2])

    return np.array(merged_lines).reshape(-1, 1, 4)


def find_walls(edges):
    lines = cv2.HoughLinesP(edges, 0.25, np.pi / 180, threshold=10, minLineLength=1, maxLineGap=10)
    connected_lines = connect_lines(lines)
    return lines, connected_lines


def create_wall_mesh(line, wall_height=500):
    x1, y1, x2, y2 = line
    points = np.array([
        [x1, y1, 0], [x2, y2, 0],
        [x1, y1, wall_height], [x2, y2, wall_height]
    ])

    # Create a wall
    wall = pv.Line(points[0], points[1]).extrude([0, 0, wall_height])
    return wall


def generate_3d_model(lines):
    plotter = pv.Plotter()
    all_walls = pv.MultiBlock()

    for line in lines:
        wall_mesh = create_wall_mesh(line[0])
        all_walls.append(wall_mesh)

    # Merge all meshes
    combined_mesh = all_walls.combine()
    plotter.add_mesh(combined_mesh, color='white', show_edges=True)
    plotter.show()


def main(image_path):
    image = cv2.imread(image_path)
    image2 = np.copy(image)
    edges, corners = process_image(image)
    lines, connect = find_walls(edges)
    connect = refine_lines(lines, corners)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for line in connect:
        for x1, y1, x2, y2 in line:
            cv2.line(image2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("plan_lines.jpg", image)

    cv2.imwrite("plan_connected.jpg", image2)

    generate_3d_model(connect)


if __name__ == "__main__":
    main("./floorplan2.png")
