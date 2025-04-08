import numpy as np
import cv2
from char_to_model import char_to_model

char_list = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
]

viewport = np.array([0, 0, 128, 128], dtype=np.int32)

light_direction = np.array([0.2, -0.6, -0.5], dtype=np.float64)
light_direction /= np.linalg.norm(light_direction)

canvas = np.zeros((viewport[3], viewport[2]), dtype=np.uint8)
canvas.fill(230)
canvas_depth = -np.ones((viewport[3], viewport[2]), dtype=np.float64)

def check_with_interpolation(p1, p2, p3, x, y):
    # Barycentric coordinates
    denom = (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1])
    if denom == 0:
        return False, None
    a = ((p2[1] - p3[1]) * (x - p3[0]) + (p3[0] - p2[0]) * (y - p3[1])) / denom
    b = ((p3[1] - p1[1]) * (x - p3[0]) + (p1[0] - p3[0]) * (y - p3[1])) / denom
    c = 1.0 - a - b
    if 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1:
        return True, a * p1 + b * p2 + c * p3
    return False, None

def render_char(char, is_blk, model_matrix, view_matrix, projection_matrix):
    vertex, index = char_to_model(char, 0.5, is_blk)
    normal_matrix = np.linalg.inv(view_matrix @ model_matrix).T[:3, :3]
    light_dir = normal_matrix @ light_direction
    light_dir /= np.linalg.norm(light_dir)
    def load_point(i):
        p1 = vertex[index[i][0]]
        p2 = vertex[index[i][1]]
        p3 = vertex[index[i][2]]
        p1 = np.array([p1[0], p1[1], p1[2], 1])
        p2 = np.array([p2[0], p2[1], p2[2], 1])
        p3 = np.array([p3[0], p3[1], p3[2], 1])
        return p1, p2, p3
    
    def get_normal(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1
        normal = -np.cross(v1[:3], v2[:3])
        if np.linalg.norm(normal) == 0:
            return None
        return normal / np.linalg.norm(normal)
    
    def sample_with_shading(view_space_pos, ndc_space_pos, normal, ndc_x, ndc_y, last_depth):
        result, p = check_with_interpolation(ndc_space_pos[0], ndc_space_pos[1], ndc_space_pos[2], ndc_x, ndc_y)
        if not result or p[2] < last_depth or p[2] > 1:
            return None
        brightness = max(np.dot(normal, -light_dir), 0) * 0.7 + 0.3
        return brightness, p[2]

    def render_triangle(p1, p2, p3):
        if np.cross((p2 - p1)[:3], (p3 - p1)[:3])[2] > 0:
            return
        triangle = np.array([p1, p2, p3])
        normal = get_normal(p1, p2, p3)
        if normal is None:
            return
        normal = normal_matrix @ normal
        normal /= np.linalg.norm(normal)
        view_space_pos = (view_matrix @ model_matrix @ triangle.T).T
        ndc_space_pos = (projection_matrix @ view_space_pos.T).T
        ndc_space_pos /= ndc_space_pos[:, 3][:, np.newaxis]
        xmin = np.min(ndc_space_pos[:, 0])
        xmax = np.max(ndc_space_pos[:, 0])
        ymin = np.min(ndc_space_pos[:, 1])
        ymax = np.max(ndc_space_pos[:, 1])
        def fast_check(x, y):
            return xmin <= x <= xmax and ymin <= y <= ymax
        for y in range(viewport[1], viewport[3]):
            smaple_y = y + 0.5
            for x in range(viewport[0], viewport[2]):
                smaple_x = x + 0.5
                ndc_x = (smaple_x - viewport[0]) / (viewport[2] - viewport[0]) * 2 - 1
                ndc_y = 1 - (smaple_y - viewport[1]) / (viewport[3] - viewport[1]) * 2
                if not fast_check(ndc_x, ndc_y):
                    continue
                result = sample_with_shading(view_space_pos, ndc_space_pos, normal, ndc_x, ndc_y, canvas_depth[y, x])
                if result:
                    canvas[y, x] = int(result[0] * 255)
                    canvas_depth[y, x] = result[1]
    
    for i in range(len(index)):
        p1, p2, p3 = load_point(i)
        render_triangle(p1, p2, p3)

pitch = np.radians(25)
forward_rotation = np.array([
    [1, 0, 0, 0],
    [0, np.cos(pitch), -np.sin(pitch), 0],
    [0, np.sin(pitch), np.cos(pitch), 0],
    [0, 0, 0, 1]
], dtype=np.float64)
sideway_yaw = -np.radians(40)
sideway_rotation = np.array([
    [1, 0, 0, 0],
    [0, np.cos(pitch), -np.sin(pitch), 0],
    [0, np.sin(pitch), np.cos(pitch), 0],
    [0, 0, 0, 1]
], dtype=np.float64) @ np.array([
    [np.cos(sideway_yaw), 0, np.sin(sideway_yaw), 0],
    [0, 1, 0, 0],
    [-np.sin(sideway_yaw), 0, np.cos(sideway_yaw), 0],
    [0, 0, 0, 1]
], dtype=np.float64)
scale = np.eye(4, dtype=np.float64) * 0.85
scale[3, 3] = 1
view_matrix = np.eye(4, dtype=np.float64)
projection_matrix = np.eye(4, dtype=np.float64)

def make_template(char, is_forward, is_blk):
    global canvas, canvas_depth
    canvas.fill(230)
    canvas_depth.fill(-1)
    model_matrix = None
    if is_forward:
        model_matrix = forward_rotation @ scale
    else:
        model_matrix = sideway_rotation @ scale
    render_char(char, is_blk, model_matrix, view_matrix, projection_matrix)
    cv2.imwrite("output.png", canvas)
    return canvas.copy()