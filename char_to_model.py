import xml.etree.ElementTree as ET
import cv2
import numpy as np
import math

font_xml = ET.parse('arial.xml') # 通过fontTools生成的xml文件
font_blk_xml = ET.parse('ariblk.xml') # 通过fontTools生成的xml文件

# 获取字体的贝塞尔曲线
def get_bezier_curve(font_xml: ET.ElementTree, char):
    root = font_xml.getroot()
    cmap = root.find("cmap")
    char_code = ord(char)
    # print(f"char_code: {char_code:#x}")
    glyph_names = cmap.findall(f".//map[@code=\"{char_code:#x}\"]")
    glyph_name = glyph_names[0].get('name')
    char_glyph = root.find("glyf").find(f"TTGlyph[@name=\"{glyph_name}\"]")
    char_contours = char_glyph.findall('contour')
    return char_glyph, char_contours

# 将一段贝塞尔曲线转换为多边形
def get_curve_segment_points(curve_segment):
    n = len(curve_segment) - 1
    bt = lambda t, i:  math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
    points = []
    t_list = np.linspace(0, 1, 3 * n - 2, endpoint=False)
    for t in t_list:
        point = np.array([0, 0], dtype=float)
        for i, p in enumerate(curve_segment):
            x = float(p.get('x'))
            y = float(p.get('y'))
            point += bt(t, i) * np.array([x, y])
        points.append(point)
    return points

# 将贝塞尔曲线转换为多边形
def bezier_curve_to_polygon(contour):
    polygon_point = []
    contour_points = contour.findall('pt')
    new_contour_points = [contour_points.pop(0)]
    for contour_point in contour_points:
        if contour_point.get('on') == '0':
            last_point = new_contour_points[-1]
            if last_point.get('on') == '0':
                new_point = ET.Element('pt')
                new_point.set('x', str((float(last_point.get('x')) + float(contour_point.get('x'))) / 2))
                new_point.set('y', str((float(last_point.get('y')) + float(contour_point.get('y'))) / 2))
                new_point.set('on', '1')
                new_contour_points.append(new_point)
        new_contour_points.append(contour_point)
    contour_points = new_contour_points
    start_point = contour_points.pop(0)
    last_point = start_point
    next_segment = [start_point]
    while len(contour_points) > 0:
        point = contour_points.pop(0)
        if point.get('on') == '0':
            next_segment.append(point)
        else:
            last_point = point
            next_segment.append(point)
            polygon_point.extend(get_curve_segment_points(next_segment))
            next_segment = [last_point]
    next_segment.append(start_point)
    polygon_point.extend(get_curve_segment_points(next_segment))
    return np.array(polygon_point, dtype=float)

# 将简单多边形转换为三角形
def polygon_to_triangles(source_polygon):
    polygon = source_polygon.copy()
    def point_is_ear(polygon, i):
        polygon_len = len(polygon)
        p1 = polygon[(i + polygon_len - 1) % polygon_len]
        p2 = polygon[i]
        p3 = polygon[(i + 1) % polygon_len]
        l1 = p2 - p1
        l2 = p3 - p2
        cross_product = np.cross([l1[0], l1[1], 0], [l2[0], l2[1], 0])[2]
        if cross_product < 0:
            for p in polygon:
                if cv2.pointPolygonTest(np.array([p1, p2, p3]).astype(np.float32), p, False) > 0:
                    return False
            return True
    
    indexs = np.arange(len(polygon))
    triangles = []
    while len(polygon) > 3:
        # 剔除直线的中点
        for i in range(len(polygon)):
            p1 = polygon[(i + len(polygon) - 1) % len(polygon)]
            p2 = polygon[i]
            p3 = polygon[(i + 1) % len(polygon)]
            l1 = p2 - p1
            l2 = p3 - p2
            if np.cross([l1[0], l1[1], 0], [l2[0], l2[1], 0])[2] == 0:
                polygon = np.delete(polygon, i, axis=0)
                indexs = np.delete(indexs, i, axis=0)
                break
        else:
            for i in range(len(polygon)):
                if point_is_ear(polygon, i):
                    triangles.append([indexs[(i + len(polygon) - 1) % len(polygon)], indexs[i], indexs[(i + 1) % len(polygon)]])
                    polygon = np.delete(polygon, i, axis=0)
                    indexs = np.delete(indexs, i, axis=0)
                    break
        
    triangles.append([indexs[0], indexs[1], indexs[2]])
    return triangles

# 将多边形转换为3d模型
def polygon_to_model(glyph, polygon, width):
    triangles = np.array(polygon_to_triangles(polygon))
    xmin = int(glyph.get('xMin'))
    ymin = int(glyph.get('yMin'))
    xmax = int(glyph.get('xMax'))
    ymax = int(glyph.get('yMax'))
    ylen = ymax - ymin
    xlen = xmax - xmin
    maxlen = max(xlen, ylen)
    polygon = polygon.copy()
    polygon[:, 0] = ((polygon[:, 0] - xmin + (maxlen - xlen) / 2) * 2 / maxlen) - 1
    polygon[:, 1] = ((polygon[:, 1] - ymin + (maxlen - ylen) / 2) * 2 / maxlen) - 1
    vertex_front = []
    vertex_back = []
    index = np.tile(triangles, (2, 1))
    index[len(triangles):] += len(polygon)
    index[len(triangles):] = np.flip(index[len(triangles):], axis=1)
    for p in polygon:
        vertex_front.append([p[0], p[1], width / 2])
        vertex_back.append([p[0], p[1], -width / 2])
    vertex = []
    vertex.extend(vertex_front)
    vertex.extend(vertex_back)
    vertex = np.array(vertex, dtype=np.float64)
    i = 0
    z_index = []
    while i < len(vertex_front):
        l = len(vertex_front)
        i_next = (i + 1) % l
        z_index.append([i, i + l, i_next + l])
        z_index.append([i, i_next + l , i_next])
        i += 1
    z_index = np.array(z_index, dtype=int)
    index = np.concatenate((index, z_index), axis=0)
    return vertex, index

# 构建多边形的嵌套树
# 只考虑单层嵌套
def make_polygons_tree(polygons):
    tree = [(i, []) for i in range(len(polygons))]
    def tree_search(tree, value):
        if len(tree) == 0:
            return None
        for i in range(len(tree)):
            if tree[i][0] == value:
                return i, tree
            else:
                result = tree_search(tree[i][1], value)
                if result is not None:
                    return result
        return None
    
    def tree_move_to(tree, new_parent_value, node_value):
        result = tree_search(tree, node_value)
        if result is None:
            print("node not found")
            return
        index, parent = result
        node = parent.pop(index)
        new_parent = tree_search(tree, new_parent_value)
        if new_parent is None:
            print("new parent not found")
            return
        new_parent_index, new_parent_super = new_parent
        new_parent_super[new_parent_index][1].append(node)
        return
        
    for i in range(len(polygons)):
        for j in range(len(polygons)):
            if i == j:
                continue
            if cv2.pointPolygonTest(np.array(polygons[j]).astype(np.float32), polygons[i][0], False) > 0:
                tree_move_to(tree, j, i)
    return tree

# 将多边形的内外连接
# 只考虑单层嵌套
def connect_inside_to_out(inside, outside):
    def check_po(pi, p, po, outside):
        points_index = []
        triangle = np.array([pi, p, po]).astype(np.float32)
        for i in range(len(outside)):
            if cv2.pointPolygonTest(triangle, outside[i], False) > 0:
                points_index.append(i)
        return points_index
        
    new_polygon = []
    indexi = np.argmax(inside[:, 0])
    pi = inside[indexi]
    indexo = None
    for i in range(len(outside)):
        po1 = outside[i]
        po2 = outside[(i + 1) % len(outside)]
        if po2[1] == po1[1]:
            continue
        r = (pi[1] - po1[1]) / (po2[1] - po1[1])
        if r < 0 or r > 1:
            continue
        s = po1[0] + r * (po2[0] - po1[0]) - pi[0]
        if s < 0:
            continue
        p = po1 + r * (po2 - po1)
        indexo = i if po1[0] > po2[0] else (i + 1 ) % len(outside)
        po = outside[indexo]
        
        in_triangle_points = check_po(pi, p, po, outside)
        while len(in_triangle_points) > 0:
            indexo = in_triangle_points.pop(0)
            po = outside[indexo]
            in_triangle_points = check_po(pi, p, po, outside)
    new_polygon.extend(outside[:indexo + 1])
    new_polygon.extend(inside[indexi:])
    new_polygon.extend(inside[:indexi + 1])
    new_polygon.extend(outside[indexo:])
    new_polygon = np.array(new_polygon, dtype=float)
    return new_polygon

# 将多个多边形转换为3d模型
def polygons_to_model(glyph, polygons, width):
    new_polygons = []
    tree = make_polygons_tree(polygons)
    for i in range(len(tree)):
        if len(tree[i][1]) == 0:
            new_polygons.append(polygons[tree[i][0]])
        else:
            for j in range(len(tree[i][1])):
                inside = polygons[tree[i][1][j][0]]
                outside = polygons[tree[i][0]]
                polygon = connect_inside_to_out(inside, outside)
                new_polygons.append(polygon)
    model_vertex = []
    model_index = []
    for i in range(len(new_polygons)):
        vertex, index = polygon_to_model(glyph, new_polygons[i], width)
        model_vertex.extend(vertex)
        model_index.extend(index + len(model_vertex) - len(vertex))
    model_vertex = np.array(model_vertex, dtype=float)
    model_index = np.array(model_index, dtype=int)
    return model_vertex, model_index

def char_to_model(char, width, is_blk):
    font = font_blk_xml if is_blk else font_xml
    glyph, contours = get_bezier_curve(font, char)
    polygons = []
    for contour in contours:
        polygons.append(bezier_curve_to_polygon(contour))
    model_vertex, model_index = polygons_to_model(glyph, polygons, width)
    return model_vertex, model_index

# debug用
import time
def draw_polygon(source_polygon, xmin, ymin, xmax, ymax, sleep_time=0.5):
    polygon = source_polygon.astype(np.int32)
    polygon = polygon.copy()
    ylen = ymax - ymin
    xlen = xmax - xmin
    maxlen = max(xlen, ylen)
    img = np.zeros((maxlen, maxlen, 3), dtype=np.uint8)
    polygon[:, 0] = (polygon[:, 0] - xmin + (maxlen - xlen) / 2)
    polygon[:, 1] = (polygon[:, 1] - ymin + (maxlen - ylen) / 2)
    for i in range(len(polygon)):
        pt1 = polygon[i].astype(np.int32)
        pt2 = polygon[(i + 1) % len(polygon)].astype(np.int32)
        cv2.line(img, pt1, pt2, (255, 0, 0), 2)
        cv2.imwrite('polygon.png', cv2.flip(img, 0))
        time.sleep(sleep_time)

def draw_triangles(source_polygon, triangles, xmin, ymin, xmax, ymax, sleep_time=0.5):
    polygon = source_polygon.astype(np.int32)
    polygon = polygon.copy()
    ylen = ymax - ymin
    xlen = xmax - xmin
    maxlen = max(xlen, ylen)
    img = np.zeros((maxlen, maxlen, 3), dtype=np.uint8)
    polygon[:, 0] = (polygon[:, 0] - xmin + (maxlen - xlen) / 2)
    polygon[:, 1] = (polygon[:, 1] - ymin + (maxlen - ylen) / 2)
    for i in range(len(triangles)):
        pt1 = polygon[triangles[i][0]].astype(np.int32)
        pt2 = polygon[triangles[i][1]].astype(np.int32)
        pt3 = polygon[triangles[i][2]].astype(np.int32)
        cv2.line(img, pt1, pt2, (255, 0, 0), 2)
        cv2.imwrite('triangles.png', cv2.flip(img, 0))
        time.sleep(sleep_time)
        cv2.line(img, pt2, pt3, (0, 255, 0), 2)
        cv2.imwrite('triangles.png', cv2.flip(img, 0))
        time.sleep(sleep_time)
        cv2.line(img, pt3, pt1, (0, 0, 255), 2)
        cv2.imwrite('triangles.png', cv2.flip(img, 0))
        time.sleep(sleep_time)
    