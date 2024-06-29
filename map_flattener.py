import numpy as np
from numba import jit
from itertools import combinations
from scipy.optimize import minimize


@jit(nopython=True)
def distance_between_2_points(x1, x2):
    return np.sqrt(np.dot(x1 - x2, x1 - x2))


@jit(nopython=True)
def ratio_difference_3_points(a3, b3, c3, a2, b2, c2):
    AB3 = np.sqrt(np.dot(a3 - b3, a3 - b3))
    AC3 = np.sqrt(np.dot(a3 - c3, a3 - c3))
    AB2 = np.sqrt(np.dot(a2 - b2, a2 - b2))
    AC2 = np.sqrt(np.dot(a2 - c2, a2 - c2))

    ratio3 = AB3/AC3
    ratio2 = AB2/AC2

    return (ratio2 - ratio3) ** 2


def rotate_and_translate(point, center_of_rotation, theta, translation):

    point = np.array(point)
    center_of_rotation = np.array(center_of_rotation)
    translation = np.array(translation)

    translated_point = point - center_of_rotation

    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    rotated_point = rotation_matrix @ translated_point
    final_point = rotated_point + center_of_rotation + translation

    return final_point


def ratio_difference_n_point(x3, x2):
    total = 0
    count = 0
    indexes = np.arange(0, x3.shape[0])
    x2 = x2.reshape(-1, 2)

    for i in range(x3.shape[0]):
        ia = indexes[i]
        not_ia = np.concatenate([indexes[:i], indexes[i + 1:]], axis=0)

        for ibc in combinations(not_ia, 2):
            ib, ic = ibc
            count += 1
            total += ratio_difference_3_points(x3[ia], x3[ib], x3[ic], x2[ia], x2[ib], x2[ic])

    return total / count


def map_3d_to_2d_small(points):

    points = np.array(points)
    initial_guess = points[:, 1:]
    initial_guess += np.random.random(initial_guess.shape) * 0.001

    print(f'Flattening {points.shape[0]} points...')

    objective_function = lambda x2: ratio_difference_n_point(points, x2)
    result = minimize(objective_function, initial_guess.flatten(), method='Nelder-Mead')
    map_2d = result.x.reshape(-1, 2)
    print(f'Average error {result.fun:.1e}')

    i_test_1, i_test_2 = 0, 1

    dist_3d = distance_between_2_points(points[i_test_1], points[i_test_2])
    dist_2d = distance_between_2_points(map_2d[i_test_1], map_2d[i_test_2])

    map_2d = map_2d * (dist_3d / dist_2d)
    map_2d = map_2d - map_2d[0]

    print('Done')

    return map_2d


class octree_node:

    def __init__(self, x_bounds, y_bounds, z_bounds, parent=None):
        self.bounds = np.array([x_bounds, y_bounds, z_bounds])
        self.center = np.array([0.5 * (x_bounds[0] + x_bounds[1]),
                                0.5 * (y_bounds[0] + y_bounds[1]),
                                0.5 * (z_bounds[0] + z_bounds[1])])
        self.parent = parent
        self.children = np.empty((2,2,2), dtype=object)

        self.indexes = []
        self.points_3d, self.points_2d = {}, {}
        self.count = 0
        self.mapped = False

        if self.parent is None:
            self.level = 0
        else:
            self.level = self.parent.level + 1

    def make_2d_map_from_points(self):

        keys = list(self.points_3d.keys())

        if self.count > 2:
            points_2d = map_3d_to_2d_small(np.array(list(self.points_3d.values())))

            for i in range(len(self.points_3d)):
                self.points_2d[keys[i]] = points_2d[i]

        elif self.count == 2:

            d = distance_between_2_points(self.points_3d[keys[0]], self.points_3d[keys[1]])

            self.points_2d[keys[0]] = np.array([0, 0])
            self.points_2d[keys[1]] = np.array([d, 0])

        elif self.count == 1:
            self.points_2d[keys[0]] = np.array([0, 0])

        self.mapped = True

    def make_2d_map_from_children(self):

        initial_guess = np.zeros(24)

        def error_function(x, y, theta):

            x3, x2 = np.zeros((self.count, 3)), np.zeros((self.count, 2))
            n = 0

            for i_x in range(2):
                for i_y in range(2):
                    for i_z in range(2):

                        child = self.children[i_x, i_y, i_z]
                        j = 4 * i_z + 2 * i_y + i_x

                        for i in child.indexes:
                            x3[n, :] = child.points_3d[i]
                            x2[n, :] = rotate_and_translate(child.points_2d[i], (0, 0), theta[j], (x[j], y[j]))

                        n += 1

            return ratio_difference_n_point(x3, x2)

        objective_function = lambda x : error_function(x[0:8], x[8:16], x[16:24])
        result = minimize(objective_function, initial_guess, method='Nelder-Mead')

        x, y, theta = result.x[0:8], result.x[8:16], result.x[16:24]

        print(x)
        print(y)
        print(theta)

        self.mapped = True


def build_octree(points):

    tree = octree_node((np.min(points[:, 0]), np.max(points[:, 0])),
                       (np.min(points[:, 1]), np.max(points[:, 1])),
                       (np.min(points[:, 2]), np.max(points[:, 2])))

    tree.indexes = np.arange(points.shape[0])

    def recursive_build(node):
        node.count = len(node.indexes)

        for i in node.indexes:
            node.points_3d[i] = points[i, :]

        if node.count > 1:

            for i_x in range(2):
                for i_y in range(2):
                    for i_z in range(2):
                        x_min = node.center[0] if bool(i_x) else node.bounds[0, 0]
                        x_max = node.bounds[0, 1] if bool(i_x) else node.center[0]
                        y_min = node.center[1] if bool(i_y) else node.bounds[1, 0]
                        y_max = node.bounds[1, 1] if bool(i_y) else node.center[1]
                        z_min = node.center[2] if bool(i_z) else node.bounds[2, 0]
                        z_max = node.bounds[2, 1] if bool(i_z) else node.center[2]
                        node.children[i_x, i_y, i_z] = \
                            octree_node((x_min, x_max), (y_min, y_max), (z_min, z_max), parent=node)

            for i in node.indexes:
                i_x = int(points[i, 0] > node.center[0])
                i_y = int(points[i, 1] > node.center[1])
                i_z = int(points[i, 2] > node.center[2])

                node.children[i_x, i_y, i_z].indexes.append(i)

            for i_x in range(2):
                for i_y in range(2):
                    for i_z in range(2):
                        recursive_build(node.children[i_x, i_y, i_z])

        return node

    return recursive_build(tree)


def map_3d_to_2d_large(points, n=8):

    points = np.array(points)
    print(f'Flattening {points.shape[0]} points...')

    tree = build_octree(points)

    node_list = []

    def get_nodes(node):

        if node is not None:
            if node.count > 0:
                indent = '-' * node.level
                print(indent + f'{node.count}')

            if 8 >= node.count > 0:
                node_list.append(node)
            else:
                for i_x in range(2):
                    for i_y in range(2):
                        for i_z in range(2):
                            get_nodes(node.children[i_x, i_y, i_z])

    get_nodes(tree)

    for node in node_list:
        node.make_2d_map_from_points()

    for node in node_list:
        print(f'{node.level} : {node.indexes}')

    for node in node_list:
        if not node.parent.mapped and node.level == 2:
            node.parent.make_2d_map_from_children()
