# from matplotlib import rcParams
import numpy as np
import pandas as pd
from hexalattice.hexalattice import *
import re
from sklearn.cluster import KMeans
import networkx as nx
from tqdm import tqdm

# rcParams['font.family'] = 'monospace'

from graph_flattener import construct_graph
from hexkit_interface import make_hex_map


class cluster:

    def __init__(self, cluster_id, cluster_table, cluster_type):

        self.id = cluster_id
        self.table = cluster_table
        self.type = cluster_type

        i_main = self.table['ABS_MAG_V'].idxmin()
        if np.isnan(i_main):
            i_main = self.table['PLX_VALUE'].idxmax()
        self.name = self.table.loc[i_main, 'MAIN_ID']

        coord_columns = ['X', 'Y', 'Z']
        self.center = self.table[coord_columns].mean()
        distances = np.sqrt(((cluster_table[coord_columns] - self.center) ** 2).sum(axis=1))
        self.size = distances.max()

        self.links = []

    def make_map_table(self, star_table):

        coord_columns = ['X', 'Y', 'Z']
        dist = np.sqrt(((star_table[coord_columns] - self.center) ** 2).sum(axis=1))

        max_distance = self.size * 1.2
        links, map_table = [], None

        for i in range(100):

            map_table = pd.DataFrame(star_table[dist < max_distance])
            links = set(list(map_table[f'{self.type}_ID']))

            too_small, too_big = (len(links) < 4), (len(map_table) > 20)

            max_distance *= 1.1 if too_small else 1
            max_distance /= 1.1 if too_big else 1

            if not (too_big or too_small):
                break

        self.links = links

        return map_table


def find_nearest_point(points, p):
    distances = np.sum((points - p) ** 2, axis=1)
    nearest_index = np.argmin(distances)
    return nearest_index


def replace_greek_abbreviation(s):

    greek_letters = {
        'alf': r'alpha',
        'bet': r'beta',
        'gam': r'gamma',
        'del': r'delta',
        'eps': r'epsilon',
        'zet': r'zeta',
        'eta': r'eta',
        'tet': r'theta',
        'iot': r'iota',
        'kap': r'kappa',
        'lam': r'lambda',
        'mu.': r'mu',
        'nu.': r'nu',
        'ksi': r'xi',
        'omi': r'o',
        'pi': r'pi',
        'pi.': r'pi',
        'rho': r'rho',
        'sig': r'sigma',
        'tau': r'tau',
        'ups': r'upsilon',
        'phi': r'phi',
        'chi': r'chi',
        'psi': r'psi',
        'w': r'omega'
    }

    match = re.match(r'^([A-Za-z]+)(\d+)?', s)
    if match:
        abbrev = match.group(1)
        number = match.group(2)
        if abbrev in greek_letters:
            if number:
                return re.sub(r'^' + abbrev + number, r'$\\' + greek_letters[abbrev] + f'^{{{int(number)}}}$', s)
            else:
                return re.sub(r'^' + abbrev, r'$\\' + greek_letters[abbrev] + '$', s)
    return s


def first_capital_letter(s):
    match = re.search(r'[A-Z]', s)
    if match:
        return match.group(0)
    return 'M'


def hex_grid_dimensions(xs, ys, hex_size):
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    return int(((max_x - min_x) * 1.5) / hex_size), int(((max_y - min_y) * 1.5) / hex_size)


table = pd.read_csv(f'simbad/Stars_plx_20.csv')

plt.style.use('dark_background')
sc = None
label = []
dpi = 200


def generate_map_data(star_table, hex_size=1, division=None, snap_to_hex=True, central_star=None, cluster_name=None):

    coords = ['X', 'Y', 'Z']
    center = star_table[coords].mean()
    distances = np.sqrt(((star_table[coords] - center) ** 2).sum(axis=1))
    if central_star is None:
        central_star = star_table.loc[distances.idxmin()]['MAIN_ID']

    star = star_table[star_table['MAIN_ID'] == central_star]
    star_table['X'] -= float(star['X'].iloc[0])
    star_table['Y'] -= float(star['Y'].iloc[0])
    star_table['Z'] -= float(star['Z'].iloc[0])

    star_table['DIST_TO_CENTER'] = np.sqrt((star_table['X'] ** 2) + (star_table['Y'] ** 2) + (star_table['Z'] ** 2))
    star_table = star_table.sort_values(by=['DIST_TO_CENTER'], ascending=True)
    star_points = np.array([star_table['X'], star_table['Y'], star_table['Z']]).T

    nodes = construct_graph(list(star_table['MAIN_ID']), star_points, n=None, find_map_positions=True)
    x_map_pos, y_map_pos = [], []

    for n in nodes:
        x_map_pos.append(n.map_position[0])
        y_map_pos.append(n.map_position[1])

    star_table['X_MAP'], star_table['Y_MAP'], star_table['COLOUR'] = x_map_pos, y_map_pos, 'M'

    if cluster_name is not None:
        for i, row in star_table.iterrows():
            star_table.loc[i, 'LABEL'] = row['MAIN_ID'] +\
                                         (f" ({row[division]})" if row[division] != cluster_name else '')
    else:
        star_table['LABEL'] = star_table['MAIN_ID']

    for i, row in star_table.iterrows():
        star_table.loc[i, 'COLOUR'] = first_capital_letter(list(row['SP_TYPE'])[0])

    if snap_to_hex:

        star_table['HEX'], star_table['NUM_IN_HEX'] = 0, 0
        hex_x, hex_y = hex_grid_dimensions(star_table['X_MAP'], star_table['Y_MAP'], hex_size)

        hex_centers, _ = create_hex_grid(nx=hex_x, ny=hex_y, min_diam=hex_size, do_plot=False, align_to_origin=True)
        hex_centers = hex_centers[::-1]
        stars_in_hex = np.zeros(len(hex_centers))

        for i, row in star_table.iterrows():
            hx = find_nearest_point(hex_centers, np.array([row['X_MAP'], row['Y_MAP']]))
            star_table.loc[i, 'HEX'] = hx
            stars_in_hex[hx] += 1
            star_table.loc[i, 'NUM_IN_HEX'] = stars_in_hex[hx]

        for i, row in star_table.iterrows():
            hx = row['HEX']
            l, n, total = 0.4 * hex_size, row['NUM_IN_HEX'], stars_in_hex[hx]
            y_disp = ((n / (total - 1)) * -l) + (l / 2) if total > 1 else 0
            star_table.loc[i, 'X_MAP'] = hex_centers[hx, 0]
            star_table.loc[i, 'Y_MAP'] = hex_centers[hx, 1] + y_disp

        return star_table, hex_x, hex_y

    return star_table


def make_matplotlib_map(star_table, title=None, save=None, hex_size=1, ax=None, division=None, return_pixels=False,
                        hex_x=0, hex_y=0):
    global sc, label

    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)
        fig.set_size_inches(6, 6)
    else:
        ax.clear()

    ax.set_xticks([], [])
    ax.set_yticks([], [])

    if hex_size is not None and hex_x > 0 and hex_y > 0:
        hex_centers, _ = create_hex_grid(nx=hex_x, ny=hex_y, min_diam=hex_size, do_plot=True, align_to_origin=True,
                                         edge_color=[1, 1, 1], h_ax=ax)

    if title is not None:
        ax.set_title(title)

    sc = ax.scatter(star_table['X_MAP'], star_table['Y_MAP'], c='white', s=1)
    for i, row in star_table.iterrows():
        ax.text(row['X_MAP'], row['Y_MAP'], row['LABEL'],
                color='white', fontsize='xx-small', rotation=30)

    padding = 1
    x_size = star_table['X_MAP'].max() - star_table['X_MAP'].min() + 2 * padding
    y_size = star_table['Y_MAP'].max() - star_table['Y_MAP'].min() + 2 * padding
    x_pad = 0.5 * (y_size - x_size) if y_size > x_size else 0
    y_pad = 0.5 * (x_size - y_size) if x_size > y_size else 0

    ax.set_xlim([star_table['X_MAP'].min() - padding - x_pad, star_table['X_MAP'].max() + padding + x_pad])
    ax.set_ylim([star_table['Y_MAP'].min() - padding - y_pad, star_table['Y_MAP'].max() + padding + y_pad])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

    if save is None:
        plt.show()
    else:
        plt.savefig(save, bbox_inches='tight')

    if return_pixels:
        bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).transformed(fig.dpi_scale_trans)
        pixel_pos = np.floor(ax.transData.transform(np.array([star_table['X_MAP'], star_table['Y_MAP']]).T)).astype(int)
        pixel_pos[:, 1] = int(bbox.height) - pixel_pos[:, 1]
        pixel_pos += np.array([dpi * 0.1, dpi * 0.1], dtype=int) + np.array([8, -2])
        plt.close()
        return list(star_table[division]), pixel_pos.tolist()

    plt.close()


def make_hexkit_map(star_table, save, hex_x, hex_y, hex_size=1):

    hex_map_data = []

    for i, row in star_table.iterrows():
        hex_map_data.append((row['LABEL'], row['HEX'], row['COLOUR']))

    make_hex_map(save, hex_x, hex_y, hex_map_data)


# def interactive_map(central_star='Sol', max_distance=3):
#     fig, ax = plt.subplots()
#
#     def on_click(event):
#         if event.inaxes:
#             cont, ind = sc.contains(event)
#             if cont:
#                 index = ind['ind'][0]
#                 generate_map_data(table, central_star=label[index], max_distance=max_distance)
#                 make_map(table, ax=ax, central_star=label[index], max_distance=max_distance)
#
#     cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)
#
#     make_map(table, ax=ax, central_star=central_star, max_distance=max_distance)


def get_nearest_stars(star_name):
    star_table = pd.DataFrame(table)

    star = star_table[star_table['MAIN_ID'] == star_name]

    star_table['X2'] = star_table['X'] - float(star['X'].iloc[0])
    star_table['Y2'] = star_table['Y'] - float(star['Y'].iloc[0])
    star_table['Z2'] = star_table['Z'] - float(star['Z'].iloc[0])

    star_table['DIST_TO_CENTER'] = np.sqrt((star_table['X2'] ** 2) + (star_table['Y2'] ** 2) + (star_table['Z2'] ** 2))
    star_table = star_table.sort_values(by=['DIST_TO_CENTER'], ascending=True)
    # star_table.reset_index(drop=True, inplace=True)

    count, max = 0, 40
    for index, row in star_table.iterrows():
        name = row['MAIN_ID']
        RA = row['RA']
        DEC = row['DEC']
        x, y, z = row['X'], row['Y'], row['Z']
        DIST = row['DIST_TO_CENTER']
        print(f'{name}: ({x}, {y}, {z})')
        count += 1
        if count > max:
            break


def make_sectors(cluster_size=8, cutoff_distance=10, make_hexkit=False):
    global table

    table = table[table['DIST'] < cutoff_distance]
    table.reset_index(drop=True, inplace=True)

    def clustering(star_table, division_name):

        star_table[division_name] = '-'
        star_table[f'{division_name}_ID'] = -1
        star_points = np.array([star_table['X'], star_table['Y'], star_table['Z']]).T

        n = int(len(star_table) / cluster_size)

        kmeans = KMeans(n_clusters=n)
        kmeans.fit(star_points)

        for index, row in star_table.iterrows():
            star_table.loc[index, f'{division_name}_ID'] = kmeans.labels_[index]

        cluster_list, cluster_names, star_table[division_name]  = [], [], ''

        for cluster_id in range(n):
            c = cluster(cluster_id, star_table[star_table[f'{division_name}_ID'] == cluster_id], division_name)
            cluster_list.append(c)
            cluster_names.append(c.name)

        for index, row in star_table.iterrows():
            star_table.loc[index, division_name] = cluster_names[star_table.loc[index, f'{division_name}_ID']]

        return cluster_list

    def make_cluster_map(cluster_list, star_table):

        for c in tqdm(cluster_list):

            map_table = c.make_map_table(star_table)

            if len(map_table) > 0:
                map_table, hex_x, hex_y = generate_map_data(map_table, division=c.type, cluster_name=c.name)
                if make_hexkit:
                    make_hexkit_map(map_table, f'atlas/hexkit/{c.type} {c.name}.map', hex_x, hex_y)
                make_matplotlib_map(map_table, save=f'atlas/hexkit/{c.type} {c.name}.png', hex_x=hex_x, hex_y=hex_y)

    def make_cluster_graph(cluster_list):

        G = nx.Graph()
        cluster_names = []

        for c in cluster_list:
            cluster_names.append(c.name)
            G.add_node(c.name)

        for c in cluster_list:
            for i_l in c.links:
                if i_l != c.id:
                    G.add_edge(c.name, cluster_names[i_l])

        nx.draw(G, nx.spring_layout(G), with_labels=True, node_size=200, node_color='blue', font_size=7,
                edge_color='gray', font_color='white')
        plt.savefig(f'atlas/images/{cluster_list[0].type} LINKS.png', bbox_inches='tight')
        plt.close()

    sector_list = clustering(table, 'SECTOR')
    make_cluster_map(sector_list, table)
    make_cluster_graph(sector_list)

    # region_table, region_names, region_centers, region_sizes = cluster(sectors_table, 'REGION')
    # zone_table, zone_names, zone_centers, zone_sizes = cluster(region_table, 'ZONE')

    # region_region, region_pixels = make_cluster_map(sectors_table, 'REGION', region_names, region_centers, region_sizes)
    # zone_zone, zone_pixels = make_cluster_map(region_table, 'ZONE', zone_names, zone_centers, zone_sizes)

    print('DONE')


make_sectors(cluster_size=5, cutoff_distance=5, make_hexkit=True)
