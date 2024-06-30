# from matplotlib import rcParams
import pandas as pd
from hexalattice.hexalattice import *
import re
from sklearn.cluster import KMeans
import networkx as nx
from tqdm import tqdm

#rcParams['font.family'] = 'monospace'

from graph_flattener import construct_graph


def find_nearest_point(points, x, y):
    distances = np.sum((points - np.array([x, y])) ** 2, axis=1)
    nearest_index = np.argmin(distances)
    return nearest_index


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


def replace_greek_abbreviation(s):
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


table = pd.read_csv(f'simbad/Stars_plx_20.csv')

plt.style.use('dark_background')
sc = None
label = []


def make_map(star_table, central_star=None, max_distance=None, ax=None, snap_to_hex=True, save=None, cluster_name=None,
             division_name=None, title=False):
    global sc, label

    if ax is None:
        fig, ax = plt.subplots(dpi=200)
        fig.set_size_inches(6, 6)
    else:
        ax.clear()

    coord_columns = ['X', 'Y', 'Z']
    center = star_table[coord_columns].mean()
    distances = np.sqrt(((star_table[coord_columns] - center) ** 2).sum(axis=1))
    if central_star is None:
        central_star = star_table.loc[distances.idxmin()]['MAIN_ID']

    star = star_table[star_table['MAIN_ID'] == central_star]

    star_table['X'] -= float(star['X'].iloc[0])
    star_table['Y'] -= float(star['Y'].iloc[0])
    star_table['Z'] -= float(star['Z'].iloc[0])

    star_table['DIST_TO_CENTER'] = np.sqrt((star_table['X'] ** 2) + (star_table['Y'] ** 2) + (star_table['Z'] ** 2))

    if max_distance is not None:
        star_table = star_table[star_table['DIST_TO_CENTER'] < max_distance]
    else:
        max_distance = distances.max()

    star_table = star_table.sort_values(by=['DIST_TO_CENTER'], ascending=True)

    star_points = np.array([star_table['X'], star_table['Y'], star_table['Z']]).T
    names = list(star_table['MAIN_ID'])
    if division_name is not None:
        clusters = list(star_table[division_name])
    colours = []

    if cluster_name is not None and division_name is not None:
        if title:
            ax.set_title(f'{replace_greek_abbreviation(cluster_name)} {division_name}')
        for i in range(len(clusters)):
            if clusters[i] == cluster_name:
                colours.append('white')
            else:
                txt = replace_greek_abbreviation(clusters[i])
                names[i] = f'{names[i]} ({txt})'
                colours.append('red')
    else:
        for i in range(len(clusters)):
            colours.append('white')

    nodes = construct_graph(names, star_points, n=None, find_map_positions=True)

    center = np.array([0, 0])
    for n in nodes:
        if n.name == central_star:
            center = n.map_position
    for n in nodes:
        n.map_position -= center

    hex_size = 1

    hex_centers, _ = create_hex_grid(nx=int(4*max_distance), ny=int(4*max_distance),
                                     min_diam=hex_size, do_plot=True, align_to_origin=True, edge_color=[1, 1, 1],
                                     h_ax=ax)

    ax.set_xticks([], [])
    ax.set_yticks([], [])

    xs, ys = [], []
    label = []
    stars_in_hex = np.zeros(hex_centers.shape[0], dtype=int)
    hex = np.zeros(len(nodes), dtype=int)
    number_in_hex = np.zeros(len(nodes), dtype=int)

    if snap_to_hex:

        for i in range(len(nodes)):
            hex_i = find_nearest_point(hex_centers, nodes[i].map_position[0], nodes[i].map_position[1])
            hex[i] = hex_i
            number_in_hex[i] = stars_in_hex[hex_i]
            stars_in_hex[hex_i] += 1

        for i in range(len(nodes)):

            l = 0.4 * hex_size
            y_displacement = 0
            if stars_in_hex[hex[i]] > 1:
                y_displacement = ((number_in_hex[i] / (stars_in_hex[hex[i]] - 1)) * -l) + (l/2)

            xs.append(hex_centers[hex[i], 0])
            ys.append(hex_centers[hex[i], 1] + y_displacement)
            label.append(nodes[i].name)

    else:
        for i in range(len(nodes)):
            xs.append(nodes[i].map_position[0])
            ys.append(nodes[i].map_position[1])
            label.append(nodes[i].name)

    sc = ax.scatter(xs, ys, c=colours, s=1)
    for i in range(len(xs)):
        txt = replace_greek_abbreviation(label[i])
        txt = 'Prox Cen' if txt == 'Proxima Centauri' else txt
        # ax.text(xs[i], ys[i], f' {txt}', color='white', verticalalignment='center', fontsize='xx-small')
        ax.text(xs[i], ys[i], txt, color='white', fontsize='xx-small', rotation=30)

    padding = 1
    x_size, y_size = max(xs) - min(xs) + 2 * padding, max(ys) - min(ys) + 2 * padding
    x_pad = 0.5 * (y_size - x_size) if y_size > x_size else 0
    y_pad = 0.5 * (x_size - y_size) if x_size > y_size else 0

    ax.set_xlim([min(xs) - padding - x_pad, max(xs) + padding + x_pad])
    ax.set_ylim([min(ys) - padding - y_pad, max(ys) + padding + y_pad])

    if save is None:
        plt.show()
    else:
        plt.savefig(save, bbox_inches='tight')
        plt.close()

    display_coords = ax.transData.transform(np.array([xs, ys]).T)

    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bbox_display = bbox.transformed(fig.dpi_scale_trans)

    pixel_positions = display_coords - bbox_display.min
    pixel_positions = np.floor(pixel_positions).astype(int)

    return pixel_positions.tolist()


def interactive_map(central_star='Sol', max_distance=3):

    fig, ax = plt.subplots()

    def on_click(event):
        if event.inaxes:
            cont, ind = sc.contains(event)
            if cont:
                index = ind['ind'][0]
                make_map(table, ax=ax, central_star=label[index], max_distance=max_distance)

    cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)

    make_map(table, ax=ax, central_star=central_star, max_distance=max_distance)


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


def make_sectors(cluster_size=8, cutoff_distance=10, make_web_version=False):
    global table

    table = table[table['DIST'] < cutoff_distance]
    table.reset_index(drop=True, inplace=True)

    def cluster(star_table, division_name):

        star_table[division_name] = '-'
        star_table[f'{division_name}_ID'] = -1
        star_points = np.array([star_table['X'], star_table['Y'], star_table['Z']]).T

        n = int(len(star_table) / cluster_size)

        kmeans = KMeans(n_clusters=n)
        kmeans.fit(star_points)

        for index, row in star_table.iterrows():
            star_table.loc[index, f'{division_name}_ID'] = kmeans.labels_[index]

        cluster_names, cluster_centers, cluster_sizes = [], [], []
        clusters_list = []

        for cluster_id in range(n):
            cluster_table = star_table[star_table[f'{division_name}_ID'] == cluster_id]
            i_main = cluster_table['ABS_MAG_V'].idxmin()
            if np.isnan(i_main):
                i_main = cluster_table['PLX_VALUE'].idxmax()
            cluster_name = cluster_table.loc[i_main, 'MAIN_ID']
            for index, row in cluster_table.iterrows():
                star_table.loc[index, division_name] = cluster_name

            clusters_list.append(dict(star_table.loc[index]))

            coord_columns = ['X', 'Y', 'Z']
            cluster_center = cluster_table[coord_columns].mean()
            distances = np.sqrt(((cluster_table[coord_columns] - cluster_center) ** 2).sum(axis=1))

            cluster_sizes.append(distances.max())
            cluster_names.append(cluster_name)
            cluster_centers.append(cluster_center)

        clusters_table = pd.DataFrame(clusters_list)

        return clusters_table, cluster_names, cluster_centers, cluster_sizes

    def make_cluster_map(star_table, division_name, cluster_names, cluster_centers, cluster_sizes):

        cluster_links, cluster_pixels, cluster_cluster_labels = [], [], []

        for cluster_id in tqdm(range(len(cluster_names))):

            coord_columns = ['X', 'Y', 'Z']
            distances = np.sqrt(((star_table[coord_columns] - cluster_centers[cluster_id]) ** 2).sum(axis=1))

            cluster_name = cluster_names[cluster_id]
            max_distance = cluster_sizes[cluster_id] * 1.2

            for i in range(100):

                map_table = pd.DataFrame(star_table[distances < max_distance])
                sectors = set(list(map_table[division_name]))

                too_small, too_big = (len(sectors) < 4), (len(map_table) > 20)

                max_distance *= 1.1 if too_small else 1
                max_distance /= 1.1 if too_big else 1

                if not (too_big or too_small):
                    break

            cluster_links.append(sectors)
            cluster_cluster_labels.append(list(map_table[division_name]))

            if len(map_table) > 0:
                pixels = make_map(map_table, save=f'atlas/images/{division_name} {cluster_name}.png', cluster_name=cluster_name,
                         division_name=division_name)
            else:
                pixels = []

            cluster_pixels.append(pixels)

        G = nx.Graph()

        for s in cluster_names:
            G.add_node(s)

        for i in range(len(cluster_names)):
            for s in cluster_links[i]:
                if s != cluster_names[i]:
                    G.add_edge(cluster_names[i], s)

        nx.draw(G, nx.spring_layout(G), with_labels=True, node_size=200, node_color='blue', font_size=7,
                edge_color='gray', font_color='white')
        plt.savefig(f'atlas/images/{division_name} LINKS.png', bbox_inches='tight')
        plt.close()

        return cluster_cluster_labels, cluster_pixels

    sectors_table, sector_names, sector_centers, sector_sizes = cluster(table, 'SECTOR')
    region_table, region_names, region_centers, region_sizes = cluster(sectors_table, 'REGION')
    #zone_table, zone_names, zone_centers, zone_sizes = cluster(region_table, 'ZONE')

    sector_sector_labels, sector_pixels = make_cluster_map(table, 'SECTOR', sector_names, sector_centers, sector_sizes)
    make_cluster_map(sectors_table, 'REGION', region_names, region_centers, region_sizes)
    #make_cluster_map(region_table, 'ZONE', zone_names, zone_centers, zone_sizes)

    print('DONE')

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(region_table['X'], region_table['Y'], region_table['Z'], c='blue')
    # for index, row in region_table.iterrows():
    #     ax.text(row['X'], row['Y'], row['Z'], row['MAIN_ID'])
    # plt.show()

    def make_html_page(division_name, cluster_name, pixels, cluster_labels):
        f = open(f'atlas/{division_name} {cluster_name}.html', 'w')
        f.write('<!DOCTYPE html>\n<html>\n<head>')
        f.write(f'<title>{division_name}: {cluster_name}</title>')
        f.write('</head>\n<body>')
        f.write('<img src="images/SECTOR Sirius.png" usemap="#sectors">')
        f.write('<img src="images/SECTOR Sirius.png" style="width:750px;height:750px;" usemap="#sectors">')
        f.write('<map name="sectors">')

        for i in range(len(pixels)):
            x, y, r = pixels[i][0], pixels[i][1], 8
            sector_link_name = cluster_labels[i]
            f.write(f'<area shape="circle" coords="{x}, {y}, {r}" href="{division_name} {sector_link_name}.html">')

        f.write('</map>')
        f.write('</body>\n</html>')
        f.close()

    if make_web_version:

        for i in range(len(sector_names)):
            make_html_page('SECTOR', sector_names[i], sector_pixels[i], sector_sector_labels[i])


make_sectors(cluster_size=7, cutoff_distance=6, make_web_version=False)
