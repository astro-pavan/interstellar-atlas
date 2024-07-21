from astroquery.simbad import Simbad
from astropy.table.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
from tqdm import tqdm
import numpy as np
import re

greek_letters = [
    'alf', 'bet', 'gam', 'del', 'eps', 'zet', 'eta', 'tet', 'iot', 'kap', 'lam', 'mu.', 'nu.', 'ksi', 'omi', 'pi.',
    'rho', 'sig', 'tau', 'ups', 'phi', 'chi', 'psi', 'w'
]

name_replace_exceptions = ['alf Cen A', 'alf Cen B']


def compare_star_names(name1, name2):

    def remove_suffix(name):
        return re.sub(r'\s?[ABC]$', '', name).strip()

    return remove_suffix(name1) == remove_suffix(name2)


def get_nearest_stars(plx=200, n=4):

    # print(Simbad.list_votable_fields())
    Simbad.add_votable_fields('plx', 'otype', 'sptype', 'flux(V)')

    print('Getting SIMBAD data...')

    result_tables = []
    ra = np.linspace(0, 360, num=n+1)

    for i in tqdm(range(n)):
        ra_min, ra_max = ra[i], ra[i + 1]
        query = f'plx > {plx} & ' +\
                "maintypes = '*' & maintype != '**' & sptypes <= 'M9' & "

        if ra_max == 360:
            query += f'ra >= {ra_min}'
        else:
            query += f'ra >= {ra_min} & ra < {ra_max}'

        result_tables.append(Table(Simbad.query_criteria(query)).to_pandas())

    result_table = pd.concat(result_tables, ignore_index=True)
    result_table.reset_index(inplace=True, drop=True)

    print('SIMBAD data downloaded.')

    result_table['HAS_COMPANION'] = False
    result_table['IS_COMPANION'] = False
    result_table['COMPANION_NAME'] = ''

    print('Processing SIMBAD data...')

    result_table = result_table.sort_values(by=['MAIN_ID'], ascending=True)

    name, previous_name, previous_index = '', '', None

    for index, row in result_table.iterrows():
        if previous_index is not None:
            name = row['MAIN_ID']
            if compare_star_names(name, previous_name):
                result_table.loc[previous_index, 'HAS_COMPANION'] = True
                result_table.loc[previous_index, 'COMPANION_NAME'] = name
                result_table.loc[previous_index, 'MAIN_ID'] = re.sub(r'\s?[ABC]$', '', previous_name).strip()
                result_table.loc[index, 'IS_COMPANION'] = True

        previous_name, previous_index = name, index

    for i in tqdm(range(len(result_table))):

        RA_sex, DEC_sex = result_table.at[i, 'RA'], result_table.at[i, 'DEC']
        c = SkyCoord(f'{RA_sex} {DEC_sex}', unit=(u.hourangle, u.deg))
        result_table.at[i, 'RA'], result_table.at[i, 'DEC'] = c.ra.degree, c.dec.degree

        result_table.at[i, 'MAIN_ID'] = re.sub(r'^\s*\*\s*', '', result_table.at[i, 'MAIN_ID'])

        name_split = result_table.at[i, 'MAIN_ID'].split(' ', 1)
        if name_split[0] == 'NAME':
            result_table.at[i, 'MAIN_ID'] = name_split[1]
        else:
            if name_split[0] in greek_letters and result_table.at[i, 'MAIN_ID'] not in name_replace_exceptions:

                try:
                    name_list = list(Simbad.query_objectids(result_table.at[i, 'MAIN_ID'])['ID'])
                    name_list = [elem for elem in name_list if elem.startswith("NAME")]

                    if len(name_list) == 1:
                        result_table.at[i, 'MAIN_ID'] = name_list[0].split(' ', 1)[1]
                    else:
                        for name in name_list:
                            if "Star" not in name and "star" not in name:
                                result_table.at[i, 'MAIN_ID'] = name.split(' ', 1)[1]
                                break
                except TypeError:
                    pass

        # Catalogue priority names: NAME, V*, Wolf, Ross, LHS, LFT, GJ, GI, G
        # Exclude: **

        result_table.at[i, 'MAIN_ID'] = re.sub(r'\s+', ' ', result_table.at[i, 'MAIN_ID'])
        result_table.at[i, 'MAIN_ID'] = re.sub(r'^V\*\s*', '', result_table.at[i, 'MAIN_ID'])

        if result_table.at[i, 'MAIN_ID'] == 'Proxima Centauri':
            result_table.at[i, 'MAIN_ID'] = 'Prox Cen'

    result_table['DIST'] = 1 / (result_table['PLX_VALUE'] / 1000)
    result_table['ABS_MAG_V'] = result_table['FLUX_V'] + 5 * (np.log10(result_table['PLX_VALUE'] / 1000) + 1)

    result_table['X'] = result_table['DIST'] * np.cos(result_table['DEC'].astype(np.float64)) * np.cos(result_table['RA'].astype(np.float64))
    result_table['Y'] = result_table['DIST'] * np.cos(result_table['DEC'].astype(np.float64)) * np.sin(result_table['RA'].astype(np.float64))
    result_table['Z'] = result_table['DIST'] * np.sin(result_table['DEC'].astype(np.float64))

    sol = {'MAIN_ID': 'Sol', 'X': 0.0, 'Y': 0.0, 'Z': 0.0,
           'FLUX_V': -26.74, 'SP_TYPE': 'G2V', 'ABS_MAG_V': 4.83, 'DIST': 0.0}

    result_table = result_table._append(sol, ignore_index=True)

    binary_filter = (result_table['IS_COMPANION'] == True)

    star_table = result_table[~binary_filter].reset_index(drop=True)
    companion_table = result_table[binary_filter].reset_index(drop=True)

    # star_table['MAIN_ID'] = star_table['MAIN_ID'].map(star_names).fillna(star_table['MAIN_ID'])

    print(f'Stars saved: {len(star_table)}')

    star_table.to_csv(f'simbad/Stars_plx_{plx}.csv', index=False)
    companion_table.to_csv(f'simbad/Stars_plx_{plx}_companions.csv', index=False)

    return star_table


def read_table(plx=50):
    pd.read_csv(f'simbad/Stars_plx_{plx}.csv')


if __name__ == '__main__':
    get_nearest_stars(10, n=256)
