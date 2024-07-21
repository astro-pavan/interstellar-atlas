import copy
import json
import numpy as np

spectral_type_tile = {
    'O': 'Spaceland.Space:/Q. Stars/01. Suns/sun029.png',
    'B': 'Spaceland.Space:/Q. Stars/01. Suns/sun029.png',
    'A': 'Spaceland.Space:/Q. Stars/01. Suns/sun029.png',
    'F': 'Spaceland.Space:/Q. Stars/01. Suns/sun028.png',
    'G': 'Spaceland.Space:/Q. Stars/01. Suns/sun007.png',
    'K': 'Spaceland.Space:/Q. Stars/01. Suns/sun024.png',
    'M': 'Spaceland.Space:/Q. Stars/01. Suns/sun012.png',
    'L': 'Spaceland.Space:/Q. Stars/01. Suns/sun017.png',
    'T': 'Spaceland.Space:/Q. Stars/01. Suns/sun017.png',
    'Y': 'Spaceland.Space:/G. Big Terra/hk-planet_017.png'
}

background_tile = 'Spaceland.Space:/A. Stillness Of Space/hk_empty-space_{n:03d}.png'
blue_screen_tile = 'Spaceland.Space:/E. Screen/01 Blue/bluscrn-002.png'
red_screen_tile = 'Spaceland.Space:/E. Screen/01 Blue/bluscrn-005.png'


blank_label = {
                'label': {
                    'text': '',
                    'visible': False,
                    'fontSize': 14,
                    'fontColor': '#000000',
                    'borderColor': '#000000',
                    'backgroundColor': '#FFFFFF',
                    'horizontalOffset': 0,
                    'verticalOffset': 0,
                    'opacity': 1
                },
                'data': '',
                'selected': False,
                'custom': False
            }

title_label = {
                'label': {
                    'text': '',
                    'visible': False,
                    'fontSize': 30,
                    'fontColor': '#000000',
                    'borderColor': '#000000',
                    'backgroundColor': '#FFFFFF',
                    'horizontalOffset': 100,
                    'verticalOffset': -55,
                    'opacity': 0
                },
                'data': '',
                'selected': False,
                'custom': False
            }


def make_hex_map(save, width, height, hex_data):

    with open('template.map') as json_file:
        data = json.load(json_file)

        infoLayer = data['infoLayer']  # list of dict
        layers = data['layers']  # list of dicts

        def change_tile(layer_id, coord, tile_name):
            layers[layer_id]['tiles'][coord] = {'source': tile_name, 'rotation': 0, 'mirror': False}

        def add_tile(layer_id, tile_name):
            tile = {'source': tile_name, 'rotation': 0, 'mirror': False} if tile_name is not None else None
            layers[layer_id]['tiles'].append(tile)

        def change_label(coord, text, title=False):
            label = copy.deepcopy(title_label if title else blank_label)
            label['label']['text'] = text
            label['label']['visible'] = True
            infoLayer[coord] = label

        size = width * height

        while len(layers[0]['tiles']) != size:
            add_tile(0, background_tile.format(n=np.random.randint(1, 80)))

        while len(layers[1]['tiles']) != size:
            add_tile(1, blue_screen_tile)

        while len(layers[2]['tiles']) != size:
            add_tile(2, None)
            add_tile(3, None)
            add_tile(4, None)
            infoLayer.append(dict(blank_label))

        change_tile(1, 0, blue_screen_tile)

        for d in hex_data:

            if d[0].endswith(')'):
                change_tile(1, d[1], red_screen_tile)

            change_tile(2, d[1], spectral_type_tile[d[2]])
            change_label(d[1], d[0])

        data['width'], data['height'] = width, height
        data['infoLayer'] = infoLayer
        data['layers'] = layers

        with open(save, 'w') as f:
            f.write(json.dumps(data, separators=(',', ':'), indent='\t'))


# make_hex_map('new.map', 6, 6, [])
