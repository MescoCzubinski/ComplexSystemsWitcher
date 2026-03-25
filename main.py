from connections import build_all_connections
from graphs import draw_graph, load_graph
import os

DATA_DIR = 'books'
CHARACTERS_FILE = 'characters_list.txt'
OUTPUT_DIR = 'data'

if __name__ == '__main__':
    # build_all_connections(
    #     books_dir=DATA_DIR,
    #     characters_file=CHARACTERS_FILE,
    #     output_dir=OUTPUT_DIR,
    # )

    G = load_graph(os.path.join(OUTPUT_DIR, 'connections.csv'))
    draw_graph(G, 'Witcher character co-occurrence  (window = 1 sentence)', os.path.join(OUTPUT_DIR, 'graph.png'))
