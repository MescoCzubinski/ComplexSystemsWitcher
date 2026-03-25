import csv
import os
import sys
from itertools import combinations
from collections import defaultdict
import spacy


def load_characters(filepath):
    """
    Returns two dicts mapping to canonical name:
      full_name_lookup  : 'Geralt of Rivia' -> 'Geralt of Rivia'
      firstname_lookup  : 'Geralt'          -> 'Geralt of Rivia'
    First-name entries are only added when the first name is unique across all characters.
    """
    names = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)

    full_lookup = {n: n for n in names}

    firstname_map = defaultdict(list)
    for name in names:
        firstname_map[name.split()[0]].append(name)

    firstname_lookup = {
        fw: chars[0]
        for fw, chars in firstname_map.items()
        if len(chars) == 1 and fw not in full_lookup
    }

    return full_lookup, firstname_lookup


def filter_entities(ents, full_lookup, firstname_lookup):
    found = set()
    for ent in ents:
        if ent in full_lookup:
            found.add(full_lookup[ent])
        elif ent in firstname_lookup:
            found.add(firstname_lookup[ent])
    return found


def cooccurrences(sentence_chars):
    counts = defaultdict(int)
    for chars in sentence_chars:
        for a, b in combinations(sorted(chars), 2):
            counts[(a, b)] += 1
    return counts


def build_all_connections(books_dir, characters_file, output_dir='.'):
    full_lookup, firstname_lookup = load_characters(characters_file)

    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 2_000_000

    book_files = sorted(
        os.path.join(books_dir, f)
        for f in os.listdir(books_dir)
        if f.endswith('.txt')
    )
    if not book_files:
        print('ERROR: no .txt files found in', books_dir)
        sys.exit(1)

    totals = defaultdict(int)

    for book_idx, book_path in enumerate(book_files):
        book_name = os.path.basename(book_path)
        print(f'[{book_idx + 1}/{len(book_files)}] {book_name}')

        with open(book_path, 'r', encoding='utf-8') as f:
            text = f.read()

        doc = nlp(text)

        sentence_chars = []
        for sent in doc.sents:
            ents = [ent.text for ent in sent.ents if ent.label_ == 'PERSON']
            found = filter_entities(ents, full_lookup, firstname_lookup)
            sentence_chars.append(found)

        windows = [
            sentence_chars[i] | sentence_chars[i + 1]
            for i in range(len(sentence_chars) - 1)
        ]
        for pair, cnt in cooccurrences(windows).items():
            totals[pair] += cnt

    os.makedirs(output_dir, exist_ok=True)

    rows = sorted(
        ((a, b, w) for (a, b), w in totals.items()),
        key=lambda x: -x[2],
    )
    out_path = os.path.join(output_dir, 'connections.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['name1', 'name2', 'weight'])
        writer.writerows(rows)
    print(f'  {out_path}  ({len(rows)} connections)')

    return totals


if __name__ == '__main__':
    build_all_connections(
        books_dir='books',
        characters_file='characters_list.txt',
        output_dir='data',
    )
