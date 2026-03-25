import re
import csv
import os
import sys
from itertools import combinations
from collections import defaultdict


def load_characters(filepath):
    canonical = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            canonical.append(name)

    canonical.sort(key=len, reverse=True)
    name_normalizer = {n.lower(): n for n in canonical}

    first_word_map = defaultdict(list)
    for name in canonical:
        fw = name.split()[0]
        if len(fw) >= 4:
            first_word_map[fw.lower()].append(name)

    for fw_lower, chars in first_word_map.items():
        if len(chars) == 1:
            full_name = chars[0]
            if len(full_name.split()) > 1:
                if fw_lower not in name_normalizer:
                    name_normalizer[fw_lower] = full_name

    return canonical, name_normalizer


def build_combined_pattern(name_normalizer):
    terms = sorted(name_normalizer.keys(), key=len, reverse=True)
    escaped = [re.escape(t) for t in terms]
    pattern = r'\b(' + '|'.join(escaped) + r')\b'
    return re.compile(pattern, re.IGNORECASE)


def split_into_sentences(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    parts = re.split(r'(?<=[.!?…])\s+', text)
    return [p.strip() for p in parts if len(p.strip()) > 10]


def cooccurrences(sentence_chars):
    counts = defaultdict(int)
    for chars in sentence_chars:
        for a, b in combinations(sorted(chars), 2):
            counts[(a, b)] += 1
    return counts


def build_all_connections(books_dir, characters_file, output_dir='.'):
    characters, name_normalizer = load_characters(characters_file)
    pattern = build_combined_pattern(name_normalizer)

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

        sentences = split_into_sentences(text)
        sentence_chars = []
        for sent in sentences:
            found = set()
            for m in pattern.finditer(sent):
                can = name_normalizer.get(m.group(1).lower())
                if can:
                    found.add(can)
            sentence_chars.append(found)

        for pair, cnt in cooccurrences(sentence_chars).items():
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
