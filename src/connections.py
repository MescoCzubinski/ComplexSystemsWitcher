import pandas as pd
import numpy as np
import os
import spacy


def load_characters(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip() and line.strip() != 'character']

    firstname_count = {}
    for name in names:
        firstname = name.split()[0]
        firstname_count[firstname] = firstname_count.get(firstname, 0) + 1

    rows = []
    for name in names:
        firstname = name.split()[0]
        unique_firstname = firstname if firstname_count[firstname] == 1 else None
        rows.append({"character": name, "character_firstname": unique_firstname})

    return pd.DataFrame(rows)


def ner(file_name):
    nlp = spacy.load("en_core_web_sm")
    book_text = open(file_name, encoding='utf-8').read()
    book_doc = nlp(book_text)
    return book_doc


def get_ne_list_per_sentence(spacy_doc):
    sent_entity_df = []
    for sent in spacy_doc.sents:
        entity_list = [ent.text for ent in sent.ents]
        sent_entity_df.append({"sentence": sent, "entities": entity_list})
    return pd.DataFrame(sent_entity_df)


def filter_entity(ent_list, character_df):
    firstname_to_full = {
        row.character_firstname: row.character
        for row in character_df.itertuples()
        if row.character_firstname is not None
    }
    result = []
    for ent in ent_list:
        if ent in character_df['character'].values:
            result.append(ent)
        elif ent in firstname_to_full:
            result.append(firstname_to_full[ent])
    return result


def create_relationships(df, window_size):
    relationships = []
    for i in range(df.index[-1]):
        end_i = min(i + window_size, df.index[-1])
        char_list = sum((df.loc[i: end_i].character_entities), [])
        char_unique = [char_list[i] for i in range(len(char_list))
                       if (i == 0) or char_list[i] != char_list[i - 1]]
        if len(char_unique) > 1:
            for idx, a in enumerate(char_unique[:-1]):
                b = char_unique[idx + 1]
                relationships.append({"character1": a, "character2": b})

    relationship_df = pd.DataFrame(relationships)
    relationship_df = pd.DataFrame(np.sort(relationship_df.values, axis=1),columns=relationship_df.columns)
    relationship_df["weight"] = 1
    relationship_df = relationship_df.groupby(["character1", "character2"],sort=False,as_index=False).sum()
    return relationship_df


def build_connections(window_size=5):
    character_df = load_characters(os.path.join('data', 'characters.csv'))

    book_files = sorted(
        os.path.join('books', f)
        for f in os.listdir('books')
        if f.endswith('.txt')
    )

    all_relationships = []

    for book_path in book_files:
        print(f'{os.path.basename(book_path)}')

        book_doc = ner(book_path)
        sent_entity_df = get_ne_list_per_sentence(book_doc)
        sent_entity_df['character_entities'] = sent_entity_df['entities'].apply(lambda ent_list: filter_entity(ent_list, character_df))

        relationship_df = create_relationships(sent_entity_df, window_size)
        all_relationships.append(relationship_df)

    final_df = pd.concat(all_relationships, ignore_index=True)
    final_df = final_df.groupby(["character1", "character2"], sort=False, as_index=False).sum()
    final_df = final_df.sort_values("weight", ascending=False)

    out_path = os.path.join('data', 'connections.csv')
    final_df.to_csv(out_path, index=False)

    return final_df


if __name__ == '__main__':
    build_connections()
