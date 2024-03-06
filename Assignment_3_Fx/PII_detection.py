import spacy
import pandas as pd
from collections import Counter
import re


def pii_detection(iteration):

    nlp = spacy.load("en_core_web_sm")

    config = {"overwrite_ents": True}
    ruler = nlp.add_pipe("entity_ruler", config=config, before="ner")

    patterns = [
        {"label": "PHONE_NUMBER", "pattern": [{"TEXT": {"REGEX": r"\(\d{3}\) \d{3}-\d{4}"}}]},
        {"label": "URL", "pattern": [{"TEXT": {"REGEX": r"www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"}}]},
        {"label": "AGE", "pattern": [
            {"TEXT": {"REGEX": r"(\bages?\s\d{1,2}(?:\s\d{1,2})?|\d{1,2}\s1?/?2?\s(?:years?|yrs?|months?)\sold)\b"}}]},
        {"label": "FAMILY_MEMBER", "pattern": [{"TEXT": {
            "REGEX": r"\b(children|kids|sons?|daughters?|childs?|sisters?|brothers?|mothers?|fathers?|husbands?|wives|girlfriends?|gfs?|boyfriends?|bfs?|aunts?|uncles?|grandmothers?|grandfathers?|nieces?|nephews?|cousins?)\b"}}]}
    ]

    ruler.add_patterns(patterns)


    if iteration :
        df = pd.read_csv('Last_Assignment_Dataset/pii_analysis_transformed.csv')
    else :
        file_path = 'Last_Assignment_Dataset/tripadvisor_hotel_reviews.csv'
        df = pd.read_csv(file_path, nrows=2000)

        df.to_csv("Last_Assignment_Dataset/tripadvisor_short.csv", index=False, sep=';')
        df = pd.read_csv('Last_Assignment_Dataset/tripadvisor_short.csv', sep=';')


    def clean_text_from_anonymized_markers(text):
        anonymized_pattern = re.compile(r'€€.*?€€')
        clean_text = anonymized_pattern.sub(' ', text)
        return clean_text


    def find_pii(text, nlp):
        clean_text = clean_text_from_anonymized_markers(text)
        doc = nlp(clean_text)
        pii_entities = [(ent.text, ent.label_) for ent in doc.ents if
                        ent.label_ in ["PERSON", "DATE", "LOC", "GPE", "ORG", "EVENT", "EMAIL", "PHONE_NUMBER", "URL",
                                       "AGE", "FAMILY_MEMBER", "TIME"]]
        return pii_entities

    pii_results = [find_pii(review, nlp) for review in df['Review']]

    results_df = pd.DataFrame({
        'Review': df['Review'],
        'PII': pii_results
    })

    results_df.to_csv('Last_Assignment_Dataset/pii_analysis_results.csv', index=False)


    all_pii_entities = []
    for review in df['Review']:
        pii_entities = find_pii(review, nlp)
        all_pii_entities.extend(pii_entities)

    pii_counter = Counter([label for text, label in all_pii_entities])

    for label, count in pii_counter.items():
        print(f"{label}: {count}")


