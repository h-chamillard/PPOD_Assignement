import pandas as pd
import ast
import re
import spacy
from spacy.tokens import Doc
from anonymizedf.anonymizedf import anonymize

import pandas as pd
import ast
from collections import defaultdict


def extract_entities(pii_list):
    entities = [(pii[0], pii[1]) for pii in pii_list if pii[1] in ['PERSON', 'ORG', 'EVENT', 'LOC', 'GPE', 'URL', 'FAMILY_MEMBER']]
    return entities


def create_pseudo_map(entities):
    pseudo_map = {}
    counters = defaultdict(int)

    for entity, type_ in entities:
        counters[type_] += 1
        if type_ == 'FAMILY_MEMBER' or type_ == 'URL':
            pseudo_map[entity] = f"{type_}"
        else :
            pseudo_map[entity] = f"{type_}_{counters[type_]}"

    return pseudo_map


def anonymize_review(review, entities, pseudo_map):
    anonymized_review = review
    for entity, _ in entities:  # Nous n'avons pas besoin du type ici, juste de l'entité
        if entity in pseudo_map:
            anonymized_entity = f"€€{pseudo_map[entity]}€€"
            anonymized_review = anonymized_review.replace(entity, anonymized_entity)
    return anonymized_review


def anonymize_reviews(dataframe):
    dataframe['PII'] = dataframe['PII'].apply(ast.literal_eval)
    dataframe['ENTITIES'] = dataframe['PII'].apply(extract_entities)

    all_entities = [entity for sublist in dataframe['ENTITIES'] for entity in sublist]
    pseudo_map = create_pseudo_map(all_entities)

    dataframe['Review'] = dataframe.apply(
        lambda row: anonymize_review(row['Review'], row['ENTITIES'], pseudo_map), axis=1)

    return dataframe


def anonymize_time_en(text):
    time_patterns = {
        r'\b\d{1,2}(st|nd|rd|th)?\b(?!\s(?:am|pm))': '€€OrdinalDate€€',

        r'\b\d{1,2}(:\d{2})?\s?(am|pm)?\b': '€€SpecificTime€€',

        r'\b(nights?|days?)\b': '€€ShortPeriod€€',
        r'\b(weeks?)\b': '€€MediumPeriod€€',
        r'\b(months?|years?)\b': '€€LongPeriod€€',
        r'\b(weekends?|weekday)\b': '€€FewDays€€',
        r'\b(summer|winter|spring|fall|autumn)\b': '€€Season€€',

        r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekday)s?\b': '€€Weekday€€',

        r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan\.?|feb\.?|mar\.?|apr\.?|jun\.?|jul\.?|aug\.?|sep\.?|sept\.?|oct\.?|nov\.?|dec\.?)\b': '€€Month€€',

        r'\b(early|mid|late)?\s?\d{4}\b': '€€SpecificYear€€',

        r'\b\d{1,2}(st|nd|rd|th)?(?:-\d{1,2}(st|nd|rd|th)?)?\b': '€€SpecificDay€€',

        r'\b(spring|summer|fall|autumn|winter)\b': '€€Season€€',

        r'\b(new year\'?s?( eve)?|christmas(eve| day)?|easter|thanksgiving|valentine\'?s? day|halloween|memorial day|labor day|independence day|veterans day)\b': '€€Holiday€€',

    }

    segments = re.split(r'(\,|\.)\s*', text)
    anonymized_segments = []

    for segment in segments:
        found = False
        for pattern, replacement in time_patterns.items():
            if re.search(pattern, segment, flags=re.IGNORECASE):
                anonymized_segments.append(replacement)
                found = True
                break
        if not found:
            anonymized_segments.append("€€GeneralTime€€")

    return ' '.join(anonymized_segments)



def anonymize_dates_en(text):
    date_patterns = {
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b': '€€SpecificDate€€',
        r'\b\d{1,2}(st|nd|rd|th)? (January|February|March|April|May|June|July|August|September|October|November|December),? \d{4}\b': '€€SpecificMonthYear€€',
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}(st|nd|rd|th)?,? \d{4}\b': '€€SpecificMonthYear€€',
        r'\b\d{4}-\d{2}-\d{2}\b': '€€ISODate€€',

        r'\b\d{4}\b': '€€Year€€',
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b': '€€Month€€',

        r'\b(today|yesterday|tomorrow|tonight|last night|this morning|this evening)\b': '€€RelativeDay€€',

        r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b': '€€Weekday€€',

        r'\b(night?|nights?|days?)\b': '€€ShortPeriod€€',
        r'\b(weeks?)\b': '€€MediumPeriod€€',
        r'\b(months?|years?)\b': '€€LongPeriod€€',
        r'\b(weekends?|weekday)\b': '€€FewDays€€',
        r'\b(summer|winter|spring|fall|autumn)\b': '€€Season€€',

        r'\b(the day after tomorrow|the day before yesterday)\b': '€€RelativeDayComplex€€',
        r'\b(in the past|recently|lately|in recent times|over the last few days|in the last year)\b': '€€GeneralPast€€',
        r'\b(ago)\b': '€€Ago€€',

        r'\b(early|mid|late)-(morning|afternoon|evening|night)\b': '€€PartOfDay€€',
        r'\b(early|mid|late) (January|February|March|April|May|June|July|August|September|October|November|December)\b': '€€PartOfMonth€€',
        r'\b(early|mid|late) \d{4}\b': '€€PartOfYear€€',

        r'\b\d{1,2}(:\d{2})?\s?(am|pm)\b': '€€SpecificTime€€',
        r'\b\d{1,2}:\d{2}\b': '€€Time€€',

        r'\b(new year\'?s?( eve)?|christmas(eve| day)?|easter|thanksgiving|valentine\'?s? day|halloween|memorial day|labor day|independence day|veterans day)\b': '€€Holiday€€',

        r'\b(\d+)(st|nd|rd|th)?\s*(birthday|anniversary)\b': '€€SpecialOccasion€€',
        r'\b\d+\s+year\s+olds?\b': '€€LongPeriod olds€€',
    }

    segments = re.split(r'(\,|\.)\s*', text)
    anonymized_segments = []

    for segment in segments:
        found = False
        for pattern, replacement in date_patterns.items():
            if re.search(pattern, segment, flags=re.IGNORECASE):
                anonymized_segments.append(replacement)
                found = True
                break
        if not found:
            anonymized_segments.append("€€GeneralDate€€")

    return ''.join(anonymized_segments)


# Fonction pour lire le fichier et traiter chaque ligne
def process_file(df):
    nlp = spacy.load("en_core_web_sm")

    # Traitement de chaque ligne
    for index, row in df.iterrows():
        text = row['Review']
        doc = nlp(text)

        anonymized_text = text  # Initialiser avec le texte original
        for ent in doc.ents:
            if ent.label_ == "DATE":
                anonymized_date = anonymize_dates_en(ent.text)
                anonymized_text = anonymized_text.replace(ent.text, anonymized_date)

            if ent.label_ == "TIME":
                anonymized_time = anonymize_time_en(ent.text)
                anonymized_text = anonymized_text.replace(ent.text, anonymized_time)

        # Mettre à jour le texte anonymisé dans le DataFrame
        df.at[index, 'Review'] = anonymized_text  # Assurez-vous que cette colonne existe ou est correctement nommée

    return df


def remove_euro_markers(text):
    clean_text = text.replace('€€', '')
    return clean_text


def clean_reviews_from_markers(dataframe):
    dataframe['Review'] = dataframe['Review'].apply(remove_euro_markers)
    return dataframe


if __name__ == '__main__':
    file_path = 'Last_Assignment_Dataset/pii_analysis_results.csv'
    df = pd.read_csv(file_path, nrows=1000)

    df = process_file(df)
    df = anonymize_reviews(df)

    # df = clean_reviews_from_markers(df)

    df['Review'].to_csv('Last_Assignment_Dataset/pii_analysis_transformed.csv', index=False)