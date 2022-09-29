import pandas as pd
import spacy
import unicodedata
from nltk.corpus import stopwords
from string import punctuation
import json
from fuzzywuzzy import process


class ExtractArticleEntities:
    """ Extract article entities from a document using natural language processing (NLP) and fuzzy matching.

Parameters

- text: a string or the text of a news article to be parsed

Usage:
import ExtractArticleEntities
instantiate with text parameter ie. entities = ExtractArticleEntities(text)
retrieve Who, What, When, Where entities with entities.www_json
Non-organised entities with entiities.json


"""

    def __init__(self, text):
        self.text = text  # preprocess text at initialisation
        self.text = self.preprocessing(self.text)
        self.json = {}
        # Create empty dataframe to hold entity data for ease of processing
        self.entity_df = pd.DataFrame(columns=["entity", "description"])
        # Load the spacy model
        self.nlp = spacy.load('en_core_web_lg')
        # Parse the text
        self.entity_df = self.get_who_what_where_when()
        # Disambiguate entities
        self.entity_df = self.fuzzy_disambiguation()
        # Create JSON representation of entities
        self.entity_df = self.entity_df.drop_duplicates(subset=["description"])
        self.entity_df = self.entity_df.reset_index(drop=True)

        # ungrouped entity returned as json
        self.json = self.entity_json()
        # return json with entities grouped into who, what, where, when keys
        self.www_json = self.get_wwww_json()

    def fuzzy_disambiguation(self):
        # Load the entity data
        self.entity_df['fuzzy_match'] = ''
        # Load the entity data
        person_choices = self.entity_df.loc[self.entity_df['entity'] == 'PERSON']
        org_choices = self.entity_df.loc[self.entity_df['entity'] == 'ORG']
        where_choices = self.entity_df.loc[self.entity_df['entity'] == 'GPE']
        norp_choices = self.entity_df.loc[self.entity_df['entity'] == 'NORP']
        loc_choices = self.entity_df.loc[self.entity_df['entity'] == 'LOC']
        date_choices = self.entity_df.loc[self.entity_df['entity'] == 'DATE']

        def fuzzy_match(row, choices):
            '''This function disambiguates entities by looking for maximum three matches with a score of 80 or more
            for each of the entity types. If there is no match, then the function returns None. '''
            match = process.extract(row["description"], choices["description"], limit=3)

            match = [m[0] for m in match if m[1] > 80 and m[1] != 100]
            if len(match) == 0:
                match = None

            return match

        # Apply the fuzzy matching function to the entity dataframe
        for i, row in self.entity_df.iterrows():

            if row['entity'] == 'PERSON':

                self.entity_df.at[i, 'fuzzy_match'] = fuzzy_match(row, person_choices)
            elif row['entity'] == 'ORG':

                self.entity_df.at[i, 'fuzzy_match'] = fuzzy_match(row, org_choices)
            elif row['entity'] == 'GPE':

                self.entity_df.at[i, 'fuzzy_match'] = fuzzy_match(row, where_choices)

            elif row['entity'] == 'NORP':

                self.entity_df.at[i, 'fuzzy_match'] = fuzzy_match(row, norp_choices)
            elif row['entity'] == 'LOC':

                self.entity_df.at[i, 'fuzzy_match'] = fuzzy_match(row, loc_choices)
            elif row['entity'] == 'DATE':

                self.entity_df.at[i, 'fuzzy_match'] = fuzzy_match(row, date_choices)

        return self.entity_df

    def preprocessing(self, text):
        """This function takes a text string and strips out all punctuation. It then normalizes the string to a
        normalized form (using the "NFKD" normalization algorithm). Finally, it strips any special characters and
        converts them to their unicode equivalents. """

        # remove punctuation
        text = text.translate(str.maketrans("", "", punctuation))
        # normalize the text
        stop_words = stopwords.words('english')
        filtered_words = [word for word in self.text.split() if word not in stop_words]

        # This is very hacky. Need a better way of handling bad encoding
        pre_text = " ".join(filtered_words)
        pre_text = pre_text = pre_text.replace('  ', ' ')
        pre_text = pre_text.replace('â€™', "'")
        pre_text = pre_text.replace('â€œ', '"')
        pre_text = pre_text.replace('â€', '"')
        pre_text = pre_text.replace('â€˜', "'")
        pre_text = pre_text.replace('â€¦', '...')
        pre_text = pre_text.replace('â€“', '-')
        pre_text = pre_text.replace("\x9d", '-')
        # normalize the text
        pre_text = unicodedata.normalize("NFKD", pre_text)
        # strip punctuation again as some remains in first pass
        pre_text = pre_text.translate(str.maketrans("", "", punctuation))


        return pre_text

    def get_who_what_where_when(self):
        """Get entity information in a document.


This function will return a DataFrame with the following columns:

- entity: the entity being queried
- description: a brief description of the entity

Usage:

get_who_what_where_when(text)

Example:

> get_who_what_where_when('This is a test')

PERSON
ORG
GPE
LOC
PRODUCT
EVENT
LAW
LANGUAGE
NORP
DATE
GPE
TIME"""

        # list to hold entity data
        article_entity_list = []
        # tokenize the text
        doc = self.nlp(self.text)
        # iterate over the entities in the document but only keep those which are meaningful
        desired_entities = ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'LAW', 'LANGUAGE', 'NORP', 'DATE', 'GPE',
                            'TIME']

        for ent in doc.ents:

            if ent.label_ in desired_entities:
                # add the entity to the list
                entity_dict = {ent.label_: ent.text}

                article_entity_list.append(entity_dict)

        # dedupe the entities but only on exact match of values as occasional it will assign an ORG entity to PER
        deduplicated_entities = {frozenset(item.values()):
                                     item for item in article_entity_list}.values()
        # create a dataframe from the entities
        for record in deduplicated_entities:
            record_df = pd.DataFrame(record.items(), columns=["entity", "description"])
            self.entity_df = pd.concat([self.entity_df, record_df], ignore_index=True)

        return self.entity_df

    def entity_json(self):
        """Returns a JSON representation of an entity defined by the `entity_df` dataframe. The `entity_json` function
        will return a JSON object with the following fields:
        - entity: The type of the entity in the text
        - description: The name of the entity as described in the input text
        - fuzzy_match: A list of fuzzy matches for the entity. This is useful for disambiguating entities that are similar
       """

        self.json = json.loads(self.entity_df.to_json(orient='records'))
        # self.json = json.dumps(self.json, indent=2)
        return self.json

    def get_wwww_json(self):
        """This function returns a JSON representation of the `get_who_what_where_when` function. The `get_www_json`
        function will return a JSON object with the following fields:
        - entity: The type of the entity in the text
        - description: The name of the entity as described in the input text
        - fuzzy_match: A list of fuzzy matches for the entity. This is useful for disambiguating entities that are similar
        """

        # create a json object from the entity dataframe
        who_dict = {"who": [ent for ent in self.entity_json() if ent['entity'] in ['ORG', 'PERSON']]}
        where_dict = {"where": [ent for ent in self.entity_json() if ent['entity'] in ['GPE', 'LOC']]}
        when_dict = {"when": [ent for ent in self.entity_json() if ent['entity'] in ['DATE', 'TIME']]}
        what_dict = {
            "what": [ent for ent in self.entity_json() if ent['entity'] in ['PRODUCT', 'EVENT', 'LAW', 'LANGUAGE',
                                                                             'NORP']]}
        article_wwww = [who_dict, where_dict, when_dict, what_dict]

        self.wwww_json = json.dumps(article_wwww,indent=2)

        return self.wwww_json
