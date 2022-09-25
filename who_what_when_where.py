# -*- coding: utf-8 -*-
# library to assist with disambiguation
import disamby.preprocessors as pre
import pandas as pd
import spacy
import unicodedata
from disamby import Disamby
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter
# text preprocessing library
from nltk.corpus import stopwords



class ExtractArticleEntities:
    '''
    Class to extract the Who, What,When,Where, Entities
    Parameters:

text: The text to be processed

Preprocessing:

The text is preprocessed using a sequence tagger. This tagger recognises the entities and their relationships.

The disambiguation threshold is set to 0.4. This value can be set between 0-1 to fine-tune results of disambiguation.
It is normally effective around 0.4-0.6.

The NLPSparse module is loaded to perform the natural language processing.

Results:

The results dictionary is created and contains the following information:

- entity: The name of the entity in the text
- text: The text as it was processed by the tagger'''

    def __init__(self, text):
        self.text = text  # preprocess text at initialisation
        self.text = self.preprocessing(self.text)
        self.tagger = SequenceTagger.load('ner')
        self.entity_df = pd.DataFrame(columns=["entity", "text"])
        self.disambiguation_threshold = 0.4
        self.nlp = spacy.load('en_core_web_lg')
        self.results = {}

    def disambiguate_entities(self, entities):
        """This function disambiguates entities by removing punctuation, normalizing whitespace, and pre-
processing the text. It then uses a threshold to determine if there are any
possible matches, and if so, updates the 'matches' column in the entity dataframe with the
result. Finally, it returns the entity dataframe."""

        pipeline = [
            pre.normalize_whitespace,
            pre.remove_punctuation,
            lambda x: pre.trigram(x) + pre.split_words(x)

        ]
        for record in entities:
            # attempt to get a confidence score for the entity

            record_df = pd.DataFrame(record.items(), columns=["entity", "text"])
            self.entity_df = pd.concat([self.entity_df, record_df], ignore_index=True)

        disambiguate = Disamby(self.entity_df['text'], preprocessors=pipeline)

        candidate_matches = disambiguate.disambiguated_sets(threshold=self.disambiguation_threshold)
        # create a placeholder column for matches
        self.entity_df['matches'] = ''
        for candidate in candidate_matches:
            # if there is a possible match then update a match column in the entity dataframe with it
            if len(candidate) >= 2:
                candidate = list(candidate)  # convert set to list for indexing
                self.entity_df.at[candidate[0], 'matches'] = self.entity_df.loc[candidate[1]].values[
                    1]  # all matches if more than
                # one

        # Generate a confidence score where possible
        for i, row in self.entity_df.iterrows():
            self.entity_df.at[i, 'confidence'] = self.flair_confidence_calculation(row['text'])

        return self.entity_df

    def preprocessing(self, text):
        """This function takes a string and removes all punctuation except -. It also removes malformed coding like
        â€™,â€œ,â€œ,â€˜, and â€¦. It then splits the string into words and removes common stopwords before finally
        normalizing the string to deal with outlier encoding issues """

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

        pre_text = unicodedata.normalize("NFKD", pre_text)

        return pre_text

    def flair_confidence_calculation(self, entity):
        """This function takes an entity as input and uses a sentence splitter to break it down into individual
        spans. It then uses the tagger to predict how confident the sentence is in relation to other sentences in the
        same span. Finally, it calculates the confidence for the entity and returns it. """
        splitter = SegtokSentenceSplitter()
        sentences = splitter.split(entity)
        for sentence in sentences:
            self.tagger.predict(sentence)
            for entity in sentence.get_spans('ner'):
                confidence = entity.score
                return confidence

    def get_who_what_where_when(self):
        """This function extracts entities from text using the spacy model. It takes a text document as input and
        returns a list of entities and their respective tags. It also attempts a deduplication of the initial
        results and if a match is found, the entity's corresponding text is returned.
         It then passes these results over to the disambiguate_entities function to attempt to disambiguate"""


        # dictionary to hold entity.tag/text pairs
        article_entity_list = []
        doc = self.nlp(self.text)
        desired_entities = ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'LAW', 'LANGUAGE', 'NORP', 'DATE', 'GPE',
                            'TIME']

        for ent in doc.ents:

            if ent.label_ in desired_entities:
                entity_dict = {ent.label_: ent.text}

                article_entity_list.append(entity_dict)
        # dedupe the entities but only on exact match of values as occasional it will assign an ORG entity to PER

        deduplicated_entities = {frozenset(item.values()):
                                     item for item in article_entity_list}.values()

        self.disambiguate_entities(deduplicated_entities)

    def entity_json(self):
        """Returns a JSON representation of an entity defined by the `entity_df` dataframe. The `entity_json` function
        will return a JSON object with the following fields:
        - entity: The type of the entity in the text
        - text: The name of the entity as described in the input text
        - matches: Any similar entities that were found in the text during disambiguation
        - confidence: The confidence score for the entity as determined by the `flair_confidence_calculation` function
        """
        self.json = self.entity_df.to_json(orient='records')
        return self.json
