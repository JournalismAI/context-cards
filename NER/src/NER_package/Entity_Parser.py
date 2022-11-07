import requests
import wikipedia
import pandas as pd
import spacy
import unicodedata
import numpy as np
import nltk
from newspaper import Article
nltk.download('stopwords')
from string import punctuation
import json
from datetime import datetime, timedelta
from io import BytesIO
from SPARQLWrapper import SPARQLWrapper, JSON
from fuzzywuzzy import process


sparql = SPARQLWrapper('https://dbpedia.org/sparql')

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

        self.entity_df = self.get_related_from_wiki()
        self.entity_df = self.get_page_view_wikipedia()
        # Create JSON representation of entities
        self.entity_df = self.entity_df.drop_duplicates(subset=["description"])
        self.entity_df = self.entity_df.reset_index(drop=True)
        
        # ungrouped entity returned as json

        key_column = ['entity', 'description', 'Matched Entity', 'Views']
        self.entity_df = self.entity_df[key_column]
        self.entity_df = self.get_image_and_description()
        self.json = self.entity_json()

        # return json with entities grouped into who, what, where, when keys
        # self.www_json = self.get_wwww_json()


    def get_page_view_wikipedia(self):
        '''
        This function takes input as matched entity and returns output as its page view count

        '''
        view_list = []
        for entity in self.entity_df['Matched Entity']:
            if entity:
                # We are just going to take input as first entity from matched_entity filed
                entity_to_look = entity[0]
                entity_to_look = entity_to_look.replace(' ','_')

                headers = {
                    'accept': 'application/json',
                    'User-Agent': 'Foo bar'
                }
                
                now = datetime.now()
                now_dt = now.strftime(r'%Y%m%d')
                # Taking a window frame of 7 days for getting page_view
                week_back = now - timedelta(days=7)
                week_back_dt = week_back.strftime(r'%Y%m%d')

                resp = requests.get(f'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/{entity_to_look}/daily/{week_back_dt}/{now_dt}', headers=headers)
                data = resp.json()
                df = pd.json_normalize(data['items'])
                view_count = sum(df['views'])
                
            else:
                view_count = 0
            view_list.append(view_count)

        self.entity_df['Views'] = view_list
            
        if 'Views' in self.entity_df:
            self.entity_df = self.entity_df.sort_values(by=['Views'], ascending=False).reset_index(drop=True)
        
        # Filtering out Entity which have value as 0, which in turn means that either entity does not have wiki page
        # or not enough views to be considered
     
        self.entity_df = self.entity_df[self.entity_df['Views']!=0]
        return self.entity_df

    def get_related_from_wiki(self):
        
        """
        This function gets wikipedia entity from existing entity_df.
        Right now the function is taking input entity as 'PERSON','ORG','GPE','NORP','LOC' only.
        It then searches on wikipedia and returns related wiki entity which is in turn added as a new column
        'Wikipedia Entity'

        After getting entities from wikipedia, it matches those entities based on fuzzy match with the root input word
        And return only those entity which have fuzzy match score more than a particular threshold under column 'Matched Entity'
        """

        names = self.entity_df.description
        entities = self.entity_df.entity
        self.related_entity = []
        match_scores = []
        for name, entity in zip(names, entities):
            if entity in ('PERSON','ORG','GPE','NORP','LOC'):
                related_names = wikipedia.search(name, 10)
                self.related_entity.append(related_names)
                matches = process.extract(name, related_names)
                match_scores.append([match[0] for match in matches if match[1]>= 90 ])
            else:
                self.related_entity.append([None])
                match_scores.append([])
                # Remove nulls
        
        self.entity_df['Wikipedia Entity'] = self.related_entity
        self.entity_df['Matched Entity'] = match_scores

        return self.entity_df

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
                match = []

            if match:
                self.fuzzy_match_dict[row["description"]] = match
            return match

        # Apply the fuzzy matching function to the entity dataframe

        self.fuzzy_match_dict = {}
            
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

        filtered_words = [word for word in self.text.split()]

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
        self.label_dict = {}

        # stop_words = stopwords.words('english')
        for ent in doc.ents:

            self.label_dict[ent] = ent.label_
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
    
    def filter_wiki_df(self, df):
        key_list = df.keys()[:2]
        df = df[key_list]
        df['Match Check'] = np.where(df[df.keys()[0]] != df[df.keys()[1]], True, False)
        df = df[df['Match Check']!= False]
        df = df[key_list]
        df = df.dropna(how='any').reset_index(drop=True)
        df.rename(columns = {key_list[0]: 'Attribute', key_list[1]: 'Value'}, inplace = True)
        return df

    def get_image_and_description(self):
        self.image_url_list = []
        self.comment_list = []
        self.context_list = []
        for entity_value in self.entity_df.values:
            label, name, related, view = entity_value
            related = related[0] if len(related) > 0 else ''

            related_entity_list = [name, related]
            related_entity = entity_value[1:-1]

            all_related_entity = [related_entity[0]]

            for el in related_entity[1]:
                if isinstance(el, str):
                    all_related_entity.append(el)
                elif isinstance(el, int):
                    all_related_entity.append(str(el))
                else:
                    all_related_entity.extend(el)

            for entity in all_related_entity:
                
                if True:
                    if 'disambiguation' in entity:
                        entity = entity.replace(' (disambiguation)', '')
                    entity = entity.replace(' ', '_')
                    query = f'''
                        SELECT ?name ?comment ?image
                        WHERE {{ dbr:{entity} rdfs:label ?name.
                                dbr:{entity} rdfs:comment ?comment.
                                dbr:{entity} dbo:thumbnail ?image.
                        
                            FILTER (lang(?name) = 'en')
                            FILTER (lang(?comment) = 'en')
                        }}'''
                    sparql.setQuery(query)

                    sparql.setReturnFormat(JSON)
                    qres = sparql.query().convert()
                    if qres['results']['bindings']:
                        result = qres['results']['bindings'][0]
                        name, comment, image_url = result['name']['value'], result['comment']['value'], result['image']['value']
                        
                        wiki_url = f'https://en.wikipedia.org/wiki/{entity}'
                                        
                        summary_entity = comment
                        wiki_knowledge_df = pd.read_html(wiki_url)[0]
                        wiki_knowledge_df = self.filter_wiki_df(wiki_knowledge_df)
                        break
                    else:
                        image_url = 'ERROR'
                        comment = 'ERROR'
                        summary_entity = 'ERROR'
                        wiki_url = f'https://en.wikipedia.org/wiki/{entity}'
                                        
                        summary_entity = comment
                        try:
                            wiki_knowledge_df = pd.read_html(wiki_url)[0]
                            wiki_knowledge_df = self.filter_wiki_df(wiki_knowledge_df)
                        except ValueError:
                            wiki_knowledge_df = 'ERROR'
            self.image_url_list.append(image_url)
            self.comment_list.append(comment)
            self.context_list.append(wiki_knowledge_df)
        self.entity_df['Image_Url'] = self.image_url_list
        self.entity_df['Summary'] = self.comment_list
        self.entity_df['Context'] = self.context_list
        return self.entity_df

# news_article = "Delhi Deputy Chief Minister Manish Sisodia today attacked the BJP leadership for its alleged attempts to buy out MLAs from Telangana Chief Minister KCR's party and demanded the arrest of Union Home Minister Amit Shah, if he is found to be involved."
def parse(news_article):
    parsed = ExtractArticleEntities(news_article)
    return parsed.json