# v1
 
## The ExtractArticleEntities module provides us with the information on the type of entities in our news article along with the # of related entities for each entity.
It also provides us with the information on each entity based on the context of our Article.
 
 
Input :
news_article : (string)
 
Output :
Entity : (string)
Entity_type : (string)
Related_entities : (string)
 
Further when we select the individual entity it will provide us with the information of the entity based on the Article’s context.
 
# v2
 
## Type of entities in our news article also provides us with the information on each entity based on the context of our Article.
 
Input :
news_article : (string)
 
Output :
Entity_type : (string)
Entity : (string)
Matched Entity : (list)
Views : (integer)
Image_Url : (str)
Summary : (str)
Context: (json str)
 
Installation:
    You can install this package with 
        pip install <path to whl file>
        