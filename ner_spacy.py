# -*- coding: utf-8 -*-
# library to assist with disambiguation
import disamby.preprocessors as pre
import pandas as pd
import spacy
from disamby import Disamby
# text preprocessing library
from nltk.corpus import stopwords
import string
import unicodedata
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter
from segtok.segmenter import split_single

tagger = SequenceTagger.load('ner')


def disambiguate_entities(entities):
    '''Function to disambiguate entities
    https://readthedocs.org/projects/disamby/downloads/pdf/latest/
    '''

    # create dataframe from entity dictionary
    entity_df = pd.DataFrame(columns=["entity", "text"])
    pipeline = [
        pre.normalize_whitespace,
        pre.remove_punctuation,
        lambda x: pre.trigram(x) + pre.split_words(x)

    ]
    for record in entities:
        # attempt to get a confidence score for the entity

        record_df = pd.DataFrame(record.items(), columns=["entity", "text"])
        entity_df = pd.concat([entity_df, record_df], ignore_index=True)
    print(entity_df.head())

    disambiguator = Disamby(entity_df['text'], preprocessors=pipeline)
    candidate_matches = disambiguator.disambiguated_sets(threshold=0.4)
    # create a placeholder column for matches
    entity_df['matches'] = ''
    for candidate in candidate_matches:
        # if there is a possible match then update a match column in the entity dataframe with it
        if len(candidate) >= 2:
            candidate = list(candidate)  # convert set to list for indexing
            entity_df.at[candidate[0], 'matches'] = entity_df.loc[candidate[1]].values[1]  # all matches if more than
            # one
            # print([df.loc[i].values[1] for i in candidate])
    return candidate_matches, entity_df


def preprocessing(text):
    '''Function to preprocess text'''

    # remove punctuation except -
    # text_p = "".join([char for char in text if char not in string.punctuation.replace('-', '')])

    # words = word_tokenize(text_p)

    stop_words = stopwords.words('english')
    filtered_words = [word for word in text.split() if word not in stop_words]

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


def flair_test(entity):
    splitter = SegtokSentenceSplitter()
    sentences = splitter.split(entity)
    for sentence in sentences:
        tagger.predict(sentence)
        for entity in sentence.get_spans('ner'):
            if entity.tag != 'MISC':
                confidence = entity.score
                return confidence
            else:
                confidence = 0
            return confidence


def named_entity_recognition_with_spacy(text):
    '''Function to extract entities from text using Spacy'''
    # Load the spacy model: nlp
    nlp = spacy.load('en_core_web_lg')
    # dictionary to hold entity.tag/text pairs
    article_entity_list = []
    doc = nlp(text)
    desired_entities = ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'LAW', 'LANGUAGE','NORP','DATE','GPE','TIME']

    for ent in doc.ents:

        if ent.label_ in desired_entities:
            entity_dict = {ent.label_: ent.text}

            article_entity_list.append(entity_dict)
    # dedupe the entities but only on exact match of values as occasional it will assign an ORG entity to PER

    deduplicated_entities = {frozenset(item.values()):
                                 item for item in article_entity_list}.values()

    return deduplicated_entities


if __name__ == '__main__':

    text = """After the bullet shells get counted, the blood dries and the votive candles burn out, people peer down 
    from   windows and see crime scenes gone cold: a band of yellow police tape blowing in the breeze. The South 
    Bronx, just across the Harlem River from Manhattan and once shorthand for urban dysfunction, still suffers 
    violence at levels long ago slashed in many other parts of New York City. And yet the cityâ€™s efforts to fight 
    it remain splintered, underfunded and burdened by scandal. In the 40th Precinct, at the southern tip of the 
    Bronx, as in other poor, minority neighborhoods across the country, people long hounded for   infractions are 
    crying out for more protection against grievous injury or death. By September, four of every five shootings in 
    the precinct this year were unsolved. Out of the cityâ€™s 77 precincts, the 40th has the highest murder rate but 
    the fewest detectives per violent crime, reflecting disparities in staffing that hit hardest in some 
    neighborhoods outside Manhattan, according to a New York Times analysis of Police Department data. Investigators 
    in the precinct are saddled with twice the number of cases the department recommends, even as their bosses are 
    called to Police Headquarters to answer for the sharpest crime rise in the city this year. And across the Bronx, 
    investigative resources are squeezed. It has the highest   rate of the cityâ€™s five boroughs but the thinnest 
    detective staffing. Nine of the 14   precinct detective squads for violent crime in the city are there. The 
    boroughâ€™s robbery squad is smaller than Manhattanâ€™s, even though the Bronx has had 1, 300 more cases this 
    year. And its homicide squad has one detective for every four murders, compared with one detective for roughly 
    every two murders in Upper Manhattan and more than one detective per murder in Lower Manhattan. In   lobbies and  
     family apartments, outside methadone clinics and art studios, people take note of the inequity. They hear police 
     commanders explain that they lack the resources to place a floodlight on a dangerous block or to post officers 
     at a   corner. They watch witnesses cower behind   doors, more fearful of a gunmanâ€™s crew than confident in 
     the Police Departmentâ€™s ability to protect them. So though people see a lot, they rarely testify. And in the 
     South Bronx, as in so many predominantly black and Hispanic neighborhoods like it in the United States, 
     the contract between the police and the community is in tatters. Some people have stories of crime reports that 
     were ignored, or 911 calls that went unanswered for hours. Others tell of a 911 call for help ending in the 
     callerâ€™s arrest, or of a minor charge leading to 12 hours in a fetid holding cell. This is the paradox of 
     policing in the 40th Precinct. Its neighborhoods have historically been prime targets for aggressive tactics, 
     like    that are designed to ward off disorder. But precinct detectives there have less time than anywhere else 
     in the city to answer for the blood spilled in violent crimes. Gola White, who was beside her daughter when she 
     was shot and killed in a playground this summer, four years after her son was gunned down in the same housing 
     project, ticked off the public safety resources that she said were scant in Bronx neighborhoods like hers: 
     security cameras, lights, locks, investigating police officers. â€œHere, we have nothing,â€ she said. When it 
     comes to â€œ  families,â€ she said, the authorities â€œdonâ€™t really care as much. Thatâ€™s how I feel. â€ 
     The Times has been documenting the murders logged this year in the 40th Precinct, one of a handful of 
     neighborhoods where deadly violence remains a problem in an era of   crime in New York City. The homicides  â€”  
      14 in the precinct this year, up from nine in 2015  â€”   strain detectives, and when they go unsolved, 
      as half of them have this year, some look to take the law into their own hands. From hundreds of conversations 
      with grieving relatives and friends, witnesses and police officers, the social forces that flare into murder in 
      a place like the 40th Precinct become clearer: merciless gang codes, mental illness, drugs and long memories of 
      feuds that simmered out of officersâ€™ view. The reasons some murders will never be solved also emerge: 
      paralyzing fear of retribution, victims carrying secrets to their graves and relentless casework that forces 
      detectives to move on in hopes that a break will come later. Frustrations build on all sides. Detectivesâ€™ 
      phones rarely ring with tips, and officers grow embittered with witnesses who will not cooperate. In the 
      meantime, a victimâ€™s friends conduct their own investigations, and talk of grabbing a stash gun from a wheel 
      well or a motherâ€™s apartment when they find their suspect. In the chasm between the police and the community, 
      gangs and gun violence flourish. Parents try to protect their families from drug crewsâ€™ threats, and officers 
      work to overcome the residue of years of mistrust and understaffing in communities where they still go racing 
      from one 911 call to the next. The streets around St. Maryâ€™s Park were the scene of two fatal shootings 
      logged in the 40th Precinct this year. Both are unsolved. James Fernandez heard talk of the murders through the 
      door of his   apartment on East 146th Street in a     the Betances Houses. He lived at the end of a long 
      hallway strewn with hypodermic needles, empty dope bags and discarded Hennessy bottles. A   young men who spoke 
      of being in a subset of the Bloods gang had made it their drug market, slinging marijuana and cocaine to 
      regulars, flashing firearms and blowing smoke into the Fernandez apartment. When Mr. Fernandez, 40, 
      asked the young men to move, they answered by busting up his car. This kind of crime, an anachronism in much of 
      New York, still rattles the 40th Precinct, even though murders there have fallen to 14 this year from 83 in 
      1991. It has more major felony crimes per resident than any other residential district in the city. It is also 
      one of the poorest communities in the country, and many young men find their way into underground markets. Mr. 
      Fernandez was not one to shrink from the threats. When he was growing up on the Lower East Side, he rode his 
      bicycle around to the customers of the drug dealers he worked for and collected payments in a backpack. After 
      leaving that life, he got a tech maintenance job and, three years ago, moved into the Betances Houses with his 
      wife and daughter, now 11. He had two choices to get help with the drug crew: call the police for help and risk 
      being labeled a snitch, or call his old Lower East Side bosses for muscle and risk violence. He chose the 
      police. Again and again, he walked into a local substation, Police Service Area 7, and asked for protection. 
      His daughter was using an inhaler to relieve coughs from the marijuana smoke. Mr. Fernandez and his wife got 
      terrible headaches. â€œThereâ€™s a lot of killers here, and we are going to kill you,â€ a sergeantâ€™s police 
      report quoted a    telling Mr. Fernandez in August 2015. A second report filed the same day said a    warned 
      him, â€œIâ€™m going to shoot through your window. â€ Mr. Fernandez told the police both the teenagersâ€™ 
      names, which appear in the reports, and then went home. He said one of their friends had seen him walk into the 
      substation, and they tried to intimidate him out of filing another report. Three days later, the same    
      propped his bike on their door, â€œthen said if I was to open the door and say something, they would body slam 
      me,â€ Mr. Fernandezâ€™s wife, Maria Fernandez, wrote on slips of paper she used to document the hallway ruckus 
      and the inadequate police response. The boys made comments about how easy a target she was and about how they 
      would have to â€œslapâ€ her if she opened the door while they made a drug sale, and they threatened to beat 
      the Fernandez family because â€œthey are the ones snitching,â€ her notes say. But another   complaint at the 
      substation, 10 days after the first, brought no relief. A week later, feeling desperate, Ms. Fernandez tried 
      calling: first to the substation, at 8:50 p. m. when one of the boys blew weed smoke at her door and made a   
      threat to attack her, and then to 911 at 10:36 p. m. The police never came, she wrote in her notes. She tried 
      the 40th Precinct station house next, but officers at the desk left her standing in the public waiting area for 
      a   she said, making her fear being seen again. Officers put her in worse danger some months later, she said, 
      when they came to her door and announced in front of the teenagers that they were there on a complaint about 
      drug activity. Mr. Fernandez started doing the work that he said the police had failed to do. He wired a camera 
      into his peephole to record the drugs and guns. The footage hark back to the New York of the 1980s, still very 
      much present to some of the precinctâ€™s residents. Around 6:30 each morning, Sgt. Michael J. LoPuzzo walks 
      through the tall wooden doors of the 40th Precinct station house. The cases that land on his metal desk  â€”   
      dead bodies with no known cause, strip club brawls, shooting victims hobbling into the hospital themselves  â€” 
        bring resistance at every turn, reminding him of an earlier era in the cityâ€™s   campaign. â€œI havenâ€™t 
        got one single phone call thatâ€™s putting me in the right direction here,â€ said Sergeant LoPuzzo, 
        the head of the precinctâ€™s detective squad, one day this summer as he worked on an answer to an email 
        inquiry from a murder victimâ€™s aunt about why the killer had not been caught. â€œAnd people just donâ€™t 
        understand that. â€ Often it is detectives who most feel the effects of people turning on the police. 
        Witnesses shout them away from their doors just so neighbors know they refuse to talk. Of the 184 people who 
        were shot and wounded in the Bronx through early September, more than a third  â€”   66 victims  â€”   
        refused to cooperate. Over the same period in the 40th Precinct, squad detectives closed three of 17 nonfatal 
        shootings, and 72 of 343 robbery cases. Part of the resistance stems from   preventive policing tactics, 
        like    that were a hallmark of the     style under former Mayor Michael R. Bloomberg and his police 
        commissioner, Raymond W. Kelly. Near the height of the    strategy, in 2012, the 40th Precinct had the   
        stops in the city, the   stops in which officers used force and the most frisks. Of 18, 276 stops that year, 
        15, 521 were of people who had done nothing criminal. The precinct was also one of the   areas that the 
        department flooded with its newest officers. At roll calls, they were pressured to generate numbers: write 
        tickets and make arrests. They had no choice but to give a summons to a young man playing in a park after 
        dark, even if the officers had done the same growing up in the same neighborhood. â€œI need to bring 
        something in today to justify my existence,â€ Officer Argenis Rosado, who joined the precinct in 2010, 
        said in an interview at the station house. â€œSo now youâ€™re in a small area, and day after day youâ€™re 
        hammering the same community. Of course that communityâ€™s eventually going to turn on you. â€ The pressure 
        warped the way officers and residents saw each other. Rookies had to ignore why someone might be drinking 
        outside or sitting on a stoop. â€œSome of the cops that came out at that time probably viewed the community 
        differently, too,â€ said Hector Espada, a   veteran of the precinct. â€œNot because they wanted to, 
        but because they had to. Because some way or somehow, you canâ€™t give someone a $115 summons and feel like 
        you guys could still have a civil conversation after that. â€ Morale wilted in the aged station house on 
        Alexander Avenue, in Mott Haven. Officers felt pressure to downgrade crime complaints to make them appear 
        less serious. Several said in interviews that they had overlooked crime reports from immigrants because they 
        were seen as unlikely to complain, and watched supervisors badger victims into repeating their stories in 
        hopes that they would drop their complaints. The practice of downgrading complaints resulted in the 
        disciplining of 19 officers in the precinct last year, one in a string of scandals that has left officers 
        there feeling overscrutinized for problems that also existed elsewhere. Four commanders in the precinct were 
        sent packing in five years, one of them after officers were found to be â€œticket fixing,â€ or forgiving 
        parking tickets for friends, and another after he was recorded giving guidance on whom to stop and frisk: 
        black boys and men, ages 14 to 21. Some officers fled to other commands. Others became reluctant to take 
        assignments in proactive policing units, like   that put them in   situations on the street. â€œWhenever I 
        walked through the doors of the precinct, to me, it seemed like a black cloud,â€ said Russell Lewis, 
        a    of the 40th. â€œIt was like a heaviness. When you walked in, all you wanted to do was do your 8 hours 35 
        minutes and go home, because you didnâ€™t want to get caught up in anything. â€ The precinct covers only 
        about two square miles, but the more than a dozen housing projects there mean that it overflows with people. 
        Methadone clinics draw addicts from around the city.   lofts on the southern edge of the precinct presage a 
        wave of gentrification. Even as the Police Department has hired 1, 300 more officers for neighborhood 
        policing and counterterrorism, officers in the 40th Precinct said they could still rush to 25 911 calls 
        during a shift  â€”   a number unchanged from what the new police commissioner, James P. Oâ€™Neill, 
        said he was handling in a similar South Bronx precinct 15 years ago. Several dozen calls at a time can be 
        waiting for a response. Residents know that if you want the police for a domestic problem, it helps to hint 
        that there is a weapon. Last year, the precinct drew the   number of civilian complaints for officer 
        misconduct in the city, and the most lawsuits stemming from police actions. The precinct is trying to improve 
        morale under a new commanding officer, Deputy Inspector Brian Hennessy. A cadre of what the department calls 
        neighborhood coordination officers has been on patrol since last January, part of a citywide effort under Mr. 
        Oâ€™Neill and Mayor Bill de Blasio to bring back the beat cop, unencumbered by chasing every last 911 call, 
        who can listen to peopleâ€™s concerns and help with investigations. The precinct has made among the most gun 
        arrests in the city, and officers said they now had more discretion to resolve encounters without a summons 
        or an arrest. At one corner near a school, on Courtlandt Avenue and East 151st Street, that has long spawned 
        complaints about gunfire and fights, Inspector Hennessy and some of his officers painted over graffiti and 
        swept up drug paraphernalia this summer. People said it was the first answer to their complaints in years. 
        But the inspector acknowledged that the residue of   policing lingers. â€œThat perception really sticks,
        â€ he said. The workload in the 40th Precinct is startling and reveals a gap in how detective squads are 
        equipped to answer violent crime in Manhattan compared with the Bronx, Brooklyn and Queens. Three of the 
        precinctâ€™s 16 detectives are carrying more than 400 cases each this year, and many others have loads in the 
        high 300s, even though the department advises 150 in violent precincts. When they are assigned a homicide, 
        they typically have four days to investigate before dealing with other cases. Quieter precincts can give 
        detectives a month with little distraction to investigate a murder. Detectives in the 40th Precinct have each 
        handled an average of 79 violent felonies this year through    â€”   murders, rapes, felony assaults and 
        robberies. By contrast, a detective in the precinct on the southern end of Staten Island carries nine such 
        cases a detective in the precinct patrolling Union Square and Gramercy Park handles 16 and a detective in the 
        precinct for most of Washington Heights handles 32, the citywide median. Last year, the 40th was the    for 
        violent crime, with 65 cases per detective. In the Bronx as a whole, a precinct detective has carried an 
        average of 58 violent felonies this year, compared with 27 in Manhattan, 37 in Brooklyn, 38 in Queens and 25 
        on Staten Island. Rape cases and robbery patterns are later sent to more specialized units, but precinct 
        detectives do extensive initial work to interview victims, write reports and process evidence. Precincts in 
        much of Manhattan, which are whiter and wealthier than the South Bronx, often have more property felonies, 
        like stolen laptops or credit cards, and the police say those can be complex. But even accounting for those 
        crimes, the 40th Precinct has some of the heaviest caseloads of overall crime per detective in the city. 
        Michael Palladino, the head of the Detectivesâ€™ Endowment Association and a former Bronx officer, 
        said staffing disparities affected the departmentâ€™s efforts to build trust in communities like the South 
        Bronx. Witnesses make a calculation, he said: â€œIf I cooperate with the detectives, thereâ€™s so much work, 
        thereâ€™s so few of them there, they wonâ€™t even get the chance to protect me, or theyâ€™ll be there too 
        late when the retaliation comes. â€ Sergeant LoPuzzo, who turned down a more prestigious post to stay in the 
        40th Precinct, said that his squad worked tirelessly to handle cases with the people he had, and that while 
        every squad wanted more detectives, staffing needs for counterterrorism units and task forces had created new 
        deployment challenges across the department. â€œWe fight with the army we have, not the army we wish we had,
        â€ he said. Details of how the Police Department assigns its 36, 000 officers are closely held and 
        constantly in flux, and the public has minimal information on how personnel are allocated. Presented with The 
        Timesâ€™s analysis of confidential staffing data, the departmentâ€™s chief of detectives, Robert K. Boyce, 
        vowed to send more detectives to the 40th Precinct and said the department would reassess its deployment more 
        broadly in troubled precincts. He said a recent decision to bring gang, narcotics and vice detectives under 
        his command made it easier to shift personnel. Chief Boyce said the burdens on detectives went beyond felony 
        crimes to include   and   cases. And he noted the support that precinct squads got from centralized units 
        focusing on robberies, gangs or grand larcenies, for example. Major crime keeps pounding the 40th Precinct, 
        at rates that in 2015 were only a tenth of a percent lower than in 2001, even as citywide crime dropped by 
        more than a third over the same period. But the precinctâ€™s detective squad shrank by about eight 
        investigators during those years, according to staffing data obtained from the City Council through a Freedom 
        of Information Law request. The squad covering Union Square and Gramercy Park, where crime dropped by a third 
        over that period, grew by about 11 investigators. (The 40th Precinct was given an additional detective and 
        four   investigators this summer, when it was already missing three detectives for illness or other reasons.) 
        Retired detectives are skeptical that community relations alone can drive down crime in the cityâ€™s last 
        â€œâ€ the busiest precincts. Rather, they say, the Police Department should be dedicating more resources to 
        providing the same sort of robust investigative response that seems standard in Manhattan. â€œAny crime in 
        Manhattan has to be solved,â€ said Howard Landesberg, who was a 40th Precinct detective in the late 1980s. 
        â€œThe outer boroughs are, like, forgotten. â€ Retired detectives said that understaffing made it harder to 
        solve crimes in the Bronx, Brooklyn and Queens, where the higher prevalence of gang and drug killings already 
        saddled investigators with cases in which people were not inclined to cooperate. Through   detectives had 
        closed 67 percent of homicides in Manhattan and 76 percent of those in Staten Island this year, compared with 
        54 percent of those in the Bronx, 42 percent of those in Queens and 31 percent of those in Brooklyn. Of last 
        yearâ€™s homicides, detectives cleared 71 percent in Manhattan, 63 percent in the Bronx, 62 percent in 
        Queens, 57 percent in Staten Island and 31 percent in Brooklyn. â€œItâ€™s the culture of the Police 
        Department that they worry about Manhattan,â€ said Joseph L. Giacalone, a former sergeant in the Bronx Cold 
        Case Squad, in part â€œbecause thatâ€™s where the money is. â€ He added: â€œWhen de Blasio came in, 
        he talked about the tale of two cities. And then heâ€™s done the complete opposite of what he said. Itâ€™s 
        just business as usual. â€ The Bronxâ€™s struggles extend into prosecutions. In each of the last five years, 
        prosecutors in the Bronx have declined to prosecute violent felony cases more than anywhere else in the city. 
        And the rate of conviction in the Bronx is routinely the lowest in the city as well, but has ticked up this 
        year to surpass Brooklynâ€™s rate through November as Bronx prosecutors work to streamline cases. Some cases 
        have become even more difficult to win because of the   problem in the 40th Precinct, which has allowed 
        defense lawyers to attack the credibility of officers who were implicated, said Patrice Oâ€™Shaughnessy, 
        a spokeswoman for the Bronx District Attorneyâ€™s office. The district attorney, Darcel D. Clark, elected in 
        2015, said in a statement, â€œI was a judge here in the Bronx, and I heard from jurors that they canâ€™t be 
        impartial because they donâ€™t trust the police. â€ Against that tide of mistrust, Sergeant LoPuzzoâ€™s 
        detectives work 36 hours straight on some fresh cases. They buy Chinese takeout with their own money for a 
        murder suspect. They carry surveillance videos home in hopes that their personal computers may enhance them 
        better than a squad computer. They buy an urn for a homeless mother who has her murdered sonâ€™s ashes in a 
        box. In the months after a killing, they can seem like the only people in this glittering city who are paying 
        attention to the 40th Precinctâ€™s homicide victims. Newly fatherless children go back to school without a 
        therapistâ€™s help. Victimsâ€™ families wander confused through a courthouse and nearly miss an appearance. 
        Newspapers largely ignore killings of people with criminal pasts, pushing them down the priority lists of the 
          chiefs at Police Headquarters. In a stuffy   squad room, the detectives of the 40th Precinct grapple with 
          an inheritance of government neglect. They meet mothers who believe their sons might never have been 
          murdered had a city guidance counselor listened to pleas to help them stay enrolled, or had a city housing 
          worker fixed the locks or lights on a building. And the detectives work alongside a vicious system on the 
          streets for punishing police cooperators. Young men scan court paperwork in prison, looking for the names 
          of people who turned on them. One murder victim in the precinct this year was cast out of his crew after he 
          avoided being arrested with them in a gang takedown some believed he was cooperating. A longtime 40th 
          Precinct detective, Jeff Meenagh, said a witness in a homicide case was going to testify until he went back 
          to his neighborhood and was told that anyone who testified would â€œget what you deserve. â€ The allies 
          Sergeant LoPuzzo makes are friendly only for so long. He helped clear a womanâ€™s son of a robbery charge 
          by locating surveillance video that proved he was not the robber. The mother started calling with tips 
          under a code name  â€”   about a gun under a car, for example. But she always refused to testify. And she 
          cut ties this year after Sergeant LoPuzzo arrested her son in the stabbing of two people and her   in a 
          shooting. New York City owns East 146th Street and the   buildings on each side. But James Fernandez, 
          in the Betances Houses, said the reality on the ground was different: The drug boss ran the block. By 
          October, Mr. Fernandez was increasingly afraid  â€”   and fed up. Mr. Fernandez and his wife went so far as 
          to give officers keys to the building door, so they could get in whenever they wanted, showed them the 
          videos and offered them   access to his camera so they could see what was happening in the hallway. A 
          couple of officers said they needed a supervisorâ€™s permission to do more. Others answered that the young 
          men were only making threats. Officers occasionally stopped outside their building, causing the young men 
          to scatter, but did not come inside, Mr. Fernandez said. The menacing worsened. Mr. Fernandezâ€™s daughter 
          was harassed as she arrived home from school. She grew more and more distressed, and her parents had her 
          start seeing a therapist. Mr. Fernandez made several complaints at the office of the borough president, 
          Ruben Diaz Jr. and visited a victimâ€™s advocate in the district attorneyâ€™s office. On Oct. 20, 2015, 
          he sent an online note to the police commissionerâ€™s office. â€œWe went to all proper channels for help,
          â€ the note said. â€œBoth precincts failed us, except 2 officers who helped us, but their hands are tied. 
          No one else to turn to. I have months of video of multiple crimes taking place and we are in extreme 
          danger. â€ â€œ40th and PSA 7 wonâ€™t do anything,â€ he wrote, referring to the local substation. 
          â€œPlease we need to speak to some one with authority. â€ The local substation commander, Deputy Inspector 
          Jerry Oâ€™Sullivan, and the Bronx narcotics unit were alerted to the complaints. But Mr. Fernandez said he 
          never heard from them. So he relied on his own street instincts to protect his family. He made pleas to a 
          man he thought was employing the dealers in the hallway. The activity quieted briefly, but it returned 
          after the young men rented a room in a womanâ€™s apartment upstairs. Mr. Fernandez approached a different 
          man who he learned was the boss of the operation. The man agreed to ask the dealers to calm down. He even 
          hired a drug customer to sweep the hallway, Mr. Fernandez said. But two weeks later, the dealing and the 
          harassment resumed. So he went to his old Lower East Side bosses, who hired men to trail his wife and 
          daughter on their way out of the building and make sure they made it safely to school. At other times they 
          sat outside the Betances Houses. He also bought two bulletproof vests, for about $700 each. He could not 
          find one small enough for his daughter. â€œI have no faith in the City of New York, I have no faith in the 
          police, I have no faith in the politicians,â€ Mr. Fernandez said. â€œThe only thing I know for sure: God, 
          if weâ€™re in a situation again, I will be left to defend my family. â€ Paying such close attention to 
          what was happening in the hallway, Mr. Fernandez said he learned some details about two recent homicides 
          that the 40th Precinct was investigating. But because his calls for help were going nowhere, he said he 
          decided not to put himself in greater risk by talking: He would not tell the police what he had learned. 
          â€œIâ€™m bending over backward, and nobodyâ€™s not even doing anything,â€ he said. â€œWhy am I going to 
          help you, if you ainâ€™t going to help me?â€ By last January, a new neighborhood coordination officer was 
          working with residents of the Betances Houses, and ended up with the most arrests in his housing command, 
          Inspector Oâ€™Sullivan said. Chief Boyce said that the silos in which gang and narcotics detectives used to 
          work made responding to complaints more difficult, but that the recent restructuring would remove those 
          obstacles. â€œNo one should live like Mr. Fernandez lived, with people dealing drugs outside of his 
          apartment,â€ he said. Mr. Fernandezâ€™s complaints did not spur any arrests, but two men from the hallway 
          were caught separately this year in shootings. One of them, whom Mr. Fernandez named in a police report, 
          was charged this summer with hitting an officer with a metal folding chair and firing three gunshots into a 
          crowd, court papers say. He is being held on Rikers Island on an attempted murder charge. That was too late 
          for Mr. Fernandez. By May, he had moved his family away. """
    text = preprocessing(text)
    # print(text)
    named_entities = named_entity_recognition_with_spacy(text)

    candidates, df = disambiguate_entities(named_entities)
    print(df.head(len(df)))
    for i, row in df.iterrows():
        df.at[i, 'confidence'] = flair_test(row['text'])
    print(df.head(len(df)))
    df.to_csv('article_output.csv',index="false")
