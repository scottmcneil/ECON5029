import requests
import re
import datetime
import csv
from bs4 import BeautifulSoup


########## FETCH FOMC STATEMENTS ##########

url_template = 'http://www.federalreserve.gov/newsevents/press/monetary/{0}monetary.htm'
response_list = [requests.get(url_template.format(year)) for year in range(1996,2017)]
soup_list = [BeautifulSoup(response.text,'html.parser') for response in response_list]

ul_list = [soup.find('ul',{'id':'releaseIndex'}) for soup in soup_list]
divs_list = [ul.findAll('div',{'class':'indent'}) for ul in ul_list]
link_list = [div.find('a')['href'] for divs in divs_list for div in divs if 'FOMC statement' in div.text]


def parse_fomc_page(res):
    
    content = parse_new_fomc(res) if ('/newsevents/press/monetary/' in res.url) else parse_old_fomc(res)
    return content
    
def parse_new_fomc(res):
    
    soup = BeautifulSoup(res.text)
    paragraphs = soup.find('div',{'id':'leftText'}).findAll('p',{'id':None})
    content = ' '.join([paragraph.text for paragraph in paragraphs])
    
    date_regex = r'Release Date: (?P<date>\w+ \d{1,2}, \d{4})'
    date_div  = soup.find('p',{'id':'prContentDate'})
    date_string = re.search(date_regex,date_div.text).group('date')
    date_time = datetime.datetime.strptime(date_string, '%B %d, %Y')
    
    return content, date_time
    
def parse_old_fomc(res):
    
    try:
        soup = BeautifulSoup(res.text,'html.parser')
        old_regex = r'Release Date: (?P<date>\w+ \d{1,2}, \d{4})\s+For immediate release(?P<content>[\s\S]+?)\n\n\d{4} Monetary policy\n'
        re_search = re.search(old_regex,soup.text)
        content = re_search.group('content')
    
        date_string = re_search.group('date')
        date_time = datetime.datetime.strptime(date_string, '%B %d, %Y')
        
        content_date = (content, date_time)
    except:
        content_date = False
    
    return content_date

fomc_url_template = 'http://www.federalreserve.gov/{0}'
fomc_response_list = [requests.get(fomc_url_template.format(link),'html.parser') for link in link_list]
fomc_content_date_list = [parse_fomc_page(soup) for soup in fomc_response_list if parse_fomc_page(soup)]
fomc_content_list = [content[0] for content in fomc_content_date_list]

all_state_content = "###################".join(["====".join([content[0],content[1].strftime("%Y-%m-%d")]) for content in content_date_list])
text_file = open("/Users/scott/Dropbox/School/ECON 5029-W/FOMC_Statements.txt", "w")
text_file.write(all_state_content)
text_file.close()

########## FETCH BEIGE BOOK REPORTS ##########

def parse_bb(soup):
     
    article = soup.find('section',{'class':'article-content'})
    content_strings = article.strings
    good_strings = [s for s in content_strings if parse_bb_strings(s)]
    content = "".join(good_strings)
    
    return content
    
    
def parse_bb_strings(s):
    
    try:
        p = s.parent.name == 'p'
    except:
        p = False
    
    return p

bb_url_template = 'https://www.minneapolisfed.org/news-and-events/beige-book-archive/{0}'
years = [str(year) for year in range(1996,2017)]
months = [str(mo).zfill(2) for mo in range(1,13)]
codes = ['at','bo','ch','cl','da','kc','mi','ny','ph','ri','sf','sl','su']
year_month_code = ['{0}-{1}-{2}'.format(year, month, code) for year in years for month in months for code in codes]
bb_response_list = [requests.get(bb_url_template.format(entry)) for entry in year_month_code]
 
bb_soup_list = [BeautifulSoup(response.text,'html.parser') for response in bb_response_list if response.status_code == 200]
bb_content_list = [parse_bb(soup) for soup in bb_soup_list]

all_content = "###################".join(bb_content_list)
text_file = open("/Users/scott/Dropbox/School/ECON 5029-W/BeigeBook.txt", "w")
text_file.write(all_content)
text_file.close()


article_list = [soup.find('section',{'class':'article-content'}) for soup in soup_list]
content_list = [''.join([s for s in strings if s.parent.name == 'p']) for article in article_list for strings in article.strings]

########## FETCH MONETARY POLICY STUFF ##########

initial_mp_url = 'https://www.federalreserve.gov/monetarypolicy/mpr_default.htm'
initial_mp_res = requests.get(initial_mp_url)
mp_link_list = BeautifulSoup(initial_mp_res.text, 'html.parser').findAll('a')

good_mp_link_list = [link['href'] for link in mp_link_list if link.text == 'Report']
trunc_links = [link[:29] if link[:29][-1]=='_' else link[:30] for link in good_mp_link_list]
report_sections = ['summary.htm','part1.htm','part2.htm','part3.htm','part4.htm']
full_mp_link_list = [link + section for link in trunc_links for section in report_sections]

mp_base_url = 'https://www.federalreserve.gov{0}'
mp_res_list = [requests.get(mp_base_url.format(link)) for link in full_mp_link_list]
mp_soup = [BeautifulSoup(res.text,'html.parser') for res in mp_res_list if res.status_code == 200]
mp_content = [parse_mp(soup) for soup in mp_soup]

def parse_mp(soup):
     
    content = soup.find('div',{'id':'leftText3'})
    content_strings = content.strings
    good_strings = [s for s in content_strings if parse_bb_strings(s)]
    content = "".join(good_strings)
    
    return content
    
def parse_mp_strings(s):
    
    try:
        p = s.parent.name == 'p' and s.parent['class'][0] != 'footnotes'
    except:
        p = False
    
    return p

all_mp_content = "###################".join(mp_content)
text_file = open("/Users/scott/Dropbox/School/ECON 5029-W/MonetaryPolicy.txt", "w")
text_file.write(all_mp_content)
text_file.close()


########## NORMALIZING DOCS ##########

import gensim
from collections import namedtuple

DocumentTuple = namedtuple('DocumentTuple', 'words tags')

def normalize_text(text, tag):
    norm_text = text.lower()

    # Replace breaks with spaces
    norm_text = norm_text.replace('\n', '').replace('\r','').strip()

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')
        
    tokens = gensim.utils.to_unicode(text).split()
    tags = [tag]

    return DocumentTuple(tokens, tags)
    
non_normal_docs = mp_content + bb_content_list + fomc_content_list
alldocs = [normalize_text(doc, i) for i, doc in enumerate(non_normal_docs)]
doc_list = alldocs[:]
    
########## RUNNING DOC2VEC MODEL ##########

from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

simple_models = [
    # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DBOW 
    Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
]

# speed setup by sharing results of 1st model's vocabulary scan
simple_models[0].build_vocab(alldocs)  # PV-DM/concat requires one special NULL word so it serves as template
print(simple_models[0])
for model in simple_models[1:]:
    model.reset_from(simple_models[0])
    print(model)

models_by_name = OrderedDict((str(model), model) for model in simple_models)

from random import shuffle

alpha, min_alpha, passes = (0.025, 0.001, 20)
alpha_delta = (alpha - min_alpha) / passes

# for timing
from contextlib import contextmanager
from timeit import default_timer

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

for epoch in range(passes):
    shuffle(doc_list)  # shuffling gets best results
    
    for name, train_model in models_by_name.items():
        # train
        duration = 'na'
        train_model.alpha, train_model.min_alpha = alpha, alpha
        with elapsed_timer() as elapsed:
            train_model.train(doc_list)
            duration = '%.1f' % elapsed()            

    print('completed pass %i at alpha %f' % (epoch + 1, alpha))
    alpha -= alpha_delta
    
print("END %s" % str(datetime.datetime.now()))

########## SAVING DOCVECS ##########

import pandas as pd
import numpy as np

np_dates = np.asarray([content[1].strftime("%Y-%m-%d") for content in content_date_list])

fomc_docvecs_dmc = [dv for dv in simple_models[0].docvecs][-149:]
dmc_docvec_df = pd.DataFrame(np.concatenate(fomc_docvecs_dmc).reshape(149,100))
dmc_docvec_df['Date'] = np_dates
dmc_docvec_df.to_csv('/Users/scott/GitHub/ECON5029/data/docvec_dmc.csv')


fomc_docvecs_dbow = [dv for dv in simple_models[1].docvecs][-149:]
dbow_docvec_df = pd.DataFrame(np.concatenate(fomc_docvecs_dbow).reshape(149,100))
dbow_docvec_df['Date'] = np_dates
dbow_docvec_df.to_csv('/Users/scott/GitHub/ECON5029/data/docvec_dbow.csv')

fomc_docvecs_dmm = [dv for dv in simple_models[1].docvecs][-149:]
dmm_docvec_df = pd.DataFrame(np.concatenate(fomc_docvecs_dmm).reshape(149,100))
dmm_docvec_df['Date'] = np_dates
dmm_docvec_df.to_csv('/Users/scott/GitHub/ECON5029/data/docvec_dmm.csv')
