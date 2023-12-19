# my_app.py
import folium as fl
from streamlit_folium import st_folium,  folium_static
import streamlit as st
from folium import plugins
import geopy
from geopy.geocoders import Nominatim

import json

                
import pandas as pd
import numpy as np

import re

import itertools


import nltk
from nltk.corpus import stopwords


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from scipy.sparse import hstack



nltk.download('stopwords')
# download list of stopwords from nltk lib.
stop_words = set(stopwords.words('english'))

st.set_page_config(layout='wide',page_title = "Map")
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)



@st.cache_data
def import_book_list():
    df = pd.read_json('data/downloaded.json')
    df['first_publish_year'] =  df['first_publish_year'].astype('str')
    df['last_publish_year']  =  df['last_publish_year'].astype('str')
    df['favorite']=False
    df=df.reset_index().drop('index', axis=1)
    return df






class helper_functions():

    def contains(self,sublist, item):
        if item is None:
            return False
        for l in sublist:
            for i in item:
                if re.search(r".*{}.*".format(i.lower()), l.lower()):
                    return True
        return False


    def collapse_column(self,column):
        return 1 if any(column) else 0









class geo_functions():
    
    def get_pos(self, lat, lng):
        return lat, lng


    def get_map_coords(self):

        m = fl.Map(location=[50, 15], zoom_start=5)
        minimap = plugins.MiniMap()
        m.add_child(minimap)
        m.add_child(fl.LatLngPopup())
        map =  st_folium(m, height=500, width=700)
        data = None
        if map.get("last_clicked"):
            data = gf.get_pos(map["last_clicked"]["lat"], map["last_clicked"]["lng"])
        return data


    def get_country(self,data):

        geocoder = Nominatim(user_agent="http")
        location = geocoder.reverse(f"{round(data[0],4)},{round(data[1],4)}", language='en')
        country = location.raw.get('address').get('country')

        st.write("Selected country:", country)

        return country.lower()

    @st.cache_data
    def compile_ancient_country_names(_self):
        all_names_lists = {
            'iran': ['iran', 'persia', 'persepolis', 'pasargad', 'elam', 'media'],
            'iraq': ['iraq', 'mesopotamia', 'sumer', 'akkad', 'babylon', 'assyr', 'parthi', 'sassanian'],
            'united kingdom': ['england','ireland', 'scotland', 'wales', 'london'],
            'pakistan': ['pakistan', 'indus', 'mohendsch', 'mohenj'],
            'greece': ['greece', 'achaea', 'aeolis', 'arcadia', 'boeotia', 'chalcidice', 'crete', 'cyprus', 'cyzicus', 'delphi', 'dodona', 'euboea', 'epirus', 'etolia', 'heracleia', 'ionia', 'laconia', 'lesbos', 'lydia', 'macedonia', 'megaris', 'messinia', 'mycenae', 'olbia', 'peloponnese', 'phocis', 'phoenicia', 'thebes', 'thessaly', 'crete'],
            'albania': ['albania', 'dardania', 'ancient epirus'],
            'algeria': ['algeria', 'numidia', 'roman province of mauretania'],
            'angola': ['angola', 'kingdom of kongo'],
            'armenia': ['armenia', 'urartu', 'arsacid empire'],
            'austria': ['austria', 'ostmark', 'roman province of noricum'],
            'belarus': ['belarus', 'white rus', 'slavic settlements'],
            'belgium': ['belgium', 'belgium', 'roman province of gallia belgica', 'habsburg netherlands'],
            'bosnia and herzegovina': ['bosnia and herzegovina', 'bosna', 'hum'],
            'bulgaria': ['bulgaria', 'thrace', 'odysian kingdom'],
            'croatia': ['croatia', 'panonia', 'illyria'],
            'cyprus': ['cyprus', 'cypriot civilization', 'minoan settlements'],
            'czechia': ['czech republic', 'czech lands', 'great moravian empire', 'bohemian kingdom'],
            'denmark': ['denmark', 'denmark', 'vikings', 'viking age'],
            'finland': ['finland', 'finland', 'samoyede'],
            'france': ['france', 'gaul', 'celtic tribes', 'roman province of gaul'],
            'georgia': ['georgia', 'iberia', 'colchis'],
            'germany': ['germany', 'teutonic tribes', 'holy roman empire'], 
            'hungary': ['hungary', 'hungary', 'avar khaganate'],
            'iceland': ['iceland', 'norse settlers', 'viking age'],
            'ireland': ['ireland', 'ireland', 'celtic tribes'],
            'italy': ['italy', 'latium', 'etruria', 'ausonia', 'enotria', 'roma', 'rome'],
            'kazakhstan': ['kazakhstan', 'saka tribes', 'khazar khanate'],
            'kosovo': ['kosovo', 'kosovo', 'serbian empire'],
            'latvia': ['latvia', 'latvia', 'baltic tribes'],
            'lithuania': ['lithuania', 'lithuania', 'baltic tribes'],
            'luxembourg': ['luxembourg', 'grand duchy of luxembourg'],
            'macedonia': ['macedonia', 'aegae', 'eordaia', 'upper macedonia', 'chalcidice'],
            'north macedonia': ['macedonia', 'aegae', 'eordaia', 'upper macedonia', 'chalcidice'],
            'moldova': ['moldova', 'dacia', 'roman province of dacia'],
            'morocco': ['morocco', 'berber kingdoms'],
            'netherlands': ['netherlands', 'low countries', 'frankish empire', 'dutch republic'],
            'poland': ['poland', 'vistula river trade routes', 'polish-lithuanian commonwealth'],
            'portugal': ['portugal', 'lusitanian tribes', 'roman province of lusitania'],
            'romania': ['romania', 'romania', 'dacia', 'roman province of dacia'],
            'russia': ['russia', 'scythia', 'sarmatians', 'khazar khanate'],
            'serbia': ['serbia', 'serbia', 'serbian empire'],
            'slovakia': ['slovakia', 'slovak lands', 'principality of nitra', 'great moravian empire'],
            'slovenia': ['slovenia', 'slovenian lands', 'slovenia'],
            'spain': ['spain', 'hispania', 'iberian peninsula', 'tartessian civilization', 'numidians', 'roman province of hispanial'],
            'switzerland': ['switzerland', 'helvetian confederacy', 'roman province of helvetia'],
            'turkey': ['turkey', 'anatolia', 'hittite empire', 'hattu', 'phrygian kingdom', 'phrygia', 'ancient greek colonies'],
            'ukraine': ['ukraine', 'kievan rus']
            }

        all_names_string={}
        for k,i in all_names_lists.items():
            all_names_string.update({k:' '.join(all_names_lists[k])})

        return all_names_lists, all_names_string










class selector():

    @st.cache_data
    def get_book_list_for_country(_self,df, country):


        all_names, _ = gf.compile_ancient_country_names()

        if country in all_names.keys():
            filtered_df = df[df['place_key'].fillna('').apply(hf.contains, item=all_names[country])]
        else:
            filtered_df = df[df['place_key'].fillna('').apply(hf.contains, item=[country])]

        filtered_df = filtered_df[['title', 'author_name', 'last_publish_year', 'subject_key', 'url','key', 'favorite']] #,'place_key'
        filtered_df['Country']=country.title()
        filtered_df = filtered_df.rename(columns={'title':'Title','author_name':'Author','last_publish_year':'Year'})

        #filtered_df['image']=

        return filtered_df







    def make_selection(self,filtered_df):
        selected_df = st.data_editor(
            filtered_df.drop_duplicates(subset=['Title']),
            column_config={
                "Title": st.column_config.Column(width='medium'),
                "favorite": st.column_config.CheckboxColumn(
                    "Your favorite?",
                    help="Select your **favorite** books",
                    #default=False,
                ),
                #'image': st.column_config.ImageColumn(label=None, *, width=None, help=None)
                "url": st.column_config.LinkColumn(),
            },
            column_order=('Title', 'Author', 'Year','url', 'favorite'),
            hide_index=True,
            on_change=sel.on_checkbox_klick

        )

        return selected_df

    def on_checkbox_klick(self):
        selected_index_true = selected_df.index[selected_df['favorite'] == True]
        selected_index_false = selected_df.index[selected_df['favorite'] == False]


        df['favorite'].iloc[selected_index_true]=True
        df['favorite'].iloc[selected_index_false]=False

        st.session_state.df=df['favorite']




class word_preprocessor():

    def convert_to_string(self,row, column):
        if isinstance(row[column], list):
            # If the value is a list, join the strings using a comma
            subject_key_str = ' '.join(row[column])
        else:
            # If the value is a string, just return the string itself
            subject_key_str = row[column]
        return subject_key_str


    def nlp_preprocessing(self,total_text, index, column, dataframe):
        if type(total_text) is str:
            string = ""
            for words in total_text.split():
                # remove the special chars like '"#$@!%^&*()_+-~?>< etc.
                word = ("".join(e for e in words if e.isalnum()))
                # Convert all letters to lower-case
                word = word.lower()
                # stop-word removal
                if not word in stop_words:
                    string += word + " "
            dataframe[column][index] = string
        else:
            dataframe[column][index] = ""



    def preprocess_keywords(self,df0):
        df_all = df0.copy()
        _, all_names = gf.compile_ancient_country_names()
        # Apply convert_to_string function to each row of the 'subject_key' column
        df_all['subject_key'] = df_all.apply(wp.convert_to_string, args=('subject_key',), axis=1)
        df_all['place_key'] = df_all.apply(wp.convert_to_string, args=('place_key',), axis=1)
        df_all['person_key'] = df_all.apply(wp.convert_to_string, args=('person_key',), axis=1)

        for index, row in df_all.iterrows():
            wp.nlp_preprocessing(row['subject_key'], index, 'subject_key',df_all)
        for index, row in df_all.iterrows():
            wp.nlp_preprocessing(row['place_key'], index, 'place_key',df_all)
        for index, row in df_all.iterrows():
            wp.nlp_preprocessing(row['person_key'], index, 'person_key',df_all)


        df_all['place_key']=df_all['place_key'].replace(all_names, regex=True)



        return df_all

    
  
    
class recommender():
    
    def fit_transform_word_vectors(self,df_all, df_fav):
        subject_key_vectorizer = CountVectorizer()
        subject_key_features   = subject_key_vectorizer.fit_transform(df_all['subject_key'])


        place_key_vectorizer = CountVectorizer()
        place_key_features   = place_key_vectorizer.fit_transform(df_all['place_key'])


        person_key_vectorizer = CountVectorizer()
        person_key_features   = person_key_vectorizer.fit_transform(df_all['person_key'])

        all_features_df = hstack((subject_key_features, place_key_features,person_key_features)).tocsr()



        subject_key_features   = subject_key_vectorizer.transform(df_fav['subject_key'])
        place_key_features   = place_key_vectorizer.transform(df_fav['place_key'])
        person_key_features   = person_key_vectorizer.transform(df_fav['person_key'])

        all_features_df_fav = hstack((subject_key_features, place_key_features,person_key_features)).tocsr()

        return all_features_df, all_features_df_fav




    def wvec_recommender(self,all_features_df, all_features_df_fav,num_results):

        # Collapse each column into a matrix of a single row
        collapsed_matrix = np.array([[hf.collapse_column(i) for i in all_features_df_fav.toarray().T]])

        selection = collapsed_matrix

        pairwise_dist = pairwise_distances(all_features_df,selection)

        indices = np.argsort(pairwise_dist.flatten())[0:num_results]

        df_indices = list(df_all.index[indices])

        return df_all.loc[df_indices]




if __name__ == "__main__":

    hf=helper_functions()
    gf = geo_functions()
    sel=selector()
    wp = word_preprocessor()  
    rec = recommender()    

    df = import_book_list()
    st.markdown("## Please select a country and click (always twice) on your favorite books about the history of this country.")

    map_screen_column, selection_column = st.columns([2,3], gap="small")


    with map_screen_column:

        data = gf.get_map_coords()

    if data is not None:

        with selection_column:

            country = gf.get_country(data)

            if hasattr(st.session_state, 'df'):
                df['favorite'] = st.session_state.df

            filtered_df = sel.get_book_list_for_country(df, country)

            selected_df = sel.make_selection(filtered_df)

        #if hasattr(st.session_state, 'df'):
        st.markdown("### Selected books:")
        df_fav=df[df['favorite']==True]
        df_fav=df_fav.rename(columns={'title':'Title', 'author_name':'Author', 'last_publish_year':'Year', 'place':'Country', 'person':'Person(s)', 'isbn':'ISBN'})
        st.dataframe(df_fav[['Title', 'Author', 'Year', 'Country','Person(s)', 'url', 'ISBN']], hide_index=True, use_container_width=True)

        st.session_state.final_df_fav=df_fav





    if st.button('Get recommendations', type="primary"):



        if hasattr(st.session_state, 'final_df_fav'):


            df_fav=st.session_state.final_df_fav
            
            if df_fav.empty:
                st.error('Please select your favorite books.')
                
            else:
            
                df_all=df[~df.index.isin(df_fav.index)]


                df_fav=wp.preprocess_keywords(df_fav)
                df_all=wp.preprocess_keywords(df_all)


                all_features_df, all_features_df_fav = rec.fit_transform_word_vectors(df_all, df_fav)


                df_recommended = rec.wvec_recommender(all_features_df, all_features_df_fav,num_results=10)
                df_recommended = df_recommended[['title', 'author_name', 'last_publish_year', 'place','person', 'url', 'isbn']]
                df_recommended = df_recommended.rename(columns={'title':'Title', 'author_name':'Author', 'last_publish_year':'Year', 'place':'Country', 'person':'Person(s)', 'isbn':'ISBN'})

                


                df_recommended['image_link']='https://covers.openlibrary.org/b/isbn/'+df_recommended['ISBN']+'-M.jpg'


                st.dataframe(
                df_recommended,
                column_config={
                    "image_link": st.column_config.ImageColumn(
                        "Preview Cover", help="Click on book cover to enlarge"
                    )
                },
                hide_index=True,
                use_container_width=True
                )