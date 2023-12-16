# my_app.py
import folium as fl
from streamlit_folium import st_folium,  folium_static
import streamlit as st
from folium import plugins
import geopy
from geopy.geocoders import Nominatim

import json
from urllib.request import urlopen
import requests

import pandas as pd
import numpy as np

import re

import itertools



def contains(sublist, item):
    if item is None:
        return False
    for l in sublist:
        for i in item:
            if re.search(r".*{}.*".format(i.lower()), l.lower()):
                return True
    return False



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





def get_pos(lat, lng):
    return lat, lng

def make_clickable(link):
    text = link.split('===')[1]
    url =  link.split('===')[0]
    return f'<a target="_blank" href="{url}">{text}</a>'

map_screen, selector = st.columns([2,3], gap="small")

def get_map_coords():

    m = fl.Map(location=[50, 15], zoom_start=5)
    minimap = plugins.MiniMap()
    m.add_child(minimap)
    m.add_child(fl.LatLngPopup())
    map =  st_folium(m, height=500, width=700)
    data = None
    if map.get("last_clicked"):
        data = get_pos(map["last_clicked"]["lat"], map["last_clicked"]["lng"])
    return data








def get_country(data):
    
    geocoder = Nominatim(user_agent="http")
    location = geocoder.reverse(f"{round(data[0],4)},{round(data[1],4)}", language='en')
    country = location.raw.get('address').get('country')

    st.write("Selected country:", country)

    return country.lower()



@st.cache_data
def import_book_list():
    df = pd.read_json('data/downloaded.json')
    df['first_publish_year'] =  df['first_publish_year'].astype('str')
    df['last_publish_year']  =  df['last_publish_year'].astype('str')
    df['favorite']=False
    df=df.reset_index().drop('index', axis=1)
    return df



@st.cache_data
def get_book_list_for_country(df, country):
    

    
    all_names={
        'iran': ['iran','persia','persepolis', 'pasargad', 'elam', 'media'],
        'united kingdom': ['ireland','scotland','wales', 'london'],
        'pakistan': ['pakistan', 'indus', 'mohendsch', 'mohenj']
    }


    if country in all_names.keys():
        filtered_df = df[df['place_key'].fillna('').apply(contains, item=all_names[country])]
    else:
        filtered_df = df[df['place_key'].fillna('').apply(contains, item=[country])]

    filtered_df = filtered_df[['title', 'author_name', 'last_publish_year', 'subject_key', 'url','key', 'favorite']] #,'place_key'
    filtered_df['Country']=country.title()
    filtered_df = filtered_df.rename(columns={'title':'Title','author_name':'Author','last_publish_year':'Year'})

    #filtered_df['image']=

    return filtered_df







def make_selection(filtered_df):
    selected_df = st.data_editor(
        filtered_df.drop_duplicates(subset=['Title']),
        column_config={
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
        on_change=on_checkbox_klick

    )
    
    return selected_df

def on_checkbox_klick():
    selected_index_true = selected_df.index[selected_df['favorite'] == True]
    selected_index_false = selected_df.index[selected_df['favorite'] == False]


    df['favorite'].iloc[selected_index_true]=True
    df['favorite'].iloc[selected_index_false]=False

    st.session_state.df=df['favorite']
    















with map_screen:
    
    data = get_map_coords()

if data is not None:

    with selector:

        country = get_country(data)
        
        df = import_book_list()
        
        if hasattr(st.session_state, 'df'):
            df['favorite'] = st.session_state.df
    
        filtered_df = get_book_list_for_country(df, country)
        
        selected_df = make_selection(filtered_df)
        
    if hasattr(st.session_state, 'df'):
        df['favorite']=st.session_state.df
        
        df_fav=df[df['favorite']==True]
        st.write(df_fav)
      
        
        

    
   