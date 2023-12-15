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


def get_book_list_for_country(country):
    
    df = pd.read_json('data/downloaded.json')
    df['first_publish_year'] =  df['first_publish_year'].astype('str')
    df['last_publish_year']  =  df['last_publish_year'].astype('str')

    df['Country']=country.title()


    all_names={
        'iran': ['iran','persia','persepolis', 'pasargad', 'elam', 'media'],
        'united kingdom': ['ireland','scotland','wales', 'london'],
        'pakistan': ['pakistan', 'indus', 'mohendsch', 'mohenj']
    }


    if country in all_names.keys():
        filtered_df = df[df['place_key'].fillna('').apply(contains, item=all_names[country])]
    else:
        filtered_df = df[df['place_key'].fillna('').apply(contains, item=[country])]

    filtered_df = filtered_df[['title', 'author_name', 'last_publish_year', 'Country','subject_key', 'url']] #,'place_key'

    filtered_df = filtered_df.rename(columns={'title':'Title','author_name':'Author','last_publish_year':'Year'})

    filtered_df['favorite']=False

    return filtered_df


def make_selection(filtered_df):
    edited_df = st.data_editor(
        filtered_df.drop_duplicates(subset=['Title']),
        column_config={
            "favorite": st.column_config.CheckboxColumn(
                "Your favorite?",
                help="Select your **favorite** books",
                #default=False,
            ),
            "url": st.column_config.LinkColumn(),
        },
        column_order=('Title', 'Author', 'Year','url', 'favorite'),
        hide_index=True,
    )
    return edited_df



def update_session_state(full_session_history, edited_df):
    
    full_session_history.append(edited_df[edited_df['favorite']==True])
    dataframes = []
    for book in full_session_history:
        dataframes.append(book.tail(1))
    df_fav = pd.concat(dataframes).drop_duplicates(subset=['Title'])

    
    # remove books from df_fav which were un-selected (Äfavorite'==False) in edited_df
    df_fav = df_fav.loc[~df_fav['url'].isin(edited_df[edited_df['favorite'] == False]['url'])]
    df_fav=df_fav[['Title','Author', 'Year', 'Country','url', 'favorite']]

    st.session_state.df_fav = df_fav
    return st.session_state.df_fav












with map_screen:
    
    data = get_map_coords()

if data is not None:

    with selector:


        country = get_country(data)
        filtered_df = get_book_list_for_country(country)
        
        if "fav_books" not in st.session_state:
            st.session_state["fav_books"] = []


        if hasattr(st.session_state, 'df_fav'):
            df_old = st.session_state.df_fav
            #before selecting books, set those 'favorite' values to True which were already selected before
            filtered_df['favorite'].loc[filtered_df['url'].isin(df_old[df_old['favorite'] == True]['url'])]=True
            #st.write(df_old)
        
        edited_df = make_selection(filtered_df)

        
        st.session_state.df_fav = update_session_state(st.session_state["fav_books"], edited_df)




    st.write(st.session_state.df_fav)



