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





m = fl.Map(location=[50, 15], zoom_start=4)

minimap = plugins.MiniMap()
m.add_child(minimap)

m.add_child(fl.LatLngPopup())


map =  st_folium(m, height=600, width=1000)


####
data = None
if map.get("last_clicked"):
    data = get_pos(map["last_clicked"]["lat"], map["last_clicked"]["lng"])

if data is not None:


    geocoder = Nominatim(user_agent="http")
    location = geocoder.reverse(f"{round(data[0],4)},{round(data[1],4)}", language='en')
    country = location.raw.get('address').get('country')


    st.write("Selected country:", country)

    country=country.lower()
    ####


    df = pd.read_json('data/downloaded.json')
    df['first_publish_year'] =  df['first_publish_year'].astype('str')
    df['last_publish_year']  =  df['last_publish_year'].astype('str')

    df['Country']=country.title()


    all_names={
        'iran': ['iran','persia','persepolis', 'pasargad', 'elam', 'media'],
        'united kingdom': ['ireland','scotland','wales', 'london'],
        'pakistan': ['pakistan', 'indus', 'mohendsch', 'mohenj']
    }

    #country='iran'
    if country in all_names.keys():
        filtered_df = df[df['place_key'].fillna('').apply(contains, item=all_names[country])]
    else:
        filtered_df = df[df['place_key'].fillna('').apply(contains, item=[country])]





    filtered_df = filtered_df[['title', 'author_name', 'last_publish_year', 'Country','subject_key', 'url']] #,'place_key'

    ##


    def make_clickable(link):
        text = link.split('===')[1]
        url =  link.split('===')[0]
        return f'<a target="_blank" href="{url}">{text}</a>'


    filtered_df = filtered_df.rename(columns={'title':'Title','author_name':'Author','last_publish_year':'Year'})


    filtered_df['favorite']=False



    if "fav_books" not in st.session_state:
        st.session_state["fav_books"] = []




    edited_df = st.data_editor(
        filtered_df.drop_duplicates(subset=['Title']),
        column_config={
            "favorite": st.column_config.CheckboxColumn(
                "Your favorite?",
                help="Select your **favorite** books",
                default=False,
            ),
            "url": st.column_config.LinkColumn(),
        },
        column_order=('Title', 'Author', 'Year','url', 'favorite'),
        hide_index=True,
    )






    st.session_state["fav_books"].append(edited_df[edited_df['favorite']==True])
    dataframes = []
    for book in st.session_state["fav_books"]:
        dataframes.append(book.tail(1))
    df_fav = pd.concat(dataframes).drop_duplicates(subset=['Title'])


    def g(df_fav, edited_df):
        df_fav = df_fav.loc[~df_fav['url'].isin(edited_df[edited_df['favorite'] == False]['url'])]
        return df_fav

    df_fav = g(df_fav.copy(), edited_df.copy())






    df_fav['Title'] = (df_fav['url']+ "===" + df_fav['Title']).apply(make_clickable)
    df_fav = df_fav.drop('url', axis=1)
    df_fav=df_fav[['Title','Author', 'Year', 'Country','subject_key']]
    df_fav = df_fav.set_index('Title')

    df_fav = df_fav.to_html(escape=False)



    st.write(df_fav, unsafe_allow_html=True)



