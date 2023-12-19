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


import nltk
from nltk.corpus import stopwords


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from scipy.sparse import hstack

