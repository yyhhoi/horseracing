from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from lib.scrapping import get_specifichtml_by_request, scrap_horse_info, scrape_race_meta, scrape_results_table, HTMLRecorder
from bs4 import BeautifulSoup
import re
import pandas as pd
import os

class SingleResultScrapper:
    def __init__(self, race_date, race_location, html_recorder, url_base='https://racing.hkjc.com'):
        self.race_date = race_date
        self.race_location = race_location
        self.url_base = url_base
        self.html_recorder = html_recorder


    def scrape_and_formDF(self, result_html, race_place):

        # Scrap
        race_num, track_length, going, course = scrape_race_meta(result_html)
        table_dict = self._scrape_results_table(result_html)

        # Append
        num_races = len(table_dict['place'])
        table_dict['race_no'] = [race_place] * num_races
        table_dict['race_num'] = [race_num] * num_races
        table_dict['track_length'] = [track_length] * num_races
        table_dict['going'] = [going] * num_races
        table_dict['course'] = [course] * num_races
        table_dict['race_date'] = [self.race_date] * num_races
        table_dict['location'] = [self.race_location] * num_races

        # Convert to appropriate datatype
        df = pd.DataFrame(table_dict)
        return df


    def _scrape_results_table(self, html):
        table_dict = scrape_results_table(self.url_base, html, self.html_recorder)
        return table_dict


url_base = 'https://racing.hkjc.com'

# This code retrieve result table and its content from a valid website


if __name__ == '__main__':

    url = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2012/02/15'
    html, request_result = get_specifichtml_by_request(url)

    srs = SingleResultScrapper('2012/02/15', 'HV', '/data/stage2_single_race_dataframes')
    srs.scrape_and_formDF(html)