import time
import os
import pickle
import datetime
import pandas as pd
import traceback

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


from lib.scrapping import get_specifichtml_by_request, scrap_all_races, get_date_from_url, HTMLRecorder, get_html_by_both, SingleResultScrapper, HKJCCrawler
from lib.utils import generate_date_list

# Consider 3 scenarios: (1) local race, (2) foreign race, (3) non-existing

# Unique class
# 'localResults' for local race
# 'simulcastContainer' for foreign race
# 'errorout' for non-existing website


if __name__ == '__main__':
    url_base = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate='
    log_path = "data/request_log.txt"
    invalid_dates_path = 'data/invalid_dates.txt'
    start_date = datetime.date.today()
    date_list = generate_date_list(start_date, datetime.date(1997, 2, 12))

    # # Testing url
    # local_race_url = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2019/12/29'
    # foreign_race_url = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2020/02/20'
    # non_url = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2020/02/24'
    # horse_url = 'https://racing.hkjc.com/racing/information/English/Horse/Horse.aspx?HorseId=HK_2018_C413'
    # bug1_url = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2019/11/13'
    # selected_url = bug1_url


    crawler = HKJCCrawler(rawhtml_dir='data/intermediate_storage/stage1_raw_htmls',
                          raw_racedf_dir='data/intermediate_storage/stage2_single_race_dataframes',
                          html_records_path='data/intermediate_storage/html_records.pickle')

    invalid_list = []
    if os.path.isfile(invalid_dates_path):
        with open(invalid_dates_path, 'r') as fh:
            for line in fh:
                invalid_list.append(line.strip())


    for date in date_list:
        if date in invalid_list:  # SKIP recorded invalid links
            print('Date %s Invalid. Skipped' % date)
            continue
        selected_url = url_base + date
        try:
            crawler.scrape(selected_url)

        except TimeoutException:  # Error raised after querying the same url for > 10 times
            msg = 'Time out: %s\n'% selected_url
            print(msg)
            with open(log_path, 'a') as fh:
                fh.write(msg)

        except NoSuchElementException:  # Error after detecting the url is invalid, e.g. containing no race result
            with open(invalid_dates_path, 'a') as fh:
                fh.write(date + '\n')
            print('Invalid link')

        except Exception:  # Unknown Error

            msg = 'Unknown exception: %s' % selected_url
            with open(log_path, 'a') as fh:
                fh.write(msg + '\n')
                traceback.print_exc(file=fh)
                fh.write('\n')
