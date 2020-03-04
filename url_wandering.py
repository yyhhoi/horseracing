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


from lib.scrapping import get_specifichtml_by_request, scrap_all_races, get_date_from_url, HTMLRecorder, get_html_by_both, SingleResultScrapper
from lib.utils import generate_date_list

# Consider 3 scenarios: (1) local race, (2) foreign race, (3) non-existing

# Unique class
# 'localResults' for local race
# 'simulcastContainer' for foreign race
# 'errorout' for non-existing website



class HKJCCrawler:
    def __init__(self, rawhtml_dir='data/intermediate_storage/stage1_raw_htmls',
                 raw_racedf_dir='data/intermediate_storage/stage2_single_race_dataframes',
                 html_records_path='data/intermediate_storage/html_records.pickle',
                 verbose=0):
        self.rawhtml_dir = rawhtml_dir
        self.raw_racedf_dir = raw_racedf_dir
        self.verbose = verbose
        self.html_recorder = HTMLRecorder(results_dir=rawhtml_dir, html_records_path=html_records_path)



    def scrap(self, url, skip_existed=True):
        """

        Args: url (str): Url ending with the race date. E.g.
        https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2019/12/29

        Returns:

        """
        print('Main date: ', url)
        date = get_date_from_url(url)
        save_path = os.path.join(self.raw_racedf_dir, '%s.csv' % (date.replace('/', '-')))
        if skip_existed:
            if os.path.isfile(save_path):
                print('Date %s existed, skipped.' % (date))
                return None

        result_main_html = get_html_by_both(self.html_recorder, url, selected_page='result', max_trial=10)

        all_races_links, (location, max_num_races) = scrap_all_races(result_main_html, url)
        single_scrapper = SingleResultScrapper(race_date=date, race_location=location, html_recorder=self.html_recorder)

        df_list = []
        for linkid, race_link in enumerate(all_races_links):
            print('Sub-race: ', race_link)
            results_html = get_html_by_both(self.html_recorder, race_link, selected_page='result', max_trial=10)
            df = single_scrapper.scrape_and_formDF(results_html, linkid+1)
            df_list.append(df)
        df_all = pd.concat(df_list, axis=0)
        df_all.to_csv(save_path)







if __name__ == '__main__':
    url_base = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate='
    log_path = "data/request_log.txt"
    invalid_dates_path = 'data/invalid_dates.txt'
    start_date = datetime.date.today()
    # start_date = datetime.date(2010, 6, 1)
    date_list = generate_date_list(start_date, datetime.date(2000, 1, 1))


    # # Testing url
    # local_race_url = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2019/12/29'
    # foreign_race_url = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2020/02/20'
    # non_url = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2020/02/24'
    # horse_url = 'https://racing.hkjc.com/racing/information/English/Horse/Horse.aspx?HorseId=HK_2018_C413'
    # bug1_url = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2019/11/13'
    # selected_url = bug1_url
    # html = get_specifichtml_by_request(url=selected_url, selected_page='result')
    # print(html)

    crawler = HKJCCrawler(rawhtml_dir='data/intermediate_storage/stage1_raw_htmls',
                          raw_racedf_dir='data/intermediate_storage/stage2_single_race_dataframes',
                          html_records_path='data/intermediate_storage/html_records.pickle')

    invalid_list = []
    if os.path.isfile(invalid_dates_path):
        with open(invalid_dates_path, 'r') as fh:
            for line in fh:
                invalid_list.append(line.strip())


    for date in date_list:
        if date in invalid_list:
            print('Date %s Invalid. Skipped' % date)
            continue
        selected_url = url_base + date
        try:
            crawler.scrap(selected_url)
        except TimeoutException:
            msg = 'Time out: %s\n'% selected_url
            print(msg)
            with open(log_path, 'a') as fh:
                fh.write(msg)

        except NoSuchElementException:
            with open(invalid_dates_path, 'a') as fh:
                fh.write(date + '\n')
            print('Invalid link')

        except Exception:

            msg = 'Unknown exception: %s' % selected_url
            with open(log_path, 'a') as fh:
                fh.write(msg + '\n')
                traceback.print_exc(file=fh)
                fh.write('\n')