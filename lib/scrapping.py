from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
from lib.utils import write_pickle
import pandas as pd
import numpy as np
import re
import os
import pickle
import time


class HTMLRecorder:
    def __init__(self, results_dir='data/intermediate_storage/stage1_raw_htmls',
                 html_records_path='data/intermediate_storage/html_records.pickle'):
        self.results_dir = results_dir
        self.html_records_path = html_records_path
        self.html_records = self._load_html_records()

    def _load_html_records(self):
        if os.path.exists(self.html_records_path):
            with open(self.html_records_path, 'rb') as fh:
                data = pickle.load(fh)
                return data
        else:
            return dict()

    def store_as_record(self, html_link, html_text):
        # Write to file
        write_path = os.path.join(self.results_dir, html_link.replace('/', '_'))
        with open(write_path, 'w') as fh:
            fh.write(html_text)

        # Update record dictionary
        self.html_records[html_link] = write_path

        # Update record file
        with open(self.html_records_path, 'wb') as fh:
            pickle.dump(self.html_records, fh)

    def get_html_path(self, url):
        return self.html_records.get(url)

    def get_stored_html(self, url):
        html_path = self.get_html_path(url)
        html_text = None

        if html_path:  # If the path can be retrieved from recorder
            if os.path.exists(html_path):  # If the recorded path is correct
                with open(html_path, 'r') as fh:
                    html_text = fh.read()

        return html_text


class SingleResultScrapper:
    def __init__(self, race_date, race_location, html_recorder, url_base='https://racing.hkjc.com'):
        self.race_date = race_date
        self.race_location = race_location
        self.url_base = url_base
        self.html_recorder = html_recorder
        self.bet_types = ["WIN", "PLACE", "QUINELLA", "QUINELLA PLACE"]
        self.betkey_conversion_dict = {
            'WIN': 'win',
            'PLACE': 'place',
            'QUINELLA': 'quinella',
            'QUINELLA PLACE': 'quinella_place',
        }

    def scrape_and_formDF(self, result_html, race_place):

        # Scrap
        race_num, track_length, going, course = scrape_race_meta(result_html)
        bet_dict = self._scrape_bets(result_html)
        table_dict = self._scrape_results_table(result_html)

        # Append race table
        num_horses = len(table_dict['place'])
        table_dict['race_no'] = [race_place] * num_horses
        table_dict['race_num'] = [race_num] * num_horses
        table_dict['track_length'] = [track_length] * num_horses
        table_dict['going'] = [going] * num_horses
        table_dict['course'] = [course] * num_horses
        table_dict['race_date'] = [self.race_date] * num_horses
        table_dict['location'] = [self.race_location] * num_horses

        # Construct bet df
        bet_single_df = pd.DataFrame({'race_date':self.race_date, 'race_num':race_num, 'bet_dict': bet_dict})
        bet_single_df.index.name = 'bet_type'
        bet_single_df = bet_single_df.reset_index()
        # Fill columns that are not found
        for key in table_dict.keys():
            if len(table_dict[key]) == 0:
                table_dict[key] = [np.nan] * num_horses

        # Convert to appropriate datatype
        df = pd.DataFrame(table_dict)
        return df, bet_single_df

    def _scrape_results_table(self, html):
        table_dict = scrape_results_table(self.url_base, html, self.html_recorder)
        return table_dict

    def _scrape_bets(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        dividend_table = soup.find('div', class_='dividend_tab').find('table', class_='table_bd')
        if dividend_table is None:
            return {self.betkey_conversion_dict[bet_type]:[] for bet_type in self.bet_types}
        all_trs = dividend_table.find('tbody').find_all('tr')

        bet_scrape_dict = dict()  # bet_type : (tr_idx, row_spane)
        bet_output_dict = dict()  # bet_type : (horse_combination, odds)
        row_skip = 1

        # Record which tr has wanted bet information
        for tr_idx, tr_each in enumerate(all_trs):
            try:
                # Assuming title must has row_space attribute
                row_span = int(tr_each.find_all('td')[0]['rowspan'])
            except KeyError:
                continue

            title_element = tr_each.find_all('td')[0]
            bet_type = soup_string(title_element)
            if bet_type in self.bet_types:
                bet_scrape_dict[bet_type] = (tr_idx, row_span)
            else:
                continue

        # Scraping the target tr
        for bet_type in bet_scrape_dict.keys():
            (tr_idx, row_span) = bet_scrape_dict[bet_type]
            bet_output_dict[self.betkey_conversion_dict[bet_type]] = []
            if row_span == 1:
                all_tds = all_trs[tr_idx].find_all('td')
                assert soup_string(all_tds[0]) == bet_type
                output_dict_key = self.betkey_conversion_dict[bet_type]
                horse_nos_tuple = tuple( float(x) for x in soup_string(all_tds[1]).split(','))
                bet_odds = float(soup_string(all_tds[2]).replace(',', ''))
                bet_output_dict[output_dict_key].append((horse_nos_tuple, bet_odds))

            if row_span > 1:

                for row_idx in range(row_span):
                    progress_idx = tr_idx + row_idx
                    if row_idx == 0:
                        all_tds = all_trs[progress_idx].find_all('td')
                        assert "".join(all_tds[0].strings).strip() == bet_type
                        output_dict_key = self.betkey_conversion_dict[bet_type]
                        horse_nos_tuple = tuple(float(x) for x in soup_string(all_tds[1]).split(','))
                        bet_odds = float(soup_string(all_tds[2]).replace(',', ''))
                        bet_output_dict[output_dict_key].append((horse_nos_tuple, bet_odds))
                    else:
                        all_tds = all_trs[progress_idx].find_all('td')
                        output_dict_key = self.betkey_conversion_dict[bet_type]
                        horse_nos_tuple = tuple(float(x) for x in soup_string(all_tds[0]).split(','))
                        bet_odds = float(soup_string(all_tds[1]).replace(',', ''))
                        bet_output_dict[output_dict_key].append((horse_nos_tuple, bet_odds))

        return bet_output_dict





class HKJCCrawler:
    """
    Scrape all the races within one race date. Every url query will store the fetched HTML source, such that in the next
    encounter of the same url, the HTML source is not needed to download again.

    Example use:
        crawler = HKJCCrawler(rawhtml_dir, raw_racedf_dir, html_records_path)
        crawler.scrape("https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2019/12/29")

    """

    def __init__(self, rawhtml_dir='data/intermediate_storage/stage1_raw_htmls',
                 raw_racedf_dir='data/intermediate_storage/stage2_single_race_dataframes',
                 bet_table_dir='data/intermediate_storage/stage2_bet_tables',
                 html_records_path='data/intermediate_storage/html_records.pickle'):
        """

        Args:
            rawhtml_dir (str): Directory that stores html source code every time after query.

            raw_racedf_dir (str): Directory that stores output dataframes. Each of them contains all horse and race
            information within a race date after scrapping.

            html_records_path (str): Path for the dictionary object which store the links between url and the
            corresponding downloaded HTML source.

        """

        self.rawhtml_dir = rawhtml_dir
        self.raw_racedf_dir = raw_racedf_dir
        self.bet_table_dir = bet_table_dir
        self.html_recorder = HTMLRecorder(results_dir=rawhtml_dir, html_records_path=html_records_path)

    def scrape(self, url, skip_existed=True):
        """
        Scrape all race information within the url, which contains all races of a race date.
        Args: url (str): Url ending with the race date. E.g.
        https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2019/12/29

        Returns:

        """
        print('Main date: ', url)
        date = get_date_from_url(url)
        save_path = os.path.join(self.raw_racedf_dir, '%s.csv' % (date.replace('/', '-')))
        save_bet_path = os.path.join(self.bet_table_dir, '%s_bet.pickle' % (date.replace('/', '-')))
        if skip_existed:
            if os.path.isfile(save_path) and os.path.isfile(save_bet_path):
                print('Date %s existed, skipped.' % (date))
                return None

        result_main_html = get_html_by_both(self.html_recorder, url, selected_page='result', max_trial=10)

        all_races_links, (location, max_num_races) = scrap_all_races(result_main_html, url)
        single_scrapper = SingleResultScrapper(race_date=date, race_location=location, html_recorder=self.html_recorder)

        df_list = []
        bet_df_list = []
        for linkid, race_link in enumerate(all_races_links):
            print('Sub-race: ', race_link)
            results_html = get_html_by_both(self.html_recorder, race_link, selected_page='result', max_trial=10)
            df, bet_single_df = single_scrapper.scrape_and_formDF(results_html, linkid + 1)
            df_list.append(df)
            bet_df_list.append(bet_single_df)

        df_all = pd.concat(df_list, axis=0)
        df_all.to_csv(save_path)
        bet_df_all = pd.concat(bet_df_list, axis=0)
        write_pickle(bet_df_all, save_bet_path)

def get_specifichtml_by_request(url, selected_page='result'):
    """
    Query for the HTML source based on the input url and the wanted type (result page or horse page).

    Args:
        url (str): Target url to query the source HTML.
        selected_page (str): 'result' or 'horse'

    Returns:
        html_text (str or None): If query is successful/valid, return the HTML source. Otherwise, None.
        result_tag (str): 'valid' if the url matches the arg:selected_page, 'invalid' if not. 'timeout' if failed after
        re-querying the url for too many times.
    """
    if selected_page == 'result':
        selected_css_class = '.performance'  # '.performance' is necessary for identifying race performance result page
    elif selected_page == 'horse':
        selected_css_class = '.horseProfile'  # '.horseProfile' is necessary for identifying horse profile page
    else:
        selected_css_class = '.performance, .horseProfile'  # If not specify, then either one of both.

    print('requesting url: ', url)
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    wait = WebDriverWait(driver, 5)
    html_text = None
    result_tag = ''  # 'valid', 'invalid', 'timeout'
    try:
        # First we screen for all possible known page types.
        # ".localResults" does not guarantee having race results, but ".performance" does.
        element = wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, ".performance, .localResults, .horseProfile, .errorout, .simulcastContainer, #main-content")))

        try:
            # .performance or .horseProbile are our main targets
            driver.find_element_by_css_selector(selected_css_class)
            driver.get(url)
            html_text = driver.page_source
            result_tag = 'valid'
            return html_text, result_tag

        except NoSuchElementException:  # If our target is not found.
            try:
                # Known invalid page types.
                # ".simulcastContainer" is foreign race page, which we don't want.
                # ".errorout" or "#main-content"  are error pages with no race result at all.
                driver.find_element_by_css_selector('.simulcastContainer, .errorout, #main-content')
                result_tag = 'invalid'
                return html_text, result_tag
            except NoSuchElementException:
                result_tag = 'invalid'
                return html_text, result_tag

    except TimeoutException:  # If waiting for too long.
        result_tag = 'timeout'
        return html_text, result_tag


def get_html_by_both(html_recorder, url, selected_page, _counter=0, max_trial=10):
    """
    Check if the asked url was already recorded in html_recorder. If yes, get the HTML source from the html_recorder.
    If no, request the HTML source from the url's server. Raise "NoSuchElementException" if the url page is not the
    target type (arg:selected_page). Raise "TimeoutException" if unsuccessful request is made too many times.

    Args:
        html_recorder (HTMLRecorder object): HTML recorder that record downloaded HTML sources.
        url (str): URL link whose HTML source is wanted.
        selected_page (str): Target page type. Can be 'result' or 'horse'.
        _counter (int): Internal recursive counter.
        max_trial (int): Maximum number of unsuccessful request before TimeoutException is raised.

    Returns:

    """
    html_text = html_recorder.get_stored_html(url)
    if html_text is None:
        html_text, request_result = get_specifichtml_by_request(url, selected_page=selected_page)
        if (html_text is not None) & (request_result == 'valid'):
            html_recorder.store_as_record(url, html_text)
            return html_text
        elif request_result == 'invalid':
            print('Invalid request')
            raise NoSuchElementException
        else:

            if _counter > max_trial:
                raise TimeoutException
            _counter += 1
            time.sleep(_counter * 10)
            print('Retry#%d/%d: %s ' % (_counter, max_trial, url))
            return get_html_by_both(html_recorder, url, selected_page=selected_page, _counter=_counter)
    else:
        return html_text


def scrap_all_races(result_html, url):
    """
    Scrape links of different races within the result page of a racing date
    Args:
        result_html (str): HTML source of the local results of a racing date.
        html_base (str): URL of the HKJC racing website.

    Returns:
        all_links (list): All links of different races within the day
        location (str): Location of the racing
        max_race_num (int):

    """
    soup = BeautifulSoup(result_html, 'html.parser')
    found = soup.find('table', class_='js_racecard').tbody.tr.find_all('a')  # Race selection table
    link = found[-2]['href']  # Get the link of the second last data (last race number) from the table
    location = re.search('Racecourse=(.*?)&', link).group(1)
    max_race_num = int(re.search('RaceNo=(.*?$)', link).group(1))

    all_links = []
    for i in range(max_race_num):
        race_num = i + 1
        link_each_num = url + "&Racecourse=%s" % (location) + '&RaceNo=%d' % (race_num)
        all_links.append(link_each_num)
    return all_links, (location, max_race_num)


def scrap_horse_info(horse_html):
    """
    Get origin information from the HTML source of a horse
    Args:
        horse_html (str): HTML source of a horse's website
    Returns:
        origin (str): origin of the horse
    """
    horse_soup = BeautifulSoup(horse_html, 'html.parser')
    table_temp = horse_soup.find('table', class_='horseProfile').find('table', class_='table_eng_text')
    origin_age_td = table_temp.tbody.find_all('tr')[0].find_all('td')[2]
    origin_age_text = "".join(origin_age_td.strings).strip()
    match_obj = re.search('\\/', origin_age_text)
    if match_obj:  # Origin + Age (unused) case
        origin = origin_age_text.split('/')[0].strip()
    else:
        origin = origin_age_text.strip()
    return origin


def get_date_from_url(url):
    return re.search("RaceDate=(\d\d\d\d\\/\d\d\\/\d\d)", url).group(1)


def scrape_race_meta(html):
    soup = BeautifulSoup(html, 'html.parser')

    # Information from the page (date, race_num, track length, location, going, course)
    page_tbody = soup.find('tbody', class_='f_fs13')

    # Race number
    race_title = "".join(page_tbody.parent.thead.tr.td.strings)
    race_num = re.search("\\((\d*?)\\)", race_title).group(1)

    # Track length, Going and Course
    page_trs = page_tbody.find_all('tr')
    page_tds1 = page_trs[1].find_all('td')  # <tr> for the track length and going
    page_tds2 = page_trs[2].find_all('td')  # <tr> for course
    track_info = "".join(page_tds1[0].strings)
    track_length = re.search('(\d*?)M', track_info).group(1)
    going = "".join(page_tds1[2].strings)
    course = "".join(page_tds2[2].strings)

    return race_num, track_length, going, course


def scrape_results_table(url_base, html, html_recorder):
    """
    Scrape horse and race information from the performance table of the HTML page of race result.

    Args:
        url_base (str): Expected to be "https://racing.hkjc.com". Normally no need to change.
        html (str): HTML source text of the result page
        html_recorder (HTMLRecorder object): html recorder to store downloaded source/ retrieve downloaded source.

    Returns:

    """


    soup = BeautifulSoup(html, 'html.parser')

    # Information of horses
    data_dict = {
        'place': [],
        'horse_no': [],
        'horse_code': [],
        'horse_name': [],
        'horse_link': [],
        'horse_origin': [],
        'jockey_name': [],
        'trainer_name': [],
        'act_weight': [],
        'decla_weight': [],
        'draw': [],
        'lbw': [],
        'time': [],
        'odds': []
    }
    conversion_dict = {
        'Plc.': 'place',
        'Horse No.': 'horse_no',
        'Horse': 'horse',
        'Jockey': 'jockey_name',
        'Trainer': 'trainer_name',
        'Actual Wt.': 'act_weight',
        'Declar. Horse Wt.': 'decla_weight',
        'Draw': 'draw',
        'LBW': 'lbw',
        'RunningPosition': 'running_pos',
        'Finish Time': 'time',
        'Win Odds': 'odds'
    }

    all_trs = soup.find('div', class_='performance').find('table').find_all('tr')

    for tr_idx, tr in enumerate(all_trs):
        if tr_idx == 0:
            all_tds_0 = tr.find_all('td')
            html_cols = ["".join(x.strings).strip() for x in all_tds_0]
        else:
            all_tds = tr.find_all('td')
            for td_idx, td in enumerate(all_tds):

                data_title = conversion_dict[html_cols[td_idx]]  # Column title
                # Running Position is not needed
                if data_title == "running_pos":
                    continue

                # Get all of texts inside <td></td>
                td_text = "".join(td.strings).strip()
                # For horse_code, horse_name, horse_origin, horse_link
                if data_title == "horse":
                    horse_code = re.search('\\((.*?)\\)', td_text).group(1)
                    horse_name = re.search('(.*?)\\(', td_text).group(1)
                    data_dict['horse_code'].append(horse_code)
                    data_dict['horse_name'].append(horse_name)
                    a_tag = td.find('a')  # Remember to except the case when no <a> is found
                    if a_tag:
                        link = a_tag['href']
                        horse_link = url_base + link

                        # Speed up by using html recorder
                        html_text = get_html_by_both(html_recorder, horse_link, selected_page='horse', max_trial=10)
                        origin = scrap_horse_info(html_text)

                    else:
                        horse_link = None
                        origin = None
                    data_dict['horse_origin'].append(origin)
                    data_dict['horse_link'].append(horse_link)
                else:
                    # For place, horse_no, horse, jockey, trainer, actual_weight, decla_weight, draw, lbw, finish_time, odds
                    try:
                        data_dict[data_title].append(td_text)
                    except KeyError:
                        pass
    return data_dict



def soup_string(element, strip=True):
    if strip:
        return "".join(element.strings).strip()
    else:
        return "".join(element.strings)