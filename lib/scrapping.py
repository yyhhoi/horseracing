from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import re
import os
import pickle
import time

class HTMLRecorder:
    def __init__(self, results_dir='data/intermediate_storage/stage1_raw_htmls', html_records_path='data/intermediate_storage/html_records.pickle'):
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


def get_specifichtml_by_request(url):
    print('requesting url: ', url)
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    wait = WebDriverWait(driver, 5)
    html_text = None
    result_tag = ''  # 'valid', 'invalid', 'timeout'
    try:
        element = wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, ".horseProfile, .errorout, .simulcastContainer, .localResults")))

        try:
            driver.find_element_by_css_selector('.localResults, .horseProfile')
            driver.get(url)
            html_text = driver.page_source
            result_tag = 'valid'
            return html_text, result_tag

        except NoSuchElementException:
            try:
                driver.find_element_by_css_selector('.simulcastContainer, .errorout')
                result_tag = 'invalid'
                return html_text, result_tag
            except NoSuchElementException:
                result_tag = 'invalid'
                return html_text, result_tag

    except TimeoutException:
        result_tag = 'timeout'
        return html_text, result_tag



def get_html_by_both(html_recorder, link, counter=0, max_trial=10):

    html_text = html_recorder.get_stored_html(link)
    if html_text is None:
        html_text, request_result = get_specifichtml_by_request(link)
        if (html_text is not None) & (request_result == 'valid'):
            html_recorder.store_as_record(link, html_text)
            return html_text
        elif request_result == 'invalid':
            print('Invalid request')
            raise NoSuchElementException
        else:

            if counter > max_trial:
                raise TimeoutException
            counter += 1
            time.sleep(counter * 10)
            print('Retry#%d/%d: %s '%(counter, max_trial, link))
            return get_html_by_both(html_recorder, link, counter=counter)
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
    found = soup.find('table', class_='js_racecard').find_all('td')  # Race selection table
    link = found[-2].a['href']  # Get the link of the second last data (last race number) from the table
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
    soup = BeautifulSoup(html, 'html.parser')

    # Information of horses
    all_trs = soup.find('div', class_='performance').find('table').find_all('tr')
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
    col_titles = ['place', 'horse_no', 'horse', 'jockey_name', 'trainer_name', 'act_weight', 'decla_weight', 'draw', 'lbw',
                  'running_pos', 'time', 'odds']

    for tr_idx, tr in enumerate(all_trs):
        if tr_idx > 0:
            all_tds = tr.find_all('td')
            for td_idx, td in enumerate(all_tds):

                data_title = col_titles[td_idx]  # Column title
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
                        html_text = get_html_by_both(html_recorder, horse_link, max_trial=10)
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