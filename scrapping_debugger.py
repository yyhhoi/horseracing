from lib.scrapping import HTMLRecorder, get_html_by_both, scrap_horse_info
from bs4 import BeautifulSoup
import re
import pprint


pprinter = pprint.PrettyPrinter(indent=4)
race_link = "https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2006/10/25&Racecourse=HV&RaceNo=1"
# race_link = "https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2019/12/29&Racecourse=ST&RaceNo=1"
url_base='https://racing.hkjc.com'

rawhtml_dir='data/intermediate_storage/stage1_raw_htmls'
html_records_path='data/intermediate_storage/html_records.pickle'
html_recorder = HTMLRecorder(results_dir=rawhtml_dir, html_records_path=html_records_path)
html = get_html_by_both(html_recorder, race_link, selected_page='result', max_trial=10)

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



pprinter.pprint(data_dict)
print()