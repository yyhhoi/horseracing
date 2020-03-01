import datetime

def generate_date_list(newday, oldday):
    diff = newday-oldday
    date_list = []
    for i in range(diff.days):
        subtractor = datetime.timedelta(days=i + 1)
        new_date = newday - subtractor
        new_date_str = new_date.strftime("%Y/%m/%d")
        date_list.append(new_date_str)
    return date_list
