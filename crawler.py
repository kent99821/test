import requests
from bs4 import BeautifulSoup
import re

url = 'http://libtc.gdou.edu.cn:8080/opac/ranking/bookLoanRank?libcode=GDHYDX&bookType=&limitDays=30'
try:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    html_doc = r.text
except:
    print("失败")

page = BeautifulSoup(html_doc,"html.parser",from_encoding="utf-8")

rank_dir = []
for i, table in enumerate(page.find_all('table')):
	for j, tr in enumerate(table.find_all('tr')):
		if j != 0:
			tds = tr.find_all('td')
			book = re.findall(r'>(.*)<', str(tds[1].contents[0]))
			book = book[0]
			rank_dir.append({
				'排名': tds[0].contents[0],
				'书名': book,
				'作者': tds[2].contents[0],
				'出版社': tds[3].contents[0],
				'出版时间': tds[4].contents[0],
				'借阅量': tds[5].contents[0]
				})

for rd in rank_dir:
	print(rd)
	print("\n")