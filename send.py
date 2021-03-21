def send2(ycx):
    # _____________________发数据到服务器
    import requests
    import time
    import json
    x = time.localtime()
    now = time.mktime(x)
    url = 'http://127.0.0.1:3000/api/rec'
    j = json.dumps(ycx)
    h = {
        'Content-Type':'application/x-www-form-urlencode'
    }
    r = requests.post(url, data=j,headers=h)

data = {
    'seat_taken': ['9'], 
    'leave': ['1', '5', '8'], 
    'empty_seat': ['2', '3', '4', '6', '7', '10', '11', '12', '13', '14', '15', '16']
}

def sendFuck(data):
    import requests
    import json
    url = 'http://127.0.0.1:3000/api/rec'
    j = json.dumps(data)
    requests.post(url,
                  json=[j])
sendFuck(data)
