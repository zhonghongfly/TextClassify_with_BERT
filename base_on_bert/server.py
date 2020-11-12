# -*- coding: utf-8 -*-
'''
@author: zhonghongfly@foxmail.com
@license: (C) Copyright 2020
@desc: 线上服务
@DateTime: Created on 2020/9/27, at 下午 05:07 by PyCharm
'''

from sanic import Sanic
from sanic.response import json as Rjson
from base_on_bert.predict_GPU import Bert_Class

import re

REGEX = "#{1,2}[0-9]+|[img:(.*?)]|[a-zA-Z0-9  \n\t]"

app = Sanic()
my = Bert_Class()


@app.route("/", methods=['GET', 'POST'])
async def home(request):
    # 1，首先要从HTTP请求获取用户的字符串
    dict1 = {'tips': '请用POST方法，传递“question”字段'}
    if request.method == 'GET':
        key_str = request.args.get('question')
    elif request.method == 'POST':
        for k, v in request.form.items():
            dict1[k] = v  # 最关心的问题字段是keyword
        key_str = request.form.get('question')
    else:
        return Rjson(dict1)
    if not key_str:  # 如果有空的字段，返回警告信息。
        return Rjson(dict1)

    # 2，调用自身的功能，执行搜索引擎爬虫
    dict1.pop('tips')
    # 过滤无用字符
    key_str = re.sub(REGEX, "", key_str)
    print(key_str)
    dict1['Type'] = my.predict_on_pb(key_str)
    return Rjson(dict1)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5400)
    # key_str = "#324235435 ??????? 、、、、、、、 高峰违规？\n？？？ 恢复规划法规和img:fggghgghghgh???"
    # key_str = re.sub(REGEX, "", key_str)
    # print("key_str ==> " + key_str)
