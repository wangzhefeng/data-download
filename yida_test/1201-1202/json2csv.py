# -*- encoding: utf-8 -*-
'''
@文件    :解析平台api获取的json数据.py
@说明    :
@时间    :2020/07/02 11:35:41
@作者    :stawary
@版本    :1.0
'''

'''
For parse dm API json files
'''

import pandas as pd
import numpy as np
import json
import glob
import datetime
import time


def convert2csv(path, type="merge"):
    if path == '':
        json_files = glob.glob('*.json')
    else:
        json_files = glob.glob(path + '/' + '*.json')
    print("json_files:", json_files)

    if type == 'merge':
        all_df = pd.DataFrame()
        all_df['time'] = ''
        for json_file in json_files:
            with open(json_file,'r') as load_f:
                result_dict = json.load(load_f)
            key = result_dict['name']
            data = result_dict['data']

            times = []
            values = []

            for d in data:
                times.append(d['name'])
                values.append(d['value']['sum']/d['value']['cnt'])

            df = pd.DataFrame({"time":times,key:values})
            df['time'] = df['time'].apply(lambda x:x.split('.')[0][1:])
            df['time'] = df['time'].apply(lambda x:time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(int(x))))
            df['time'] = pd.to_datetime(df['time'])

            all_df = pd.merge(all_df,df,on='time', how='outer')
        all_df.to_csv('result-{}.csv'.format(time.strftime('%m%d_%H%M%S',time.localtime(time.time()))),index=False)

    elif type == 'concat':
        all_df = pd.DataFrame()
        for json_file in json_files:
            with open(json_file,'r') as load_f:
                result_dict = json.load(load_f)
            key = result_dict['name']
            data = result_dict['data']

            times = []
            values = []

            for d in data:
                times.append(d['name'])
                values.append(d['value']['sum']/d['value']['cnt'])

            df = pd.DataFrame({"time":times,key:values})
            df['time'] = df['time'].apply(lambda x:x.split('.')[0][1:])
            df['time'] = df['time'].apply(lambda x:time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(int(x))))
            df['time'] = pd.to_datetime(df['time'])

            all_df = pd.concat([all_df,df],axis=0)
            print(all_df)
        all_df.to_csv('result-{}.csv'.format(time.strftime('%m%d_%H%M%S',time.localtime(time.time()))),index=False)

if __name__ == '__main__':
    #path:json文件目录，如果多个json文件的metric要生成到同一个csv文件里变成多列，可将其放到同一目录下，然后函数type='merge'
    #如果多个json文件是同一metric的，将其放在同一目录下，然后函数type='concat'，生成只有一个metric的csv文件

    path = "/mnt/e/dev/data-download/yida_test/1201-1202"
    # path = '/Users/zfwang/work/dev/data-download/yida_test/1109-1110/1110'
    convert2csv(path, type='merge')
