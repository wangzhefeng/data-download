#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import requests
import json
import time
import logging
import traceback
from tqdm import tqdm
import os


"""
# 项目配置参数参考信息，请自便(如其他项目需要，请自行补充进来)：
    
    ## 煤气项目 dev
    gas_url_domain = "dev.yo-i.com.cn:8443"
    gas_project_scopes = "energy/gas"
    gas_cookie = "thingswise.web.proxy.session_id=s%3ABbuVqLYF98fuSmX3B9M7DAe77Fc1nkJH.1FJtGevRaRi9WvjOymc03zV6%2FDe8DCUT55AbZ9zCi6s; __guid=165663402.1502813834321714700.1590023534580.6956; thingswise.web.app.session_id=s%3AOdDY10p1wYEoyXPqxxI5rezN.J8b3vEoIeHd0c9y6VDjx6MIrEo%2BJ2iAqf1SXUQRocgQ; monitor_count=65"
    ## 溢达项目 dev
    yida_dev_url_domain = "dev.yo-i.com.cn:8443"
    yida_dev_project_scopes = "yida/chp"
    yida_dev_cookie = "thingswise.web.app.session_id=s%3A7Db7aRmHzUOE259JdIVML4UH.01rlLTbr3myKvvWbjxjVdVCggsvJFxOp1%2F6MgoeTXGA; thingswise.web.proxy.session_id=s%3A-1_02v2Cl87VeVQ9oS2vKwVmjAXIYAIa.EvYq32YSCsLxayNvp0EhW6fbITpVAaJyLhKADZVvGrQ"
    ## 溢达热电生产环境
    yida_production_url_domain = "192.168.108.221"
    yida_production_project_scopes = "yida/energy"
    yida_production_cookie = "thingswise.web.proxy.session_id=s%3AYcEyCnrz_KQjNp21XjAJI6lf0F22XHLh.n9nq%2Fiqbr0tjicblNFjV9TKr4XE8GiE7GFk0TYJ%2BMRs; thingswise.web.app.session_id=s%3AG9urI4nMlxQp5bJJ4m5fafy7.xKRdcDgxg7Y4Rmtex%2BoLQZKYq9Cm0zHJcrXMpts%2Bdi8"
"""


# 下载数据文件地址
result_path = "result/"
if not os.path.exists(result_path):
    os.mkdir(result_path)


def generate_config(url_domain, project_scopes, cookie, from_timestamp, to_timestamp, class_to_be_download):
    """
    生成配置信息
    """
    root_path = "config/"
    config_df = pd.read_csv(root_path + "config.csv")
    parameters = {}
    for i in range(len(config_df)):
        if config_df["class"].iloc[i] not in class_to_be_download:
            continue
        temp_dict = {
            config_df["geo"].iloc[i].split("|")[-1] + "_" + config_df["resolution"].iloc[i]:{
                "url_domain": url_domain,
                "project_scope": project_scopes,
                "cookie": cookie,
                "geo": config_df["geo"].iloc[i],
                "from_timestamp": from_timestamp,
                "to_timestamp": to_timestamp,
                "resolution": config_df["resolution"].iloc[i],
                "metrics": pd.read_csv(root_path + config_df["metric_file"].iloc[i])["metric_list"].to_list()

            }
        }
        parameters.update(temp_dict)

    return parameters



def unix2time(value):
    """
    将UNIX时间戳转换为日期时间格式
    """
    format = "%Y-%m-%d %H:%M:%S"
    value = time.localtime(value)
    dt = time.strftime(format, value)

    return dt


def query_data(url_domain, project_scope, cookie, geo, metric, from_timestamp, to_timestamp, resolution):
    """
    请求时序数据, 一次只能查一个 geo 的 一个 metric
    """
    # 请求数据参数
    requests.packages.urllib3.disable_warnings()
    url = "https://%s/%s/api/view/metric" % (url_domain, project_scope)
    params = {
        "name": metric,
        "metricType": "timeseries",
        "geo": geo,
        "from": from_timestamp,
        "to": to_timestamp,
        "period": resolution,
    }
    headers = {
        'Cookie': cookie
    }
    try:
        # 数据请求
        response = requests.get(
            url = url,
            params = params,
            headers = headers,
            verify = False,
            timeout = 10
        )
        # logging.error("response url:%s", response.url)
        # 数据解析
        if response.status_code == 200:
            result = response.json()
            result = {
                "data":result["data"],
                "name": result["name"]
            }
    except:
        traceback.print_exc()

    return result


def parse_data(result):
    """
    解析请求到的时序数据, 一次只解析一个 instance 的一个 metric
    """
    metric_name = result["name"]
    df = pd.DataFrame({})
    for col in result["data"]:
        col["name"] = [unix2time(float(col["name"]))]
        col[metric_name] = [col["value"]["sum"] / col["value"]["cnt"]]
        
        temp_df = pd.DataFrame(
            data = {
                metric_name: col[metric_name]
            }, 
            index = col["name"], 
            columns = np.array([metric_name])
        )
        df = pd.concat([df, temp_df], axis = 0)
    
    return df


def integrate_data(parameters):
    
    for key, value in parameters.items():
        df = pd.DataFrame()

        instance = key
        url_domain = value["url_domain"]
        project_scope = value["project_scope"]
        cookie = value["cookie"]
        geo = value["geo"]
        from_timestamp = value["from_timestamp"]
        to_timestamp = value["to_timestamp"]
        resolution = value["resolution"]
        metrics = value["metrics"]

        for metric in tqdm(metrics):
            data = query_data(
                url_domain = url_domain,
                project_scope = project_scope,
                cookie = cookie,
                geo = geo,
                metric = metric,
                from_timestamp = from_timestamp,
                to_timestamp = to_timestamp,
                resolution = resolution
            )
            # logging.error("data=%s", data)
            temp_df = parse_data(data)
            df = pd.concat([df, temp_df], axis = 1, sort = False)
            # print(f"instance={instance}, data = ", df)

        df.to_csv(f'{result_path}{instance}_{int(time.time())}.csv' )



if __name__ == "__main__":
    # 项目配置
    url_domain = "192.168.108.221"
    project_scopes = "yida/energy"
    cookie = "thingswise.web.proxy.session_id=s%3AYcEyCnrz_KQjNp21XjAJI6lf0F22XHLh.n9nq%2Fiqbr0tjicblNFjV9TKr4XE8GiE7GFk0TYJ%2BMRs; thingswise.web.app.session_id=s%3AG9urI4nMlxQp5bJJ4m5fafy7.xKRdcDgxg7Y4Rmtex%2BoLQZKYq9Cm0zHJcrXMpts%2Bdi8"
    # 查询时间
    from_timestamp = 1596988800000  # 2020-08-17 00:00:00
    to_timestamp = 1597593600000    # 2020-08-18 00:00:00
    # 需要下载的类
    class_to_be_download = ["CFBoiler"]
    # 生成下载数据配置参数
    parameters = generate_config(url_domain, project_scopes, cookie, from_timestamp, to_timestamp, class_to_be_download)
    # 下载数据
    integrate_data(parameters)
