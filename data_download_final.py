#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import requests
import json
import time
import logging
import traceback



gas_url_domain = "dev.yo-i.com.cn:8443"
gas_project_scopes = "energy/gas"
gas_cookie = "thingswise.web.proxy.session_id=s%3ABbuVqLYF98fuSmX3B9M7DAe77Fc1nkJH.1FJtGevRaRi9WvjOymc03zV6%2FDe8DCUT55AbZ9zCi6s; __guid=165663402.1502813834321714700.1590023534580.6956; thingswise.web.app.session_id=s%3AOdDY10p1wYEoyXPqxxI5rezN.J8b3vEoIeHd0c9y6VDjx6MIrEo%2BJ2iAqf1SXUQRocgQ; monitor_count=65"

yida_dev_url_domain = "dev.yo-i.com.cn:8443"
yida_dev_project_scopes = "yida/chp"
yida_dev_cookie = "thingswise.web.app.session_id=s%3A7Db7aRmHzUOE259JdIVML4UH.01rlLTbr3myKvvWbjxjVdVCggsvJFxOp1%2F6MgoeTXGA; thingswise.web.proxy.session_id=s%3A-1_02v2Cl87VeVQ9oS2vKwVmjAXIYAIa.EvYq32YSCsLxayNvp0EhW6fbITpVAaJyLhKADZVvGrQ"

yida_production_url_domain = "192.168.108.221"
yida_production_project_scopes = "yida/energy"
yida_production_cookie = "thingswise.web.proxy.session_id=s%3AYcEyCnrz_KQjNp21XjAJI6lf0F22XHLh.n9nq%2Fiqbr0tjicblNFjV9TKr4XE8GiE7GFk0TYJ%2BMRs; thingswise.web.app.session_id=s%3AG9urI4nMlxQp5bJJ4m5fafy7.xKRdcDgxg7Y4Rmtex%2BoLQZKYq9Cm0zHJcrXMpts%2Bdi8"


chp_parameters = {
    "boiler_m1": {
        "url_domain": yida_production_url_domain,
        "project_scope": yida_production_project_scopes,
        "cookie": yida_production_cookie,
        "geo": "ydgm_e1|hpp_f1|boiler_w1|boiler_m1",
        "from_timestamp": 1597280400000,
        "to_timestamp": 1597280700000,
        "values": {
            "CFBoilerOP": {
                "5s": [
                        "coal_composition_Car",
                        # "coal_composition_Har",
                        # "coal_composition_Oar",
                        # "coal_composition_Nar",
                        # "coal_composition_St_ar",
                        # "coal_composition_Mt",
                        # "coal_composition_Aar",
                        # "coal_composition_Qnet_ar",
                        # "coal_composition_Vdaf",
                        # "fly_ash_carbon_content",
                        # "large_slag_carbon_content",
                        # "fly_ash_share",
                        # "big_slag_temperature",
                        # "burning_coal_actual_surface_temperature",
                        # "price_per_coal",
                        # "cost_loss",
                        # "cur_switch",
                        # "efficiency_forward",
                        # "efficiency_reverse",
                        # "efficiency_deviation",
                        # "heat_loss_q2",
                        # "heat_loss_q3",
                        # "heat_loss_q4",
                        # "heat_loss_q5",
                        # "heat_loss_q6",
                        # "efficiency_forward_optimal_quantile",
                        # "efficiency_reverse_optimal_quantile",
                        # "efficiency_deviation_optimal_quantile",
                        # "heat_loss_q2_optimal_quantile",
                        # "heat_loss_q3_optimal_quantile",
                        # "heat_loss_q4_optimal_quantile",
                        # "heat_loss_q5_optimal_quantile",
                        # "heat_loss_q6_optimal_quantile",
                        # "exhaust_oxygen_concentration_optimal_quantile",
                        # "exhaust_gas_temperature_optimal_quantile",
                        # "efficiency_forward_worst_quantile",
                        # "efficiency_reverse_worst_quantile",
                        # "efficiency_deviation_worst_quantile",
                        # "heat_loss_q2_worst_quantile",
                        # "heat_loss_q3_worst_quantile",
                        # "heat_loss_q4_worst_quantile",
                        # "heat_loss_q5_worst_quantile",
                        # "heat_loss_q6_worst_quantile",
                        # "exhaust_oxygen_concentration",
                        # "exhaust_gas_temperature",
                        # "air_inlet_temperature",
                        # "feed_water_flow",
                        # "feed_water_temperature",
                        # "feed_water_pressure",
                        # "outlet_steam_flow",
                        # "main_steam_temperature",
                        # "main_steam_pressure",
                        # "fluidized_bed_differential_pressure",
                        # "coal_consumption",
                        # "smoke_extraction_area_CO_concentration",
                        # "fluidized_bed_temperature",
                        # "damage_percentage",
                        # "hp_steam_boiler_opt",
                        # "hp_steam_boiler_opt_array",
                        # "goodness_of_fit_morning",
                        # "goodness_of_fit_afternoon",
                        # "goodness_of_fit_night",
                        # "CoalFeeder_No1_RotatorySpeed",
                        # "CoalFeeder_No2_RotatorySpeed",
                        # "CoalFeeder_No3_RotatorySpeed",
                        # "Furnace_BottomNo1_Temperature",
                        # "Furnace_BottomNo2_Temperature",
                        # "Furnace_MiddleNo1_Temperature",
                        # "Furnace_MiddleNo2_Temperature",
                        # "Furnace_TopNo1_Temperature",
                        # "Furnace_TopNo2_Temperature",
                        # "FirstCombustionAir_No1_FlowrateVolumn",
                        # "FirstCombustionAir_No2_FlowrateVolumn",
                        # "FirstCombustionAir_No1_Temperature",
                        # "SecondCombustionAir_No1_FlowrateVolumn",
                        # "SecondCombustionAir_No2_FlowrateVolumn",
                        # "SecondCombustionAir_No1_Temperature",
                        # "Exhaust_AfterHighOverHeater_Temperature",
                        # "Exhaust_AfterLowOverHeater_Temperature",
                        # "Exhaust_AfterEconomizer_Temperature",
                        # "BoilerDrum_Level",
                        # "BoilerDrum_Pressure",
                        # "Furnace_No1_PressureDiff",
                        # "Furance_OutletGas_Temperature",
                        # "Furnace_Top_Vacuum",
                        # "Exhaust_Outlet_SO2",
                        # "Exhaust_Outlet_NOX",
                        # "Exhaust_Outlet_Particle",
                        # "CoalFeeder_No1_Current",
                        # "CoalFeeder_No2_Current",
                        # "CoalFeeder_No3_Current",
                        # "DSWater_Ahead_Flowrate",
                        # "DSWater_Rear_Flowrate",
                        # "motor_speed",
                        # "blowing_steam_flow",
                        # "blowing_steam_temperature",
                        # "blowing_steam_pressure",
                        # "desuperheating_water_flow",
                        # "desuperheating_water_temperature",
                        # "desuperheating_water_pressure",
                        # "coal_cost",
                        # "heat_cost",
                        # "coal_calorific_value",
                        # "CoalFeeder_No1_Frequency",
                        # "CoalFeeder_No2_Frequency",
                        # "CoalFeeder_No3_Frequency"
                ],
            },
        },
    },
}


gas_parameters = {
    "cvt_1_e": {
        "url_domain": gas_url_domain,
        "project_scope": gas_project_scopes,
        "cookie": gas_cookie,
        "geo": "steel_c|steel_d|steel_1_p|cvt_1_e",
        "from_timestamp": 1597280400000,
        "to_timestamp": 1597280700000,
        "values": {
            "EF_BFGas": {
                # "resolution": "5s",
                "5s": ["pressure", "inlet_flow_rate"],
                # "1min": []
            },
            "EF_CVGas": {
                # "resolution": "5s",
                "5s": ["outlet_flow_rate"]
            },
            "ProcOp":{
                # "resolution": "5s",
                "5s": ["dip_angle", "oxygen_valve_state", "converter_state"]
            },
        },
    },
}


def unix2time(value):
    """
    将UNIX时间戳转换为日期时间格式
    """
    format = "%Y-%m-%d %H:%M:%S"
    value = time.localtime(value)
    dt = time.strftime(format, value)

    return dt







def query_data(url_domain, project_scope, cookie, domain, geo, metric, from_timestamp, to_timestamp, resolution):
    """
    请求时序数据, 一次只能查一个 geo 的 一个 metric
    """
    # 请求数据参数
    url = "https://%s/%s/api/view/metric" % (url_domain, project_scope)
    logging.error("url:%s", url)
    params = {
        "name": domain + "." + metric,
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
            timeout = 10
        )
        logging.error("response url:%s", response.url)
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


def parse_data(result, domain):
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
    df = pd.DataFrame({})
    for key, value in parameters.items():
        instance = key
        url_domain = value["url_domain"]
        project_scope = value["project_scope"]
        cookie = value["cookie"]
        geo = value["geo"]
        from_timestamp = value["from_timestamp"]
        to_timestamp = value["to_timestamp"]
        values = value["values"]
        for k, v in values.items():
            domain = k
            for i, j in v.items():
                resolution = i
                metrics = j
                for metric in metrics:
                    data = query_data(
                        url_domain = url_domain,
                        project_scope = project_scope,
                        cookie = cookie,
                        domain = domain,
                        geo = geo,
                        metric = metric,
                        from_timestamp = from_timestamp,
                        to_timestamp = to_timestamp,
                        resolution = resolution
                    )
                    temp_df = parse_data(data, domain)
                    df = pd.concat([df, temp_df], axis = 1)
                    print(df)
    # df.to_csv('~/Desktop/%s.csv' % instance)









if __name__ == "__main__":
    integrate_data(chp_parameters)
