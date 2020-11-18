# -*- coding: utf-8 -*-
import seuif97
import logging
import traceback
import numpy as np
from scipy.optimize import minimize
from sklearn import linear_model

from electricity_price import electricity_price_ext_preprocessing
from settings import TURBINE_STEAM_ADJUST_DELTA
from settings import ETURB_M1_MACHINE_STATUS, ETURB_M2_MACHINE_STATUS, BTURB_M1_MACHINE_STATUS


class Eturb:
    
    def __init__(self, instance, steam_flow_in, steam_flow_in_threshold, **kwargs):
        self.instance = instance
        self.steam_flow_in = steam_flow_in
        self.steam_flow_in_threshold = steam_flow_in_threshold
        self.__dict__.update(kwargs)
        self.alpha_1 = np.nan
        self.alpha_2 = np.nan
        self.beta = np.nan
        self.electricity_power = np.nan
        self.machine_status = np.nan

    def effect_m_g(self, machine_statu):
        if machine_statu == 1:
            if self.instance == "eturb_m1":
                self.alpha_1 = 0.173
                self.alpha_2 = -0.0852
                self.beta = -0.3293
            elif self.instance == "eturb_m2":
                self.alpha_1 = 0.1823
                self.alpha_2 = -0.1065
                self.beta = -0.3139
        else:
            self.alpha_1 = 0.01
            self.alpha_2 = 0.01
            self.beta = 0
        logging.error("yida_turbine_model_v2.py turbine:%s, alpha_1:%s, alpha_2:%s, beta:%s", self.instance, self.alpha_1, self.alpha_2, self.beta)

    def electricity_func(self, steam_flow_in, steam_flow_side):
        self.electricity_power = self.alpha_1 * steam_flow_in + self.alpha_2 * steam_flow_side + self.beta
        logging.error("yida_turbine_model_v2.py turbine:%s, electricity_power:%s", self.instance, self.electricity_power)
    
    def calculate_machine_statu(self):
        self.machine_status = 1 if self.steam_flow_in > self.steam_flow_in_threshold else 0
        logging.error("yida_turbine_model_v2.py turbine:%s, machine_status:%s", self.instance, self.machine_status)


class Bturb:

    def __init__(self, instance, steam_flow_in, steam_flow_in_threshold, **kwargs):
        self.instance = instance
        self.steam_flow_in = steam_flow_in
        self.steam_flow_in_threshold = steam_flow_in_threshold
        self.__dict__.update(kwargs)
        self.alpha = np.nan
        self.beta = np.nan
        self.electricity_power = np.nan
        self.machine_status = np.nan

    def effect_m_g(self, machine_statu):
        if machine_statu == 1:
            self.alpha = 0.06997
            self.beta = -0.9152
        else:
            self.alpha = 0
            self.beta = 0
        logging.error(
            "yida_turbine_model_v2.py turbine:%s, alpha:%s, beta:%s", 
            self.instance, self.alpha, self.beta
        )
    
    def electricity_func(self, steam_flow_in):
        self.electricity_power = self.alpha * steam_flow_in + self.beta
        logging.error("yida_turbine_model_v2.py turbine:%s, electricity_power:%s", self.instance, self.electricity_power)
    
    def calculate_machine_statu(self):
        self.machine_status = 1 if self.steam_flow_in > self.steam_flow_in_threshold else 0
        logging.error("yida_turbine_model_v2.py turbine:%s, machine_status:%s", self.instance, self.machine_status)


# ===================================================================================
# 
# ===================================================================================
def objective(args):
    """
    目标函数
    """
    hp_steam_dayprice, \
    electricity_price_ext, \
    hp_steam_current, \
    electricity_power_ext_current = args
    
    def obj(x):
        return hp_steam_dayprice * (hp_steam_current + x[0]) + electricity_price_ext * (electricity_power_ext_current + x[1]) * 1000

    return obj


def contraint(args):
    """
    约束条件
    (hp_steam_current + x[0])                       : 3台汽机高压蒸汽进汽量, t/h
    (electricity_power_ext_current + x[1])          : 生产车间外购电电功率, MWh
    (eturb_m1_electricity_generation_current + x[2]): #1 汽机自发电发电功率, MWh
    (eturb_m2_electricity_generation_current + x[3]): #2 汽机自发电发电功率, MWh
    (bturb_m1_electricity_generation_current + x[4]): #3 汽机自发电发电功率, MWh
    (eturb_m1_steam_flow_in_current + x[5])         : #1 汽机进汽量, t/h
    (eturb_m2_steam_flow_in_current + x[6])         : #2 汽机进汽量, t/h
    (bturb_m1_steam_flow_in_current + x[7])         : #3 汽机进汽量, t/h
    (eturb_m1_steam_flow_side_current + x[8])       : #1 汽机抽汽量, t/h
    (eturb_m2_steam_flow_side_current + x[9])       : #2 汽机抽汽量, t/h
    """
    steamflow_pred_avg, \
    electricity_power_pred_avg, \
    lp_steam_throtte, \
    eturb_m1_alpha_1, \
    eturb_m1_alpha_2, \
    eturb_m1_beta, \
    eturb_m2_alpha_1, \
    eturb_m2_alpha_2, \
    eturb_m2_beta, \
    bturb_m1_alpha, \
    bturb_m1_beta, \
    eturb_m1_in_min, \
    eturb_m1_in_max, \
    eturb_m2_in_min, \
    eturb_m2_in_max, \
    bturb_m1_in_min, \
    bturb_m1_in_max, \
    eturb_m1_out_min, \
    eturb_m1_out_max, \
    eturb_m2_out_min, \
    eturb_m2_out_max, \
    electricity_power_ext_max, \
    hp_steam_current, \
    eturb_m1_electricity_generation_current, \
    eturb_m2_electricity_generation_current, \
    bturb_m1_electricity_generation_current, \
    electricity_power_ext_current, \
    eturb_m1_steam_flow_in_current, \
    eturb_m2_steam_flow_in_current, \
    bturb_m1_steam_flow_in_current, \
    eturb_m1_steam_flow_side_current, \
    eturb_m2_steam_flow_side_current = args

    if electricity_power_ext_current > 0:
        electricity_power_ext_cons = (
            # 外购电电功率 > 0
            {"type": "ineq", "fun": lambda x: (electricity_power_ext_current + x[1]) - 0},
            # 外购电电功率<外购电最大电功率
            {"type": "ineq", "fun": lambda x: electricity_power_ext_max - (electricity_power_ext_current + x[1])},
        )
    else:
        electricity_power_ext_cons = (
            # 外购电电功率(卖电) < 0
            {"type": "ineq", "fun": lambda x: 0 - (electricity_power_ext_current + x[1])},
        )

    cons = (
        # -------------------------
        # 等式约束
        # -------------------------
        # 电力平衡
        {"type": "eq", "fun": lambda x: ((eturb_m1_electricity_generation_current + x[2]) + 
                                         (eturb_m2_electricity_generation_current + x[3]) + 
                                         (bturb_m1_electricity_generation_current + x[4])) + (electricity_power_ext_current + x[1]) - electricity_power_pred_avg},
        # 低压蒸汽平衡
        {"type": "eq", "fun": lambda x: ((bturb_m1_steam_flow_in_current + x[7]) + 
                                         (eturb_m1_steam_flow_side_current + x[8]) + 
                                         (eturb_m2_steam_flow_side_current + x[9])) + lp_steam_throtte - steamflow_pred_avg},
        # 高压蒸汽平衡
        {"type": "eq", "fun": lambda x: ((eturb_m1_steam_flow_in_current + x[5]) + 
                                         (eturb_m2_steam_flow_in_current + x[6]) + 
                                         (bturb_m1_steam_flow_in_current + x[7])) + lp_steam_throtte - (hp_steam_current + x[0])},
        # 1#汽机发电
        {"type": "eq", "fun": lambda x: eturb_m1_alpha_1 * x[5] + eturb_m1_alpha_2 * x[8] + eturb_m1_beta - x[2]},
        # 2#汽机发电
        {"type": "eq", "fun": lambda x: eturb_m2_alpha_1 * x[6] + eturb_m2_alpha_2 * x[9] + eturb_m2_beta - x[3]},
        # 3#汽机发电
        {"type": "eq", "fun": lambda x: bturb_m1_alpha * x[7] + bturb_m1_beta - x[4]},
        # -------------------------
        # 不等式约束
        # -------------------------
        # 1#汽轮发电机组进汽汽下限
        {"type": "ineq", "fun": lambda x: (eturb_m1_steam_flow_in_current + x[5]) - eturb_m1_in_min},
        # 1#汽轮发电机组进汽汽上限
        {"type": "ineq", "fun": lambda x: eturb_m1_in_max - (eturb_m1_steam_flow_in_current + x[5])},
        # 2#汽轮发电机组进汽汽下限
        {"type": "ineq", "fun": lambda x: (eturb_m2_steam_flow_in_current + x[6]) - eturb_m2_in_min},
        # 2#汽轮发电机组进汽汽上限
        {"type": "ineq", "fun": lambda x: eturb_m2_in_max - (eturb_m2_steam_flow_in_current + x[6])},
        # 3#汽轮发电机组进汽汽下限
        {"type": "ineq", "fun": lambda x: (bturb_m1_steam_flow_in_current + x[7]) - bturb_m1_in_min},
        # 3#汽轮发电机组进汽汽上限
        {"type": "ineq", "fun": lambda x: bturb_m1_in_max - (bturb_m1_steam_flow_in_current + x[7])},
        # 1#汽轮发电机组抽汽下限
        {"type": "ineq", "fun": lambda x: (eturb_m1_steam_flow_side_current + x[8]) - eturb_m1_out_min},
        # 1#汽轮发电机组抽汽上限
        {"type": "ineq", "fun": lambda x: eturb_m1_out_max - (eturb_m1_steam_flow_side_current + x[8])},
        # 2#汽轮发电机组抽汽下限
        {"type": "ineq", "fun": lambda x: (eturb_m2_steam_flow_side_current + x[9]) - eturb_m2_out_min},
        # 2#汽轮发电机组抽汽上限
        {"type": "ineq", "fun": lambda x: eturb_m2_out_max - (eturb_m2_steam_flow_side_current + x[9])},
        # 1#汽轮发电机组进汽-抽汽
        {"type": "ineq", "fun": lambda x: (eturb_m1_steam_flow_in_current + x[5]) - (eturb_m1_steam_flow_side_current + x[8]) - 20},
        # 2#汽轮发电机组进汽汽下限
        {"type": "ineq", "fun": lambda x: (eturb_m2_steam_flow_in_current + x[6]) - (eturb_m2_steam_flow_side_current + x[9]) - 20},
        # 1#汽机进汽单次调整量下限(减少进汽量)
        {"type": "ineq", "fun": lambda x: x[5] - (-TURBINE_STEAM_ADJUST_DELTA)},
        # 1#汽机进汽单次调整量上限(增加进汽量)
        {"type": "ineq", "fun": lambda x: TURBINE_STEAM_ADJUST_DELTA - x[5]},
        # 2#汽机进汽单次调整量下限(减少进汽量)
        {"type": "ineq", "fun": lambda x: x[6] - (-TURBINE_STEAM_ADJUST_DELTA)},
        # 2#汽机进汽单次调整量上限(增加进汽量)
        {"type": "ineq", "fun": lambda x: TURBINE_STEAM_ADJUST_DELTA - x[6]},
        # 3#汽机进汽单次调整量下限(减少进汽量)
        {"type": "ineq", "fun": lambda x: x[7] - (-TURBINE_STEAM_ADJUST_DELTA)},
        # 3#汽机进汽单次调整量上限(增加进汽量)
        {"type": "ineq", "fun": lambda x: TURBINE_STEAM_ADJUST_DELTA - x[7]},
        # 1#汽机抽汽单次调整量下限(减少进汽量)
        {"type": "ineq", "fun": lambda x: x[8] - (-TURBINE_STEAM_ADJUST_DELTA)},
        # 1#汽机抽汽单次调整量上限(增加进汽量)
        {"type": "ineq", "fun": lambda x: TURBINE_STEAM_ADJUST_DELTA - x[8]},
        # 2#汽机抽汽单次调整量下限(减少进汽量)
        {"type": "ineq", "fun": lambda x: x[9] - (-TURBINE_STEAM_ADJUST_DELTA)},
        # 2#汽机抽汽单次调整量上限(增加进汽量)
        {"type": "ineq", "fun": lambda x: TURBINE_STEAM_ADJUST_DELTA - x[9]}
    )
    cons = cons + electricity_power_ext_cons

    return cons


def optimizer(args_obj, args_con, x0):
    """
    目标函数优化器
    """
    cons = contraint(args_con)
    res = minimize(
        objective(args_obj),
        x0 = np.asarray(x0),
        method = "SLSQP",
        constraints = cons
    )

    return res


# ==========================================================================================================
# 汽轮发电机组负荷分配优化模型
# ==========================================================================================================
def create_turbine_instance(steam_flow_in_array):
    # 建立汽轮发电机组示例
    eturb_m1 = Eturb(
        instance = "eturb_m1",
        steam_flow_in = steam_flow_in_array[0],
        steam_flow_in_threshold = ETURB_M1_MACHINE_STATUS
    )
    eturb_m2 = Eturb(
        instance = "eturb_m2",
        steam_flow_in = steam_flow_in_array[1],
        steam_flow_in_threshold = ETURB_M2_MACHINE_STATUS
    )
    bturb_m1 = Bturb(
        instance = "bturb_m1",
        steam_flow_in = steam_flow_in_array[2],
        steam_flow_in_threshold = BTURB_M1_MACHINE_STATUS
    )
    # 判断汽机开停炉状态
    eturb_m1.calculate_machine_statu()
    eturb_m2.calculate_machine_statu()
    bturb_m1.calculate_machine_statu()
    
    return eturb_m1, eturb_m2, bturb_m1


def turbine_optimizer_main_model(
    hp_steam_dayprice,
    electricity_price_buy,
    electricity_price_sale,
    steamflow_pred_avg,
    electricity_power_pred_avg,
    lp_steam_throtte,
    steam_flow_in_array,
    steam_flow_side_array,
    electricity_generation_array,
    steam_in_upper_limit_array,
    steam_in_lower_limit_array,
    steam_out_upper_limit_array,
    steam_out_lower_limit_array,
    electricity_power_ext_max,
    electricity_power_ext):
    # ---------------------------------
    # 电价处理(外购电、外卖电)
    # ---------------------------------
    try:
        electricity_price_ext = electricity_price_ext_preprocessing(
            electricity_power_ext,
            electricity_price_buy,
            electricity_price_sale
        )
    except:
        traceback.print_exc()
    # ---------------------------------
    # 建立汽轮发电机组实例
    # ---------------------------------
    try:
        eturb_m1, eturb_m2, bturb_m1 = create_turbine_instance(steam_flow_in_array)
        # eturb_m1
        eturb_m1.steam_flow_side = steam_flow_side_array[0]
        eturb_m1.electricity_generation = electricity_generation_array[0]
        eturb_m1.machine_steam_in_upper_limit = steam_in_upper_limit_array[0] * eturb_m1.machine_status
        eturb_m1.machine_steam_in_lower_limit = steam_in_lower_limit_array[0] * eturb_m1.machine_status
        eturb_m1.machine_steam_ext_upper_limit = steam_out_upper_limit_array[0] * eturb_m1.machine_status
        eturb_m1.machine_steam_ext_lower_limit = steam_out_lower_limit_array[0] * eturb_m1.machine_status
        # eturb_m2
        eturb_m2.steam_flow_side = steam_flow_side_array[1]
        eturb_m2.electricity_generation = electricity_generation_array[1]
        eturb_m2.machine_steam_in_upper_limit = steam_in_upper_limit_array[1] * eturb_m2.machine_status
        eturb_m2.machine_steam_in_lower_limit = steam_in_lower_limit_array[1] * eturb_m2.machine_status
        eturb_m2.machine_steam_ext_upper_limit = steam_out_upper_limit_array[1] * eturb_m2.machine_status
        eturb_m2.machine_steam_ext_lower_limit = steam_out_lower_limit_array[1] * eturb_m2.machine_status
        # bturb_m1
        bturb_m1.electricity_generation = electricity_generation_array[2]
        bturb_m1.machine_steam_in_upper_limit = steam_in_upper_limit_array[2] * bturb_m1.machine_status
        bturb_m1.machine_steam_in_lower_limit = steam_in_lower_limit_array[2] * bturb_m1.machine_status
        # 估计汽轮发电机组综合效率
        eturb_m1.effect_m_g(eturb_m1.machine_status)
        eturb_m2.effect_m_g(eturb_m2.machine_status)
        bturb_m1.effect_m_g(bturb_m1.machine_status)
        hp_steam = eturb_m1.steam_flow_in + eturb_m2.steam_flow_in + bturb_m1.steam_flow_in
    except:
        traceback.print_exc()
    # ---------------------------------
    # 构造参数
    # ---------------------------------
    try:
        # 目标变量参数
        args_obj = (
            hp_steam_dayprice,      # 元/t
            electricity_price_ext,  # 元/kWh
            hp_steam,               # t/h
            electricity_power_ext   # MW
        )
        # 约束条件参数
        args_con = (
            steamflow_pred_avg,
            electricity_power_pred_avg,
            lp_steam_throtte,
            eturb_m1.alpha_1,
            eturb_m1.alpha_2,
            eturb_m1.beta,
            eturb_m2.alpha_1,
            eturb_m2.alpha_2,
            eturb_m2.beta,
            bturb_m1.alpha,
            bturb_m1.beta,
            eturb_m1.machine_steam_in_lower_limit,
            eturb_m1.machine_steam_in_upper_limit,
            eturb_m2.machine_steam_in_lower_limit,
            eturb_m2.machine_steam_in_upper_limit,
            bturb_m1.machine_steam_in_lower_limit,
            bturb_m1.machine_steam_in_upper_limit,
            eturb_m1.machine_steam_ext_lower_limit,
            eturb_m1.machine_steam_ext_upper_limit,
            eturb_m2.machine_steam_ext_lower_limit,
            eturb_m2.machine_steam_ext_upper_limit,
            electricity_power_ext_max,
            hp_steam,
            eturb_m1.electricity_generation,
            eturb_m2.electricity_generation,
            bturb_m1.electricity_generation,
            electricity_power_ext,
            eturb_m1.steam_flow_in,
            eturb_m2.steam_flow_in,
            bturb_m1.steam_flow_in,
            eturb_m1.steam_flow_side,
            eturb_m2.steam_flow_side
        )
        # 决策变量初值
        x0 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    except:
        traceback.print_exc()
    # ---------------------------------
    # 优化函数
    # ---------------------------------
    try:
        result = optimizer(args_obj, args_con, x0)
        object_value_min = result.fun
        status = result.success
        optim_steam = result.x
        logging.error("=======================================")
        logging.error("yida_turbine_model_v2.py optim_results:")
        logging.error("=======================================")
        logging.error(f"优化得到的目标函数最小值={object_value_min}")
        logging.error(f"优化状态={status}")
        logging.error(f"优化路径={optim_steam}")
        logging.error(f"锅炉高压蒸汽产汽量={optim_steam[0]}")
        logging.error(f"生产车间外购电电功率={optim_steam[1]}")
        logging.error(f"汽机1自发电发电功率={optim_steam[2]}")
        logging.error(f"汽机2自发电发电功率={optim_steam[3]}")
        logging.error(f"汽机3自发电发电功率={optim_steam[4]}")
        logging.error(f"汽机1进汽量={optim_steam[5]}")
        logging.error(f"汽机2进汽量={optim_steam[6]}")
        logging.error(f"汽机3进汽量={optim_steam[7]}")
        logging.error(f"汽机1抽汽量={optim_steam[8]}")
        logging.error(f"汽机2抽汽量={optim_steam[9]}")
        # --------------------------------------------------
        # 负荷分配优化结果
        # --------------------------------------------------
        eturb_m1_steam_flow_in_delta = optim_steam[5]
        eturb_m2_steam_flow_in_delta = optim_steam[6]
        bturb_m1_steam_flow_in_delta = optim_steam[7]
        # --------------------------------------------------
        # if status:
        # --------------------------------------------------
        # 建议进汽、抽汽量
        eturb_m1_hp_steam_machine_opt = optim_steam[5] + eturb_m1.steam_flow_in
        eturb_m2_hp_steam_machine_opt = optim_steam[6] + eturb_m2.steam_flow_in
        bturb_m1_hp_steam_machine_opt = optim_steam[7] + bturb_m1.steam_flow_in
        eturb_m1_lp_steam_machine_opt = optim_steam[8] + eturb_m1.steam_flow_side
        eturb_m2_lp_steam_machine_opt = optim_steam[9] + eturb_m2.steam_flow_side
        # 建议发电量计算
        eturb_m1_electricity_machine_opt = optim_steam[2] + eturb_m1.electricity_generation
        eturb_m2_electricity_machine_opt = optim_steam[3] + eturb_m2.electricity_generation
        bturb_m1_electricity_machine_opt = optim_steam[4] + bturb_m1.electricity_generation
        # --------------------------------------------------
        # else:
        #     total_cost_optim = hp_steam_dayprice * hp_steam + electricity_price_ext * (electricity_power_ext * 1000)
        #     logging.error("优化得到的目标函数最小值2:%s", total_cost_optim)
        #     eturb_m1_hp_steam_machine_opt = eturb_m1.steam_flow_in
        #     eturb_m2_hp_steam_machine_opt = eturb_m2.steam_flow_in
        #     bturb_m1_hp_steam_machine_opt = bturb_m1.steam_flow_in
        #     eturb_m1_lp_steam_machine_opt = eturb_m1.steam_flow_side
        #     eturb_m2_lp_steam_machine_opt = eturb_m2.steam_flow_side
        #     eturb_m1_electricity_machine_opt = eturb_m1.electricity_generation
        #     eturb_m2_electricity_machine_opt = eturb_m2.electricity_generation
        #     bturb_m1_electricity_machine_opt = bturb_m1.electricity_generation
        # --------------------------------------------------
        logging.error("=======================================")
        logging.error("yida_turbine_model_v2.py cost_params:")
        logging.error("=======================================")
        logging.error({
            "hp_steam_dayprice": hp_steam_dayprice,
            "electricity_price_ext": electricity_price_ext,
            "hp_steam": hp_steam,
            "electricity_power_ext": electricity_power_ext,
        })
        total_cost_optim = object_value_min
        logging.error("优化得到的目标函数最小值1:%s", total_cost_optim)
        object_value_min2 = hp_steam_dayprice * (hp_steam + optim_steam[5] + optim_steam[6] + optim_steam[7]) + electricity_price_ext * ((electricity_power_ext + optim_steam[1]) * 1000)
        logging.error("优化得到的目标函数最小值2:%s", object_value_min2)
    except:
        traceback.print_exc()
    # --------------------------
    # result
    # --------------------------
    eturb_m1_result = {
        "hp_steam_machine_opt": eturb_m1_hp_steam_machine_opt,       # 抽凝汽轮发电机组进汽量优化值
        "lp_steam_machine_opt": eturb_m1_lp_steam_machine_opt,       # 抽凝汽轮发电机组抽汽量优化值
        "electricity_machine_opt": eturb_m1_electricity_machine_opt, # 抽凝汽轮发电机组发电功率
    }
    eturb_m2_result = {
        "hp_steam_machine_opt": eturb_m2_hp_steam_machine_opt,       # 抽凝汽轮发电机组进汽量优化值
        "lp_steam_machine_opt": eturb_m2_lp_steam_machine_opt,       # 抽凝汽轮发电机组抽汽量优化值
        "electricity_machine_opt": eturb_m2_electricity_machine_opt, # 抽凝汽轮发电机组发电功率
    }
    bturb_m1_result = {
        "hp_steam_machine_opt": bturb_m1_hp_steam_machine_opt,       # 背压汽轮发电机组进汽量优化值
        "electricity_machine_opt": bturb_m1_electricity_machine_opt, # 背压汽轮发电机组发电功率
    }
    steampipeline_result = {
        "total_cost_optim": total_cost_optim
    }
    
    return eturb_m1_result, \
           eturb_m2_result, \
           bturb_m1_result, \
           steampipeline_result, \
           eturb_m1_steam_flow_in_delta, \
           eturb_m2_steam_flow_in_delta, \
           bturb_m1_steam_flow_in_delta











if __name__ == "__main__":
    # =================================
    # 峰电、买电
    # =================================
    eturb_m1_result, \
    eturb_m2_result, \
    bturb_m1_result, \
    steampipeline_result, \
    eturb_m1_steam_flow_in_delta, \
    eturb_m2_steam_flow_in_delta, \
    bturb_m1_steam_flow_in_delta = turbine_optimizer_main_model(
        hp_steam_dayprice = 95.788,
        electricity_price_buy = 0.879,
        electricity_price_sale = 0.397,
        steamflow_pred_avg = 58.1,
        electricity_power_pred_avg = 18.8,
        lp_steam_throtte = 0,
        steam_flow_in_array = [0, 0, 0],
        steam_flow_side_array = [0, 0],
        electricity_generation_array = [0, 0, 0],
        steam_in_upper_limit_array = [80, 80, 75],
        steam_in_lower_limit_array = [70, 70, 20],
        steam_out_upper_limit_array = [15, 30],
        steam_out_lower_limit_array = [15, 40],
        electricity_power_ext_max = 8,  # 4
        electricity_power_ext = 2.6
    )

    eturb_m1_result, \
    eturb_m2_result, \
    bturb_m1_result, \
    steampipeline_result, \
    eturb_m1_steam_flow_in_delta, \
    eturb_m2_steam_flow_in_delta, \
    bturb_m1_steam_flow_in_delta = turbine_optimizer_main_model(
        hp_steam_dayprice = 95.788,
        electricity_price_buy = 0.879,
        electricity_price_sale = 0.397,
        steamflow_pred_avg = 58.1,
        electricity_power_pred_avg = 18.8,
        lp_steam_throtte = 0,
        steam_flow_in_array = [0, 0, 0],
        steam_flow_side_array = [0, 0],
        electricity_generation_array = [0, 0, 0],
        steam_in_upper_limit_array = [80, 80, 75],
        steam_in_lower_limit_array = [70, 70, 20],
        steam_out_upper_limit_array = [15, 30],
        steam_out_lower_limit_array = [15, 40],
        electricity_power_ext_max = 8,  # 4
        electricity_power_ext = 2.6
    )
    # =================================
    # 平电、买电
    # =================================
    eturb_m1_result, \
    eturb_m2_result, \
    bturb_m1_result, \
    steampipeline_result, \
    eturb_m1_steam_flow_in_delta, \
    eturb_m2_steam_flow_in_delta, \
    bturb_m1_steam_flow_in_delta = turbine_optimizer_main_model(
        hp_steam_dayprice = 95.788,
        electricity_price_buy = 0.543,
        electricity_price_sale = 0.397,
        steamflow_pred_avg = 58.1,
        electricity_power_pred_avg = 18.8,
        lp_steam_throtte = 0,
        steam_flow_in_array = [0, 0, 0],
        steam_flow_side_array = [0, 0],
        electricity_generation_array = [0, 0, 0],
        steam_in_upper_limit_array = [80, 80, 75],
        steam_in_lower_limit_array = [70, 70, 20],
        steam_out_upper_limit_array = [15, 30],
        steam_out_lower_limit_array = [15, 40],
        electricity_power_ext_max = 8,  # 4
        electricity_power_ext = 2.6
    )

    eturb_m1_result, \
    eturb_m2_result, \
    bturb_m1_result, \
    steampipeline_result, \
    eturb_m1_steam_flow_in_delta, \
    eturb_m2_steam_flow_in_delta, \
    bturb_m1_steam_flow_in_delta = turbine_optimizer_main_model(
        hp_steam_dayprice = 95.788,
        electricity_price_buy = 0.543,
        electricity_price_sale = 0.397,
        steamflow_pred_avg = 58.1,
        electricity_power_pred_avg = 18.8,
        lp_steam_throtte = 0,
        steam_flow_in_array = [0, 0, 0],
        steam_flow_side_array = [0, 0],
        electricity_generation_array = [0, 0, 0],
        steam_in_upper_limit_array = [80, 80, 75],
        steam_in_lower_limit_array = [70, 70, 20],
        steam_out_upper_limit_array = [15, 30],
        steam_out_lower_limit_array = [15, 40],
        electricity_power_ext_max = 8,  # 4
        electricity_power_ext = 2.6
    )
    # =================================
    # 谷电、买电
    # =================================
    eturb_m1_result, \
    eturb_m2_result, \
    bturb_m1_result, \
    steampipeline_result, \
    eturb_m1_steam_flow_in_delta, \
    eturb_m2_steam_flow_in_delta, \
    bturb_m1_steam_flow_in_delta = turbine_optimizer_main_model(
        hp_steam_dayprice = 95.788,
        electricity_price_buy = 0.284,
        electricity_price_sale = 0.397,
        steamflow_pred_avg = 58.1,
        electricity_power_pred_avg = 18.8,
        lp_steam_throtte = 0,
        steam_flow_in_array = [0, 0, 0],
        steam_flow_side_array = [0, 0],
        electricity_generation_array = [0, 0, 0],
        steam_in_upper_limit_array = [80, 80, 75],
        steam_in_lower_limit_array = [70, 70, 20],
        steam_out_upper_limit_array = [15, 30],
        steam_out_lower_limit_array = [15, 40],
        electricity_power_ext_max = 8,  # 4
        electricity_power_ext = 2.6
    )

    eturb_m1_result, \
    eturb_m2_result, \
    bturb_m1_result, \
    steampipeline_result, \
    eturb_m1_steam_flow_in_delta, \
    eturb_m2_steam_flow_in_delta, \
    bturb_m1_steam_flow_in_delta = turbine_optimizer_main_model(
        hp_steam_dayprice = 95.788,
        electricity_price_buy = 0.284,
        electricity_price_sale = 0.397,
        steamflow_pred_avg = 58.1,
        electricity_power_pred_avg = 18.8,
        lp_steam_throtte = 0,
        steam_flow_in_array = [0, 0, 0],
        steam_flow_side_array = [0, 0],
        electricity_generation_array = [0, 0, 0],
        steam_in_upper_limit_array = [80, 80, 75],
        steam_in_lower_limit_array = [70, 70, 20],
        steam_out_upper_limit_array = [15, 30],
        steam_out_lower_limit_array = [15, 40],
        electricity_power_ext_max = 8,  # 4
        electricity_power_ext = 2.6
    )

    # =================================
    # 卖电
    # =================================
    eturb_m1_result, \
    eturb_m2_result, \
    bturb_m1_result, \
    steampipeline_result, \
    eturb_m1_steam_flow_in_delta, \
    eturb_m2_steam_flow_in_delta, \
    bturb_m1_steam_flow_in_delta = turbine_optimizer_main_model(
        hp_steam_dayprice = 95.788,
        electricity_price_buy = 0.879,
        electricity_price_sale = 0.397,
        steamflow_pred_avg = 58.1,
        electricity_power_pred_avg = 18.8,
        lp_steam_throtte = 0,
        steam_flow_in_array = [0, 0, 0],
        steam_flow_side_array = [0, 0],
        electricity_generation_array = [0, 0, 0],
        steam_in_upper_limit_array = [80, 80, 75],
        steam_in_lower_limit_array = [70, 70, 20],
        steam_out_upper_limit_array = [15, 30],
        steam_out_lower_limit_array = [15, 40],
        electricity_power_ext_max = 8,  # 4
        electricity_power_ext = -2.6
    )

    eturb_m1_result, \
    eturb_m2_result, \
    bturb_m1_result, \
    steampipeline_result, \
    eturb_m1_steam_flow_in_delta, \
    eturb_m2_steam_flow_in_delta, \
    bturb_m1_steam_flow_in_delta = turbine_optimizer_main_model(
        hp_steam_dayprice = 95.788,
        electricity_price_buy = 0.879,
        electricity_price_sale = 0.397,
        steamflow_pred_avg = 58.1,
        electricity_power_pred_avg = 18.8,
        lp_steam_throtte = 0,
        steam_flow_in_array = [0, 0, 0],
        steam_flow_side_array = [0, 0],
        electricity_generation_array = [0, 0, 0],
        steam_in_upper_limit_array = [80, 80, 75],
        steam_in_lower_limit_array = [70, 70, 20],
        steam_out_upper_limit_array = [15, 30],
        steam_out_lower_limit_array = [15, 40],
        electricity_power_ext_max = 8,  # 4
        electricity_power_ext = -2.6
    )


