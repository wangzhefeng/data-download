
## 数据下载方法


### 1.概述

    * 本文档是对数据下载脚本 `data_download.py` 的使用说明
    * `data_download.py` 的作用是模拟平台提供的 API 功能进行数据批量下载
    * 主要亮点功能：
        - 能够同时下载多个 `class` 、多个 `instance`、多种 `resolution`、多个 metric 的 dev环境、生产环境数据

### 2.项目目录说明

    1. 新建配置文件目录及配置文件:
        - `/your_path/config/project_config/`
            - `config.csv`
            - `Class1_resolution_metrics.csv`
            - `Class2_resolution_metrics.csv`
    2. 新建数据下载目录:
        - `/your_path/result/`
    3. 配置好数据下载脚本参数:
        - `/your_path/data_download.py`

### 3.生产环境的特殊性(以溢达项目为例)

    * 需要启动本地电脑代理

### 4.填写配置文件

#### 配置文件 `/your_path/config/config.csv`

主要字段如下：

* `class`：类名
* `geo`：要下载 metirc 所属的 `geo` (代表一个instance)
* `metric_file`：数据下载文件名(`Class_resolution_metrics.csv`)
* `resolution`：要下载数据的 resolution

溢达项目下载 3 台锅炉 `resolution=1min` 数据的示例：

|class     |geo                                 |metric_file                |resolution  |
|----------|------------------------------------|---------------------------|------------|
|`CFBoiler`|`ydgm_e1|hpp_f1|boiler_w1|boiler_m1`|`CFBoiler_1min_metrics.csv`|`1min `     |
|`CFBoiler`|`ydgm_e1|hpp_f1|boiler_w1|boiler_m1`|`CFBoiler_1min_metrics.csv`|`1min `     |
|`CFBoiler`|`ydgm_e1|hpp_f1|boiler_w1|boiler_m1`|`CFBoiler_1min_metrics.csv`|`1min `     |

#### 配置文件 `/your_path/config/Class_resolution_metrics.csv`

主要字段及其形式：

* `metric_list`
    - `Domain.metric`

溢达项目下载 3 台锅炉 resolution=1min 数据的示例：


| metric_list                 |
|-----------------------------|
| CFBoilerOP.outlet_steam_flow|
| CFBoilerOP.feed_water_flow  |

### 5.修改数据下载脚本中的参数

以煤气项目 dev 环境为例:

```python
# 项目配置
url_domain = "dev.yo-i.com.cn:8443"
project_scopes = "energy/gas"
cookie = "thingswise.web.proxy.session_id=s%3ABbuVqLYF98fuSmX3B9M7DAe77Fc1nkJH.1FJtGevRaRi9WvjOymc03zV6%2FDe8DCUT55AbZ9zCi6s; __guid=165663402.1502813834321714700.1590023534580.6956; thingswise.web.app.session_id=s%3AOdDY10p1wYEoyXPqxxI5rezN.J8b3vEoIeHd0c9y6VDjx6MIrEo%2BJ2iAqf1SXUQRocgQ; monitor_count=65"

# 查询时间
from_timestamp = 1596988800000  # 2020-08-10 00:00:00
to_timestamp = 1597507200000    # 2020-08-16 00:00:00

# 需要下载的类
class_to_be_download = ["Converter", "BFGasPipeline"]

# 生成下载数据配置参数
parameters = generate_config(
    url_domain, 
    project_scopes, 
    cookie, 
    from_timestamp, 
    to_timestamp, 
    class_to_be_download
)

# 下载数据
integrate_data(parameters)
```

其中：

* `url_domain`：是项目所在的域名，dev 环境都是 `dev.yo-i.com.cn:8443`，具体项目具体分析
* `project_scopes`：是项目建立时定义的项目 project 名称，项目 scope 名称，这两项位于紧跟域名之后的两项 `https://url_domain/project/scope`
* `cookie`: 一般下载数据时需要针对具体的项目 API 进行下载，所以需要使用账号密码登陆到项目中，所以数据下载是需要设置浏览器中的 cookie，cookie 的查找可以查看网络教程，本文档会列出一些常用项目的 cookie，请自便
* `from_timestamp`：与通过 API 下载数据时的方法一样
* `to_timestamp`：与通过 API 下载数据时的方法一样
* `class_to_be_download`：指需要下载的 `class`，这里的 `class` 已经在 `config.csv` 配置文件中已经配置好


### 6.启动下载脚本

直接运行脚本即可。

### 7.下载数据

下载的数据在同目录下的 result 中，下载的数据是按照每个 instance 一个 csv 文件组织的。



