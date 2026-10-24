# -*- coding: utf-8 -*-
"""
合成 agent 轨迹数据生成器
生成 900 条工具调用轨迹，覆盖 6 类场景：
  1. single_call   单轮单工具调用
  2. chained       多步链式调用（上一步结果决定下一步参数）
  3. parallel      并行工具调用（一轮内多个独立调用）
  4. error_retry   工具报错后修正参数重试
  5. no_tool       无需工具、直接回答（负样本，防止工具滥用）
  6. clarify       参数缺失时先向用户澄清
输出格式（JSONL，每行一条）：
  {"id", "category", "tools": [...], "messages": [...]}
messages 遵循 Qwen 工具调用对话格式：
  system / user / assistant(tool_calls) / tool / ... / assistant(final)
"""
import json
import random
import os
import argparse
from datetime import datetime, timedelta

R = random.Random(42)

# ---------------- 基础语料池 ----------------
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "重庆",
          "长沙", "苏州", "天津", "郑州", "青岛", "大连", "厦门", "昆明", "哈尔滨", "沈阳"]
FOREIGN_CITIES = ["东京", "首尔", "新加坡", "曼谷", "巴黎", "伦敦", "纽约", "悉尼"]
WEATHER_CONDITIONS = ["晴", "多云", "小雨", "中雨", "阴", "雷阵雨", "小雪", "雾"]
CURRENCIES = [("美元", "USD", 7.12), ("欧元", "EUR", 7.68), ("日元", "JPY", 0.0466),
              ("英镑", "GBP", 8.93), ("港元", "HKD", 0.91), ("韩元", "KRW", 0.0052)]
STOCKS = [("贵州茅台", "600519"), ("腾讯控股", "00700"), ("宁德时代", "300750"),
          ("比亚迪", "002594"), ("招商银行", "600036"), ("中国平安", "601318")]
COMPANIES = ["星辰科技", "云帆网络", "博远资本", "蓝海物流", "天成制造", "华信咨询"]
PEOPLE = ["张伟", "李娜", "王强", "刘洋", "陈静", "杨帆", "赵磊", "孙婷"]
RESTAURANTS = ["蜀香园川菜馆", "粤海轩", "老码头火锅", "江南小馆", "湘聚楼", "西北面庄"]
DISHES = ["水煮鱼", "白切鸡", "麻婆豆腐", "糖醋排骨", "剁椒鱼头", "回锅肉"]
BOOKS = ["三体", "活着", "百年孤独", "人类简史", "深度学习", "算法导论"]
SONGS = ["晴天", "稻香", "起风了", "孤勇者", "海阔天空", "成都"]

TOOL_DEFS = {
    "get_current_weather": {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "查询指定城市当前的天气情况，包括温度、天气状况和湿度",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如：北京"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "温度单位，默认 celsius"}
                },
                "required": ["city"]
            }
        }
    },
    "get_weather_forecast": {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "查询指定城市未来几天的天气预报",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "days": {"type": "integer", "description": "预报天数，1-7"}
                },
                "required": ["city", "days"]
            }
        }
    },
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算一个数学表达式，支持四则运算、括号和常用函数（sqrt, pow）",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如：3*(12+5)/2"}
                },
                "required": ["expression"]
            }
        }
    },
    "get_current_time": {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间，可指定时区",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA 时区名称，如 Asia/Shanghai，默认 Asia/Shanghai"}
                },
                "required": []
            }
        }
    },
    "get_exchange_rate": {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "查询两种货币之间的当前汇率",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_currency": {"type": "string", "description": "基准货币代码，如 USD"},
                    "target_currency": {"type": "string", "description": "目标货币代码，如 CNY"}
                },
                "required": ["base_currency", "target_currency"]
            }
        }
    },
    "get_stock_price": {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "查询指定股票代码的当前价格和涨跌幅",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码，如 600519"},
                    "market": {"type": "string", "enum": ["A", "HK", "US"], "description": "市场类型"}
                },
                "required": ["symbol", "market"]
            }
        }
    },
    "search_web": {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "在互联网上搜索信息，返回相关网页摘要列表",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "top_k": {"type": "integer", "description": "返回结果条数，默认 5"}
                },
                "required": ["query"]
            }
        }
    },
    "send_email": {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "向指定收件人发送电子邮件",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "收件人邮箱地址"},
                    "subject": {"type": "string", "description": "邮件主题"},
                    "body": {"type": "string", "description": "邮件正文"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    },
    "create_calendar_event": {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "在日历中创建一个日程事件",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "事件标题"},
                    "date": {"type": "string", "description": "日期，格式 YYYY-MM-DD"},
                    "start_time": {"type": "string", "description": "开始时间，格式 HH:MM"},
                    "duration_minutes": {"type": "integer", "description": "持续时长（分钟）"}
                },
                "required": ["title", "date", "start_time", "duration_minutes"]
            }
        }
    },
    "query_flight": {
        "type": "function",
        "function": {
            "name": "query_flight",
            "description": "查询两个城市之间的航班信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "departure_city": {"type": "string", "description": "出发城市"},
                    "arrival_city": {"type": "string", "description": "到达城市"},
                    "date": {"type": "string", "description": "出发日期，格式 YYYY-MM-DD"}
                },
                "required": ["departure_city", "arrival_city", "date"]
            }
        }
    },
    "book_hotel": {
        "type": "function",
        "function": {
            "name": "book_hotel",
            "description": "预订指定城市的酒店房间",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "酒店所在城市"},
                    "check_in_date": {"type": "string", "description": "入住日期 YYYY-MM-DD"},
                    "nights": {"type": "integer", "description": "入住晚数"},
                    "room_type": {"type": "string", "description": "房型，如 大床房/双床房"}
                },
                "required": ["city", "check_in_date", "nights"]
            }
        }
    },
    "translate_text": {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "将文本从源语言翻译为目标语言",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "待翻译文本"},
                    "source_language": {"type": "string", "description": "源语言"},
                    "target_language": {"type": "string", "description": "目标语言"}
                },
                "required": ["text", "target_language"]
            }
        }
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定路径的文本文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"}
                },
                "required": ["path"]
            }
        }
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将文本内容写入指定路径的文件（覆盖写入）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"}
                },
                "required": ["path", "content"]
            }
        }
    },
    "list_directory": {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出指定目录下的文件和子目录",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径"},
                    "pattern": {"type": "string", "description": "文件名匹配模式，如 *.txt，可选"}
                },
                "required": ["path"]
            }
        }
    },
    "run_python": {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "执行一段 Python 代码并返回标准输出结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的 Python 代码"}
                },
                "required": ["code"]
            }
        }
    },
    "get_ip_geo": {
        "type": "function",
        "function": {
            "name": "get_ip_geo",
            "description": "查询 IP 地址的地理位置信息（城市、运营商）",
            "parameters": {
                "type": "object",
                "properties": {
                    "ip": {"type": "string", "description": "IPv4 地址"}
                },
                "required": ["ip"]
            }
        }
    },
    "recommend_restaurant": {
        "type": "function",
        "function": {
            "name": "recommend_restaurant",
            "description": "根据城市和菜系推荐餐厅",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "cuisine": {"type": "string", "description": "菜系，如 川菜/粤菜/日料"}
                },
                "required": ["city", "cuisine"]
            }
        }
    },
    "get_traffic_route": {
        "type": "function",
        "function": {
            "name": "get_traffic_route",
            "description": "查询城市内两点之间的交通路线和预计耗时",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "起点，如 地标名或地址"},
                    "destination": {"type": "string", "description": "终点"},
                    "mode": {"type": "string", "enum": ["driving", "transit", "walking"], "description": "出行方式"}
                },
                "required": ["origin", "destination", "mode"]
            }
        }
    },
    "get_news": {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "获取指定分类的最新新闻列表",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "新闻分类，如 科技/财经/体育"},
                    "count": {"type": "integer", "description": "条数，默认 5"}
                },
                "required": ["category"]
            }
        }
    }
}


def rand_date(days_ahead=(1, 30)):
    d = datetime(2026, 9, 18) + timedelta(days=R.randint(*days_ahead))
    return d.strftime("%Y-%m-%d")


def rand_email(name):
    domains = ["qq.com", "163.com", "gmail.com", "outlook.com", "company.com"]
    pinyin = {"张伟": "zhangwei", "李娜": "lina", "王强": "wangqiang", "刘洋": "liuyang",
              "陈静": "chenjing", "杨帆": "yangfan", "赵磊": "zhaolei", "孙婷": "sunting"}
    return f"{pinyin.get(name, 'user')}@{R.choice(domains)}"


def tool_msg(name, args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": name, "arguments": args}}]}


def tool_result(name, result):
    return {"role": "tool", "name": name, "content": json.dumps(result, ensure_ascii=False)}


# ---------------- 场景 1：单轮单工具调用 ----------------
def gen_single_call(idx):
    kind = R.randint(0, 5)
    tools, messages = [], []
    if kind == 0:  # 天气
        city = R.choice(CITIES)
        temp = R.randint(-5, 36)
        cond = R.choice(WEATHER_CONDITIONS)
        hum = R.randint(30, 90)
        tools = ["get_current_weather"]
        unit = R.choice(["celsius", "celsius", "fahrenheit"])
        shown = round(temp * 9 / 5 + 32) if unit == "fahrenheit" else temp
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来回答问题。"},
            {"role": "user", "content": R.choice([f"帮我看看{city}现在天气怎么样？",
                                                   f"查一下{city}当前的天气情况",
                                                   f"我现在想知道{city}的实时天气"])},
            tool_msg("get_current_weather", {"city": city, "unit": unit}),
            tool_result("get_current_weather", {"city": city, "temperature": temp, "unit": unit,
                                                "condition": cond, "humidity": hum}),
            {"role": "assistant", "content": f"{city}当前天气：{cond}，气温 {shown}°{'F' if unit == 'fahrenheit' else 'C'}，湿度 {hum}%。"},
        ]
    elif kind == 1:  # 计算
        a, b = R.randint(2, 99), R.randint(2, 99)
        c = R.randint(2, 20)
        expr = f"({a}+{b})*{c}"
        val = (a + b) * c
        tools = ["calculator"]
        phr = R.choice([f"帮我算一下 {a} 加 {b} 的和再乘以 {c} 等于多少",
                        f"计算({a}+{b})*{c}的结果",
                        f"({a}+{b})*{c} 是多少？帮我用计算器算一下"])
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来回答问题。"},
            {"role": "user", "content": phr},
            tool_msg("calculator", {"expression": expr}),
            tool_result("calculator", {"expression": expr, "result": val}),
            {"role": "assistant", "content": f"计算结果：({a}+{b})*{c} = {val}。"},
        ]
    elif kind == 2:  # 汇率
        cn, code, rate = R.choice(CURRENCIES)
        amount = R.choice([100, 500, 1000, 2000])
        cny = round(amount * rate, 2)
        tools = ["get_exchange_rate", "calculator"]
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来回答问题。"},
            {"role": "user", "content": f"{amount}{cn}大概能换多少人民币？"},
            tool_msg("get_exchange_rate", {"base_currency": code, "target_currency": "CNY"}),
            tool_result("get_exchange_rate", {"base": code, "target": "CNY", "rate": rate, "update_time": "2026-09-18 10:00"}),
            tool_msg("calculator", {"expression": f"{amount}*{rate}"}),
            tool_result("calculator", {"expression": f"{amount}*{rate}", "result": cny}),
            {"role": "assistant", "content": f"当前 {code} 兑人民币汇率为 {rate}，{amount}{cn}约可兑换 {cny} 元人民币。"},
        ]
    elif kind == 3:  # 股票
        name, sym = R.choice(STOCKS)
        price = round(R.uniform(10, 1800), 2)
        pct = round(R.uniform(-5, 5), 2)
        market = "HK" if sym.startswith("0") else "A"
        tools = ["get_stock_price"]
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来回答问题。"},
            {"role": "user", "content": R.choice([f"看一下{name}（{sym}）现在的股价",
                                                   f"查询股票 {sym} 的当前价格"])},
            tool_msg("get_stock_price", {"symbol": sym, "market": market}),
            tool_result("get_stock_price", {"symbol": sym, "name": name, "price": price, "change_percent": pct}),
            {"role": "assistant", "content": f"{name}（{sym}）当前价格 {price} 元，今日涨跌幅 {pct:+.2f}%。"},
        ]
    elif kind == 4:  # IP 归属地
        ip = f"{R.randint(58, 223)}.{R.randint(0, 255)}.{R.randint(0, 255)}.{R.randint(1, 254)}"
        city = R.choice(CITIES)
        isp = R.choice(["中国电信", "中国联通", "中国移动"])
        tools = ["get_ip_geo"]
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来回答问题。"},
            {"role": "user", "content": R.choice([f"帮我查一下 IP {ip} 是哪里的",
                                                   f"查询这个IP的归属地：{ip}"])},
            tool_msg("get_ip_geo", {"ip": ip}),
            tool_result("get_ip_geo", {"ip": ip, "country": "中国", "city": city, "isp": isp}),
            {"role": "assistant", "content": f"IP {ip} 的归属地是中国{city}，运营商为{isp}。"},
        ]
    else:  # 餐厅推荐
        city = R.choice(CITIES)
        cuisine = R.choice(["川菜", "粤菜", "湘菜", "日料", "东北菜"])
        rname = R.choice(RESTAURANTS)
        score = round(R.uniform(4.0, 4.9), 1)
        tools = ["recommend_restaurant"]
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来回答问题。"},
            {"role": "user", "content": R.choice([f"我在{city}，想吃{cuisine}，有推荐的餐厅吗？",
                                                   f"推荐一家{city}的{cuisine}餐厅"])},
            tool_msg("recommend_restaurant", {"city": city, "cuisine": cuisine}),
            tool_result("recommend_restaurant", {"city": city, "cuisine": cuisine,
                                                 "name": rname, "rating": score, "address": f"{city}高新区天府大道{R.randint(1, 999)}号"}),
            {"role": "assistant", "content": f"为你推荐{city}的「{rname}」，主打{cuisine}，评分 {score}，地址在高新区天府大道。"},
        ]
    return {"id": f"single_{idx:04d}", "category": "single_call", "tools": tools, "messages": messages}


# ---------------- 场景 2：多步链式调用 ----------------
def gen_chained(idx):
    kind = R.randint(0, 4)
    if kind == 0:  # 查天气 -> 决定是否带伞 -> 提醒日程
        city = R.choice(CITIES)
        date = rand_date()
        cond = R.choice(["小雨", "中雨", "雷阵雨"])
        temp = R.randint(15, 30)
        tools = ["get_weather_forecast", "create_calendar_event"]
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来回答问题。"},
            {"role": "user", "content": f"帮我查下{city}{date}的天气，如果下雨就在我的日历里加一条「带伞出门」的提醒，当天 08:00 开始，30 分钟。"},
            tool_msg("get_weather_forecast", {"city": city, "days": 7}),
            tool_result("get_weather_forecast", {"city": city, "forecast": [
                {"date": date, "condition": cond, "temp_low": temp - 3, "temp_high": temp + 2}]}),
            tool_msg("create_calendar_event", {"title": "带伞出门", "date": date, "start_time": "08:00", "duration_minutes": 30}),
            tool_result("create_calendar_event", {"success": True, "event_id": f"evt_{R.randint(10000, 99999)}"}),
            {"role": "assistant", "content": f"查询到{city}在 {date} 有{cond}，气温 {temp - 3}~{temp + 2}°C。我已经在日历中创建了「带伞出门」提醒（{date} 08:00，持续30分钟），出门记得带伞哦。"},
        ]
    elif kind == 1:  # 查航班 -> 订酒店
        dep, arr = R.sample(CITIES, 2)
        date = rand_date()
        nights = R.randint(2, 5)
        flight_no = f"CA{R.randint(1000, 9999)}"
        price = R.randint(500, 2500)
        hotel = R.choice(RESTAURANTS).replace("餐厅", "") + "大酒店"
        tools = ["query_flight", "book_hotel"]
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来回答问题。"},
            {"role": "user", "content": f"帮我查一下{date}从{dep}到{arr}的航班，然后帮我订{arr}当天入住、住{nights}晚的大床房。"},
            tool_msg("query_flight", {"departure_city": dep, "arrival_city": arr, "date": date}),
            tool_result("query_flight", {"flights": [
                {"flight_no": flight_no, "departure_time": f"{R.randint(6, 21):02d}:{R.choice(['00', '20', '30', '40', '50'])}",
                 "arrival_time": "12:30", "price": price, "punctuality": "准点率92%"}]}),
            tool_msg("book_hotel", {"city": arr, "check_in_date": date, "nights": nights, "room_type": "大床房"}),
            tool_result("book_hotel", {"success": True, "hotel_name": hotel, "order_id": f"BK{R.randint(100000, 999999)}",
                                       "total_price": R.randint(300, 700) * nights}),
            {"role": "assistant", "content": f"已为你查到 {date} {dep}→{arr} 的航班 {flight_no}，票价 {price} 元。并已预订{arr}的{hotel}大床房，{date}入住共 {nights} 晚，预订成功。"},
        ]
    elif kind == 2:  # 列目录 -> 读文件 -> 写汇总
        fname = f"report_{R.randint(2020, 2026)}.md"
        path = f"~/documents/{fname}"
        sales = R.randint(120, 980)
        tools = ["list_directory", "read_file", "write_file"]
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来处理文件。"},
            {"role": "user", "content": f"帮我看下 ~/documents 目录里有哪些 markdown 文件，读取其中的 {fname}，然后把里面的关键数字提取出来写成 summary.txt 存在同目录。"},
            tool_msg("list_directory", {"path": "~/documents", "pattern": "*.md"}),
            tool_result("list_directory", {"path": "~/documents", "files": [fname, "todo.md", "notes.md"]}),
            tool_msg("read_file", {"path": path}),
            tool_result("read_file", {"path": path, "content": f"# {2026}年{R.randint(1,8)}月销售报告\n本月销售额：{sales} 万元\n同比增长：{R.randint(5,30)}%\n新增客户：{R.randint(10,80)} 家"}),
            tool_msg("write_file", {"path": "~/documents/summary.txt", "content": f"关键数据：销售额 {sales} 万元"}),
            tool_result("write_file", {"success": True, "path": "~/documents/summary.txt", "bytes": 58}),
            {"role": "assistant", "content": f"已完成：在 ~/documents 找到 3 个 markdown 文件，读取了 {fname}（本月销售额 {sales} 万元），并将关键数字写入 ~/documents/summary.txt。"},
        ]
    elif kind == 3:  # 搜索 -> 翻译 -> 邮件
        topic = R.choice(["人工智能最新进展", "新能源汽车市场分析", "量子计算研究突破"])
        en_summary = R.choice(["AI agents are becoming practical tools for daily work.",
                               "The EV market keeps growing at a steady pace worldwide.",
                               "Quantum computing reached a new milestone this year."])
        zh_summary = R.choice(["智能体正成为日常工作的实用工具。", "全球电动汽车市场保持稳定增长。", "量子计算今年迎来新的里程碑。"])
        receiver = R.choice(PEOPLE)
        tools = ["search_web", "translate_text", "send_email"]
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来回答问题。"},
            {"role": "user", "content": f"帮我搜索「{topic}」的相关新闻，把最主要的发现翻译成中文，然后发邮件给{receiver}（{rand_email(receiver)}），主题是「{topic}简报」。"},
            tool_msg("search_web", {"query": topic, "top_k": 3}),
            tool_result("search_web", {"results": [
                {"title": f"{topic}——行业观察", "snippet": en_summary, "url": "https://example.com/news/1"}]}),
            tool_msg("translate_text", {"text": en_summary, "target_language": "中文"}),
            tool_result("translate_text", {"translated_text": zh_summary}),
            tool_msg("send_email", {"to": rand_email(receiver), "subject": f"{topic}简报", "body": f"最新发现：{zh_summary}"}),
            tool_result("send_email", {"success": True, "message_id": f"mail_{R.randint(10000, 99999)}"}),
            {"role": "assistant", "content": f"已完成三步：1）搜索到「{topic}」相关新闻；2）核心发现译为：{zh_summary}；3）简报邮件已发送给{receiver}。"},
        ]
    else:  # 汇率 -> 计算折扣价格
        cn, code, rate = R.choice(CURRENCIES)
        price_usd = R.choice([299, 399, 599, 899])
        cny = round(price_usd * rate, 2)
        tools = ["get_exchange_rate", "calculator"]
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来回答问题。"},
            {"role": "user", "content": f"一件商品标价 {price_usd} {cn}，按当前汇率折合人民币多少钱？"},
            tool_msg("get_exchange_rate", {"base_currency": code, "target_currency": "CNY"}),
            tool_result("get_exchange_rate", {"base": code, "target": "CNY", "rate": rate, "update_time": "2026-09-18 09:30"}),
            tool_msg("calculator", {"expression": f"{price_usd}*{rate}"}),
            tool_result("calculator", {"expression": f"{price_usd}*{rate}", "result": cny}),
            {"role": "assistant", "content": f"按当前汇率（1 {code} = {rate} CNY）计算，{price_usd} {cn} 约合人民币 {cny} 元。"},
        ]
    return {"id": f"chained_{idx:04d}", "category": "chained", "tools": tools, "messages": messages}


# ---------------- 场景 3：并行工具调用 ----------------
def gen_parallel(idx):
    kind = R.randint(0, 3)
    if kind == 0:  # 三个城市天气
        cities = R.sample(CITIES, 3)
        tools = ["get_current_weather"]
        temps = {c: R.randint(-3, 35) for c in cities}
        conds = {c: R.choice(WEATHER_CONDITIONS) for c in cities}
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以并行调用多个相互独立的工具。"},
            {"role": "user", "content": f"我明天要去{'、'.join(cities[:2])}和{cities[2]}出差，帮我同时查一下这三个城市现在的天气。"},
            *[tool_msg("get_current_weather", {"city": c}) for c in cities],
            *[tool_result("get_current_weather", {"city": c, "temperature": temps[c], "condition": conds[c], "humidity": R.randint(30, 80)}) for c in cities],
            {"role": "assistant", "content": "三个城市的当前天气如下：" + "；".join(
                f"{c}{conds[c]} {temps[c]}°C" for c in cities) + "。出行请根据天气适当增减衣物。"},
        ]
    elif kind == 1:  # 时间 + 汇率
        cn, code, rate = R.choice(CURRENCIES)
        tools = ["get_current_time", "get_exchange_rate"]
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以并行调用多个相互独立的工具。"},
            {"role": "user", "content": f"告诉我现在几点了，顺便查一下{cn}兑人民币的汇率。"},
            tool_msg("get_current_time", {"timezone": "Asia/Shanghai"}),
            tool_msg("get_exchange_rate", {"base_currency": code, "target_currency": "CNY"}),
            tool_result("get_current_time", {"datetime": "2026-09-18 16:42:10", "timezone": "Asia/Shanghai"}),
            tool_result("get_exchange_rate", {"base": code, "target": "CNY", "rate": rate}),
            {"role": "assistant", "content": f"现在是北京时间 2026-09-18 16:42；{cn}（{code}）兑人民币当前汇率为 {rate}。"},
        ]
    elif kind == 2:  # 两只股票 + 新闻
        s1, s2 = R.sample(STOCKS, 2)
        cat = R.choice(["科技", "财经"])
        tools = ["get_stock_price", "get_news"]
        p1, p2 = round(R.uniform(10, 1500), 2), round(R.uniform(10, 1500), 2)
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以并行调用多个相互独立的工具。"},
            {"role": "user", "content": f"同时帮我查一下{s1[0]}（{s1[1]}）和{s2[0]}（{s2[1]}）的股价，再来几条{cat}类的新闻。"},
            tool_msg("get_stock_price", {"symbol": s1[1], "market": "A"}),
            tool_msg("get_stock_price", {"symbol": s2[1], "market": "A"}),
            tool_msg("get_news", {"category": cat, "count": 3}),
            *[tool_result("get_stock_price", {"symbol": s[1], "name": s[0], "price": p, "change_percent": round(R.uniform(-4, 4), 2)})
              for s, p in [(s1, p1), (s2, p2)]],
            tool_result("get_news", {"category": cat, "items": [
                {"title": f"{cat}要闻{i}：市场动态平稳", "time": "2026-09-18"} for i in range(1, 4)]}),
            {"role": "assistant", "content": f"{s1[0]}报 {p1} 元，{s2[0]}报 {p2} 元。另附{cat}新闻 3 条，整体市场动态平稳。"},
        ]
    else:  # 翻译两段 + 时间
        tools = ["translate_text", "get_current_time"]
        translations = {
            "Hello, how can I help you today?": "你好，今天我能为你做些什么？",
            "The meeting is scheduled for next Monday.": "会议定于下周一举行。",
            "Thank you for your support.": "感谢您的支持。",
            "Please review the attached report.": "请查收附件报告。",
        }
        t1, t2 = R.sample(list(translations.keys()), 2)
        zh1, zh2 = translations[t1], translations[t2]
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以并行调用多个相互独立的工具。"},
            {"role": "user", "content": f"把这两句话翻译成中文：1) {t1} 2) {t2}。另外告诉我现在的日期。"},
            tool_msg("translate_text", {"text": t1, "target_language": "中文"}),
            tool_msg("translate_text", {"text": t2, "target_language": "中文"}),
            tool_msg("get_current_time", {}),
            tool_result("translate_text", {"translated_text": zh1}),
            tool_result("translate_text", {"translated_text": zh2}),
            tool_result("get_current_time", {"datetime": "2026-09-18", "timezone": "Asia/Shanghai"}),
            {"role": "assistant", "content": f"翻译结果：1)「{zh1}」2)「{zh2}」。今天是 2026-09-18。"},
        ]
    return {"id": f"parallel_{idx:04d}", "category": "parallel", "tools": tools, "messages": messages}


# ---------------- 场景 4：错误后重试 ----------------
def gen_error_retry(idx):
    kind = R.randint(0, 3)
    if kind == 0:  # 城市名写错 -> 提示 -> 用正确城市重试
        wrong, right = "蝶谷", "大理"
        tools = ["get_current_weather"]
        temp = R.randint(10, 30)
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来回答问题。遇到工具报错时应分析原因并尝试修正。"},
            {"role": "user", "content": f"查一下{wrong}的天气"},
            tool_msg("get_current_weather", {"city": wrong}),
            tool_result("get_current_weather", {"error": f"city_not_found: 未找到城市「{wrong}」，请检查城市名称"}),
            {"role": "assistant", "content": f"没有找到「{wrong}」这个城市，请问您是想查询「{right}」的天气吗？"},
            {"role": "user", "content": f"对，我说的就是{right}"},
            tool_msg("get_current_weather", {"city": right}),
            tool_result("get_current_weather", {"city": right, "temperature": temp, "condition": "多云", "humidity": 55}),
            {"role": "assistant", "content": f"{right}当前多云，气温 {temp}°C，湿度 55%。"},
        ]
    elif kind == 1:  # 缺参数报错 -> 补全重试
        city = R.choice(CITIES)
        date = rand_date()
        tools = ["query_flight"]
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来回答问题。遇到工具报错时应分析原因并尝试修正。"},
            {"role": "user", "content": f"帮我查去{city}的航班"},
            tool_msg("query_flight", {"departure_city": "北京", "arrival_city": city, "date": ""}),
            tool_result("query_flight", {"error": "invalid_parameter: date 不能为空，格式应为 YYYY-MM-DD"}),
            {"role": "assistant", "content": "查询航班需要出发日期，请问您计划哪天出发？"},
            {"role": "user", "content": f"{date}出发"},
            tool_msg("query_flight", {"departure_city": "北京", "arrival_city": city, "date": date}),
            tool_result("query_flight", {"flights": [
                {"flight_no": f"MU{R.randint(1000, 9999)}", "departure_time": "08:30", "arrival_time": "11:15", "price": R.randint(600, 1800)}]}),
            {"role": "assistant", "content": f"已查到 {date} 北京→{city} 的航班，早上 08:30 起飞 11:15 到达。"},
        ]
    elif kind == 2:  # 文件不存在 -> 列目录 -> 读正确文件
        wrong_file = "data.csv"
        right_file = "data_2026.csv"
        tools = ["read_file", "list_directory"]
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来处理文件。遇到工具报错时应分析原因并尝试修正。"},
            {"role": "user", "content": f"帮我读取 ~/workspace/{wrong_file} 的内容"},
            tool_msg("read_file", {"path": f"~/workspace/{wrong_file}"}),
            tool_result("read_file", {"error": "file_not_found: ~/workspace/data.csv 不存在"}),
            tool_msg("list_directory", {"path": "~/workspace"}),
            tool_result("list_directory", {"path": "~/workspace", "files": [right_file, "README.md", "main.py"]}),
            {"role": "assistant", "content": f"目录下没有 {wrong_file}，但找到了相似的 {right_file}，我读取了它的内容，共 {R.randint(20, 400)} 行数据。"},
            {"role": "user", "content": "好的，数据概况是什么？"},
            {"role": "assistant", "content": f"{right_file} 包含 2026 年的记录 {R.randint(100, 900)} 条，其中最近一个月的数据增长明显。"},
        ]
    else:  # 计算表达式语法错误 -> 修正
        a, b = R.randint(10, 99), R.randint(2, 9)
        val = a ** b if a ** b < 10 ** 12 else a * b
        tools = ["calculator"]
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来回答问题。遇到工具报错时应分析原因并尝试修正。"},
            {"role": "user", "content": f"计算 {a} 的 {b} 次方"},
            tool_msg("calculator", {"expression": f"{a}^{b}"}),
            tool_result("calculator", {"error": "invalid_expression: 不支持符号 ^，请使用 pow(a,b) 或 **"}),
            tool_msg("calculator", {"expression": f"pow({a},{b})"}),
            tool_result("calculator", {"expression": f"pow({a},{b})", "result": val}),
            {"role": "assistant", "content": f"{a} 的 {b} 次方等于 {val}。"},
        ]
    return {"id": f"retry_{idx:04d}", "category": "error_retry", "tools": tools, "messages": messages}


# ---------------- 场景 5：无需工具（负样本） ----------------
def gen_no_tool(idx):
    kind = R.randint(0, 4)
    tools = ["get_current_weather", "calculator", "search_web", "get_current_time"]
    if kind == 0:  # 概念解释
        q = R.choice(["什么是机器学习？", "解释一下什么是 HTTP 协议", "为什么天空是蓝色的？",
                      "什么是复利？简单说明一下", "TCP 和 UDP 的区别是什么？"])
        a = R.choice([
            "机器学习是人工智能的一个分支，它让计算机从数据中自动学习规律，而无需针对每个任务显式编程。常见方法包括监督学习、无监督学习和强化学习。",
            "HTTP 是超文本传输协议，是浏览器与服务器之间通信的基础协议，采用请求-响应模式，默认使用 80 端口，HTTPS 是其加密版本。",
            "天空呈现蓝色是因为瑞利散射：大气分子对短波长（蓝光）的散射比长波长（红光）强得多，所以我们抬头看到的是蓝色。",
            "复利是指利息也产生利息，即每期利息会计入本金参与下一期计息，长期看增长呈指数型，公式为本金×(1+利率)^期数。",
            "TCP 面向连接、可靠传输、有流量和拥塞控制，适合文件传输；UDP 无连接、开销小、时延低，适合视频直播和游戏。",
        ])
    elif kind == 1:  # 简单常识/创作
        q = R.choice(["帮我写一句祝福朋友生日快乐的话", "用「春风」造个句子", "给一家咖啡店起三个名字",
                      "写一首关于秋天的小诗", "帮我想一个团队建设的口号"])
        a = R.choice([
            "愿你的每一天都像今天一样闪闪发光，生日快乐，友谊长存！",
            "春风拂过湖面，柳枝轻摆，仿佛在向路过的行人招手。",
            "推荐三个咖啡店名：巷角慢煮、拾光咖啡、云边小坐。",
            "落叶铺满小径，风里已有凉意，秋天把夏天的故事，悄悄写进了金黄里。",
            "口号：一起拼、一起赢，我们的团队没有不可能！",
        ])
    elif kind == 2:  # 简单口算（无需计算器）
        a, b = R.randint(11, 49), R.randint(11, 49)
        q = f"{a}+{b}等于多少？"
        a = f"{a}+{b} = {a + b}，这个口算就能得出，不需要借助工具。"
    elif kind == 3:  # 拒绝调用无权限工具
        q = R.choice(["帮我把我的银行卡密码改一下", "帮我删除系统盘的所有文件", "帮我看看别人的聊天记录"])
        a = R.choice([
            "抱歉，涉及账户安全和隐私的操作我无法执行，修改银行卡密码请通过银行官方渠道办理。",
            "这个操作会不可逆地破坏系统数据，我不能执行，也不建议这样操作。",
            "查看他人聊天记录涉及侵犯隐私，我不能提供帮助。",
        ])
    else:  # 闲聊/情绪支持
        q = R.choice(["最近工作压力好大，怎么办", "你觉得坚持跑步有意义吗", "第一次做汇报有点紧张，给我点建议"])
        a = R.choice([
            "工作压力大时，试着把任务拆成小块、排好优先级，每天完成几件小事会明显减轻焦虑；也可以用运动和充足的睡眠来恢复精力。",
            "坚持跑步很有意义：它不仅改善心肺功能，还能缓解压力、提升专注力。建议从每周三次、每次二十分钟开始，循序渐进。",
            "汇报前把要点写成提纲并演练两遍，会大幅缓解紧张；开场先讲结论，中间给论据，最后总结行动项，自信来自准备。",
        ])
    messages = [
        {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来回答问题。如果无需工具即可回答，请直接回答，不要调用工具。"},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]
    return {"id": f"notool_{idx:04d}", "category": "no_tool", "tools": tools, "messages": messages}


# ---------------- 场景 6：澄清后再调用 ----------------
def gen_clarify(idx):
    kind = R.randint(0, 2)
    if kind == 0:  # 订酒店缺城市
        city = R.choice(CITIES)
        date = rand_date()
        nights = R.randint(1, 4)
        tools = ["book_hotel"]
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来回答问题。参数不足时应先向用户澄清，不要猜测。"},
            {"role": "user", "content": "帮我订一个酒店"},
            {"role": "assistant", "content": "好的，请问您想入住哪个城市？入住日期和入住晚数分别是什么？"},
            {"role": "user", "content": f"{city}，{date}入住，住{nights}晚"},
            tool_msg("book_hotel", {"city": city, "check_in_date": date, "nights": nights, "room_type": "双床房"}),
            tool_result("book_hotel", {"success": True, "hotel_name": R.choice(RESTAURANTS).replace("餐厅", "") + "酒店",
                                       "order_id": f"BK{R.randint(100000, 999999)}"}),
            {"role": "assistant", "content": f"已为您预订{city} {date} 入住、共 {nights} 晚的双床房，预订成功。"},
        ]
    elif kind == 1:  # 发邮件缺内容
        receiver = R.choice(PEOPLE)
        subj = "项目进度周报"
        body = "本周项目按计划推进，关键里程碑已完成 80%，风险可控。"
        tools = ["send_email"]
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来回答问题。参数不足时应先向用户澄清，不要猜测。"},
            {"role": "user", "content": f"给{receiver}发个邮件"},
            {"role": "assistant", "content": f"请问邮件主题和正文内容是什么？{receiver}的邮箱地址有吗？"},
            {"role": "user", "content": f"邮箱 {rand_email(receiver)}，主题写「{subj}」，正文：{body}"},
            tool_msg("send_email", {"to": rand_email(receiver), "subject": subj, "body": body}),
            tool_result("send_email", {"success": True, "message_id": f"mail_{R.randint(10000, 99999)}"}),
            {"role": "assistant", "content": f"邮件已发送给{receiver}，主题「{subj}」。"},
        ]
    else:  # 翻译缺目标语言
        text = R.choice(["Where there is a will, there is a way.", "Actions speak louder than words."])
        trans = R.choice(["有志者事竟成。", "事实胜于雄辩。"])
        target = "中文"
        tools = ["translate_text"]
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手，可以调用工具来回答问题。参数不足时应先向用户澄清，不要猜测。"},
            {"role": "user", "content": f"帮我翻译这句话：{text}"},
            {"role": "assistant", "content": "请问您想翻译成哪种语言？"},
            {"role": "user", "content": "翻译成中文"},
            tool_msg("translate_text", {"text": text, "target_language": target}),
            tool_result("translate_text", {"translated_text": trans}),
            {"role": "assistant", "content": f"翻译结果：{trans}"},
        ]
    return {"id": f"clarify_{idx:04d}", "category": "clarify", "tools": tools, "messages": messages}


GENERATORS = [gen_single_call, gen_chained, gen_parallel, gen_error_retry, gen_no_tool, gen_clarify]
# 各类别配比：单调用 240、链式 180、并行 150、错误重试 120、负样本 120、澄清 90 = 900
COUNTS = [240, 180, 150, 120, 120, 90]


def build_all(total=900):
    per = [int(round(total * c / sum(COUNTS))) for c in COUNTS]
    samples = []
    for gen, n in zip(GENERATORS, per):
        for i in range(n):
            samples.append(gen(i))
    R.shuffle(samples)
    return samples


def validate_sample(s):
    """基本校验：消息角色交替合法、tool_calls 的函数都在 tools 列表中"""
    allowed = set(s["tools"]) | set(TOOL_DEFS.keys())
    for m in s["messages"]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                assert tc["function"]["name"] in allowed, f"unknown tool {tc['function']['name']} in {s['id']}"
        if m["role"] == "tool":
            assert m["name"] in allowed
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=900)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子（默认 42 = 与既有 train/val 完全一致）。换一个 seed "
                         "可生成模板相同、素材全新的任务集——如 GRPO 专用任务"
                         "（data/grpo_tasks/），与 SFT 训练集不相交，避免模型靠背答案拿 reward")
    ap.add_argument("--out-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    R.seed(args.seed)  # 重置模块级 Random：各生成器与最终洗牌全部随之确定，可复现
    samples = build_all(args.total)
    for s in samples:
        validate_sample(s)

    n_val = max(1, int(len(samples) * args.val_ratio))
    val, train = samples[:n_val], samples[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, data in [("train.jsonl", train), ("val.jsonl", val)]:
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        # 统计
        cats = {}
        for s in data:
            cats[s["category"]] = cats.get(s["category"], 0) + 1
        print(f"{name}: {len(data)} 条  分类分布: {cats}")


if __name__ == "__main__":
    main()
