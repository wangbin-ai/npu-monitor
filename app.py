from flask import Flask, render_template, jsonify
import base64
import requests
import time
import pprint
import pandas as pd
import re
import traceback
import threading

app = Flask(__name__)


APPID = "com.noah.pangu.rl"
TOKEN = "xxxx"

# Cache configuration
CACHE_EXPIRE = 300  # seconds (5 minutes)

# Thread-safe cache state
_cache_lock = threading.Lock()
cached_user_data = {}
cached_spec_data = {}
last_update_time = 0

from pypinyin import lazy_pinyin

# 读取Excel文件
df = pd.read_excel('算法卡池先导用卡分配.xlsx', sheet_name='能力项用卡名单')

# 获取能力项列名（跳过第一列的用卡信息和最后两列）
capability_columns = df.columns[1:-2].tolist()

# 构建结果字典
usr_dict = {}
usr_name_dict = {}


def get_first_letter(text):
    text = str(text).strip()
    if not text:
        return ''
    first_char = text[0]
    if '\u4e00' <= first_char <= '\u9fff':
        return lazy_pinyin(first_char)[0][0].lower()
    else:
        return first_char.lower()


def extract_id(text):
    match = re.search(r'\d+', str(text))
    if match:
        return match.group()
    return ''


for col in capability_columns:
    leader = str(df[col].iloc[0]).strip() if pd.notna(df[col].iloc[0]) else ''

    for member in df[col].iloc[1:]:
        if pd.notna(member) and str(member).strip() != 'nan':
            member_str = str(member).strip()
            if member_str and member_str != 'sum':
                member_id = extract_id(member_str)
                first_letter = get_first_letter(member_str)
                key = f'{first_letter}{member_id}' if member_id else first_letter
                usr_dict[key] = leader
                usr_name_dict[key] = member_str
                # 同时用纯工号存储，兼容API返回的纯数字userId
                if member_id:
                    usr_dict[member_id] = leader
                    usr_name_dict[member_id] = member_str


def fetch_gpu_data():
    """从API获取并处理GPU使用数据"""
    print("开始获取GPU数据...")
    url = "https://roma.huawei.com/csb/rest/saas/ei/eiWizard/train/job/list?"
    print(f"usr_dict有{len(usr_dict)}个条目")

    params = {
        "appid": APPID,
        "vendor": "HEC",
        "region": "cn-southwest-2",
        "trainApiVersion": "V2",
    }

    filter_param = b'{\n\t"pageSize":"500",\n\t"pageIndex":"0",\n\t"status":"8",\n\t"searchName":"",\n\t"filterParam":[\n\t\t{\n\t\t\t"key":"",\n\t\t\t"value":""\n\t\t}\n\t],\n\t"tagIds":[\n\t\t""\n\t]\n}'
    encode_params = base64.b64encode(filter_param)
    params["params"] = encode_params.decode("utf-8")

    headers = {
        "content-type": "application/json;charset=UTF-8",
        "csb-token": TOKEN,
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return count_gpu_usage(data.get("trainJobs", []))
        else:
            print(f"请求失败，状态码：{response.status_code}")
            return None, None
    except Exception as e:
        print(f"获取数据异常：{e}")
        traceback.print_exc()
        return None, None


def count_gpu_usage(task_list):
    """统计用户和资源池的GPU使用情况，按leader->member->tasks层级组织"""
    try:
        leader_data = {}
        spec_gpu = {}

        for task in task_list:
            if task.get("statusCode") != "8":
                continue

            user_id = task.get("userId")
            spec_name = task.get("specName")

            gpu_num = task.get("workingGpuNum", 0)
            duration_str = task.get("duration", "0:0:0")
            duration = int(duration_str.split(':')[0]) if duration_str and ':' in duration_str else 0
            task_name = task.get('name')

            # 获取用户全名 - 兼容多种key格式查询
            user_name = (usr_name_dict.get(user_id) or
                         usr_name_dict.get(user_id.lstrip(user_id[0]) if user_id and user_id[0].isalpha() else None) or
                         user_id)

            # 获取所属leader - 兼容多种key格式查询
            leader_name = (usr_dict.get(user_id) or
                           usr_dict.get(user_id.lstrip(user_id[0]) if user_id and user_id[0].isalpha() else None) or
                           user_name)

            task_json = {
                'user': user_name,
                'gpu_num': gpu_num,
                'duration': duration,
                'task_name': task_name,
                'spec_name': spec_name
            }

            # 按leader组织数据
            if leader_name not in leader_data:
                leader_data[leader_name] = {
                    'gpu_num': 0,
                    'task_count': 0,
                    'total_duration': 0,
                    'max_duration': 0,
                    'members': {}
                }

            leader_data[leader_name]['gpu_num'] += gpu_num
            leader_data[leader_name]['task_count'] += 1
            leader_data[leader_name]['total_duration'] += duration

            # 按组员组织任务
            if user_name not in leader_data[leader_name]['members']:
                leader_data[leader_name]['members'][user_name] = {
                    'gpu_num': 0,
                    'task_count': 0,
                    'total_duration': 0,
                    'max_duration': 0,
                    'tasks': []
                }

            leader_data[leader_name]['members'][user_name]['gpu_num'] += gpu_num
            leader_data[leader_name]['members'][user_name]['task_count'] += 1
            leader_data[leader_name]['members'][user_name]['total_duration'] += duration
            if duration > leader_data[leader_name]['members'][user_name]['max_duration']:
                leader_data[leader_name]['members'][user_name]['max_duration'] = duration
            leader_data[leader_name]['members'][user_name]['tasks'].append(task_json)

            # 统计资源池
            if spec_name in spec_gpu:
                spec_gpu[spec_name] += gpu_num
            else:
                spec_gpu[spec_name] = gpu_num

        # 计算每个leader的最长任务时长（即其成员中最长任务时长的最大值）
        for leader_name in leader_data:
            max_member_duration = 0
            for member_name in leader_data[leader_name]['members']:
                member_max_duration = leader_data[leader_name]['members'][member_name]['max_duration']
                if member_max_duration > max_member_duration:
                    max_member_duration = member_max_duration
            leader_data[leader_name]['max_duration'] = max_member_duration

        pprint.pprint(leader_data)
        pprint.pprint(spec_gpu)
        return leader_data, spec_gpu
    except Exception as e:
        print(f"统计数据异常：{e}")
        traceback.print_exc()
        return None, None


def get_cached_data():
    """Return cached GPU data, refreshing if expired. Thread-safe."""
    global cached_user_data, cached_spec_data, last_update_time

    with _cache_lock:
        current_time = time.time()
        age = current_time - last_update_time
        if age > CACHE_EXPIRE or not cached_user_data:
            print(f'缓存已过期（{age:.0f}s），重新获取数据...')
            user_data, spec_data = fetch_gpu_data()
            if user_data and spec_data:
                cached_user_data = user_data
                cached_spec_data = spec_data
                last_update_time = current_time
            else:
                print('获取数据失败，继续使用旧缓存。')
        else:
            print(f'使用缓存数据（{age:.0f}s / {CACHE_EXPIRE}s）...')

        return cached_user_data, cached_spec_data, last_update_time


@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index_multi.html')


@app.route('/data')
def get_data():
    """提供数据接口，支持缓存"""
    user_data, spec_data, update_time = get_cached_data()

    return jsonify({
        'user_data': user_data,
        'spec_data': spec_data,
        'update_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(update_time))
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5063, debug=False)
