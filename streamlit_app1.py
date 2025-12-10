# -*- coding: utf-8 -*-
"""
大学生数据素养测评系统
一键运行：  run streamlit_app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from matplotlib import font_manager

# ----------------- 全局初始化 -----------------
# 设置中文字体路径，确保在云环境中也能使用
font_path = './SimHei.ttf'  # 假设字体文件与脚本在相同目录下
prop = font_manager.FontProperties(fname=font_path)

# 让 Streamlit 在 Docker/无桌面环境也能找得到中文字体
matplotlib.rcParams['font.sans-serif'] = [prop.get_name()]  # 使用 SimHei 字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ----------------- 数据层 -----------------
# 一级指标与权重（已做归一，总和=1）
C = pd.DataFrame({
    '维度': ['C1:数据认知与采集', 'C2:数据处理与分析', 'C3:数据存储与验证',
             'C4:数据表达与交流', 'C5:数据践行', 'C6:数据道德'],
    '权重': [0.3339, 0.2361, 0.1214, 0.1214, 0.1157, 0.0715]
})
C['权重'] = C['权重'] / C['权重'].sum()  # 再次保险归一

# 题库
题库 = {
    'C1': {
        'Question': [
            '01. 在不同阶段，能够清晰分辨自己的数据需求并将其明确表述',
            '02. 在学习工作中，养成通过数据方法解决问题的基本习惯',
            '03. 对数据价值有较高的敏感性并能抓取数据背后含义、过滤无用数据',
            '04. 对元数据等数据相关概念有一定理解，有较深的数学、统计学知识储备',
            '05. 具备基本的数据检索知识和能力，掌握基本数据检索方法（布尔逻辑算法、关键词换组等）和搜索引擎使用方法，能准确识别数据源',
            '06. 可以使用大于等于一种的数据采集工具（如爬虫软件）',
            '07. 能够通过关联字段筛选提取所需数据，并能简单使用数据库提取'
        ],
        'Score': [1.9, 1.5, 1.8, 3.2, 12.0, 7.5, 5.6]
    },
    'C2': {
        'Question': [
            '08. 能较为熟练地使用数据清洗、分类、转变和取值等方法处理数据',
            '09. 能够及时对可疑数据进行核对，对残缺丢失的数据进行修补、恢复，判断"脏数据"中的无用数据进行删除',
            '10. 可以通过一定的算法完成对数据的计算',
            '11. 最少熟练使用一种数据处理与分析工具并能了解多种数据分析工具（如EXCEL、SPSS、Matlab）',
            '12. 关注重要数据、养成记忆数据的习惯，具备大数据思维和基本的数据分析的思维，能分析出数据背后的含义',
            '13. 较为准确客观地完成对得出的数据结论的解读',
            '14. 根据数据处理分析的结论来完成所需作品'
        ],
        'Score': [4.8, 2.7, 4.7, 3.9, 2.1, 2.6, 2.9]
    },
    'C3': {
        'Question': [
            '15. 可以使用不同数据库对数据进行分类保存，使用硬盘、U盘等硬件存储或者百度云盘等设备存储数据',
            '16. 具备基本的数据安全保护意识，随时备份，及时辨别数据环境的安全情况，使用杀毒软件等工具保护自己的数据隐私',
            '17. 能对手中存储的数据进行统一归档、分类、标注',
            '18. 以批判思维对各流程数据，客观公正地评价数据分析成果',
            '19. 能对各流程所得出的结论进行有效校对和测试'
        ],
        'Score': [2.5, 3.4, 3.8, 1.4, 1.1]
    },
    'C4': {
        'Question': [
            '20. 使用可视化软件（PPT等）以图表等形式展现得出的成果',
            '21. 能概括数据分析后的核心观点、成果，并以数据化语言表述',
            '22. 能使用数据分析处理后的成果，撰写工作报告或学术论文',
            '23. 通过不同媒介分享数据成果，以数据的形式与其他主体交流'
        ],
        'Score': [2.4, 2.4, 6.3, 1.0]
    },
    'C5': {
        'Question': [
            '24. 对项目有深刻的理解，完成问题量化定义，掌握项目各阶段的数据工作',
            '25. 针对不同的问题进行差异性数据流程和方法组合，利用数据构造产出成果框架和内涵，并通过产出成果与需求的匹配，进行成果优化',
            '26. 用言简意赅的数据结论和便于理解的方式（比喻、举例等）与业务相关方沟通',
            '27. 在业务理解基础上，以数据意见形式推动业务落地转化为具体成果'
        ],
        'Score': [3.8, 1.9, 1.7, 4.2]
    },
    'C6': {
        'Question': [
            '28. 能够重视和保护相关各方的数据隐私',
            '29. 了解相关的数据安全、知识产权法规，严格遵守知识产权法',
            '30. 有严格的数据自律性，不随意篡改数据，能以正确的方式引用和使用数据'
        ],
        'Score': [3.5, 1.4, 2.2]
    }
}


# ----------------- 数据存储 -----------------
def load_user_data():
    """加载用户数据"""
    if os.path.exists('user_data.json'):
        with open('user_data.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_user_data(user_data):
    """保存用户数据"""
    with open('user_data.json', 'w', encoding='utf-8') as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)


def add_user_record(user_info, scores, score_rates, total_score):
    """添加用户记录"""
    user_data = load_user_data()

    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'user_info': user_info,
        'scores': scores,
        'score_rates': score_rates,
        'total_score': total_score,
        'dimension_names': C['维度'].tolist()
    }

    user_data.append(record)
    save_user_data(user_data)
    return record


# ----------------- 工具函数 -----------------
@st.cache_data(show_spinner=False)
def calc_scores(all_answers):
    """返回：综合得分、各维度得分、各维度得分率(%)、明细表"""
    scores, score_rates, detail = [], [], []
    for i, code in enumerate(题库.keys()):
        full = np.array(题库[code]['Score'])
        ans = np.array(all_answers[i])
        if len(ans) != len(full):
            st.error(f"维度 {code} 未答完")
            st.stop()
        got = ans * full / 6
        scores.append(got.sum())
        score_rates.append(got.sum() / full.sum() * 100)
        for q, a, s, m in zip(题库[code]['Question'], ans, got, full):
            detail.append({'维度': C.loc[i, '维度'], '问题': q, '评分': a, '得分': round(s, 2), '满分': m})
    total = np.dot(scores, C['权重'].values)
    return total, scores, score_rates, pd.DataFrame(detail)


def show_weight_page():
    st.header('数据素养指标权重')
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.bar(C['维度'], C['权重'], color='skyblue')
        ax.set_ylabel('权重')
        plt.xticks(rotation=45, ha='right')
        for i, v in enumerate(C['权重']):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.pie(C['权重'], labels=C['维度'], autopct='%.1f%%', startangle=90)
        st.pyplot(fig)
        plt.close(fig)


def show_test_page():
    st.header('数据素养测评')

    # 用户信息收集
    st.subheader("个人信息")
    col1, col2, col3 = st.columns(3)
    with col1:
        grade = st.selectbox("年级", ["大一", "大二", "大三", "大四", "研究生", "其他"])
    with col2:
        major = st.selectbox("专业类别", [
            "理工类", "经管类", "人文社科类", "艺术类", "医学类", "其他"
        ])
    with col3:
        data_exp = st.selectbox("数据相关经验", [
            "无经验", "少量课程学习", "参加过相关培训", "有项目经验", "专业领域经验丰富"
        ])

    # 保存用户信息到session state
    st.session_state.user_info = {
        'grade': grade,
        'major': major,
        'data_exp': data_exp
    }

    if 'answers' not in st.session_state:
        st.session_state.answers = [[] for _ in 题库]

    tabs = st.tabs(C['维度'].tolist())
    for i, (code, tab) in enumerate(zip(题库.keys(), tabs)):
        with tab:
            ans = []
            for j, q in enumerate(题库[code]['Question']):
                ans.append(st.slider(f'{q}', 1, 6, 3, key=f'{code}_{j}'))
            st.session_state.answers[i] = ans

    if st.button('提交测评', type='primary'):
        st.session_state.test_completed = True
        st.success('提交成功！请前往"查看结果"页面。')
        st.balloons()


def show_result_page():
    st.header('测评结果')
    if not st.session_state.get('test_completed', False):
        st.warning('请先完成测评！')
        return

    total, scores, rates, detail = calc_scores(st.session_state.answers)
    max_total = sum(np.array(题库[code]['Score']).sum() * C.loc[i, '权重'] for i, code in enumerate(题库))

    # 保存用户记录
    if 'current_record' not in st.session_state:
        user_info = st.session_state.get('user_info', {})
        st.session_state.current_record = add_user_record(user_info, scores, rates, total)

    # 显示个人结果
    col1, col2, col3 = st.columns(3)
    col1.metric('综合得分', f'{total:.2f}')
    col2.metric('满分', f'{max_total:.2f}')
    col3.metric('得分率', f'{total / max_total * 100:.2f}%')

    st.subheader('各维度得分')
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(pd.DataFrame({'维度': C['维度'], '得分': [f'{s:.2f}' for s in scores],
                                   '得分率': [f'{r:.2f}%' for r in rates]}))
    with col2:
        fig, ax = plt.subplots()
        ax.bar(C['维度'], rates, color='lightgreen')
        ax.set_ylabel('得分率(%)')
        plt.xticks(rotation=45, ha='right')
        ax.set_ylim(0, max(100, max(rates) * 1.05))
        for i, v in enumerate(rates):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center')
        st.pyplot(fig)
        plt.close(fig)

    st.subheader('详细得分')
    st.dataframe(detail, use_container_width=True)
    csv = detail.to_csv(index=False).encode('utf-8')
    st.download_button('下载 CSV', csv, '数据素养测评结果.csv', 'text/csv')


def show_group_portrait():
    st.header('群体画像分析')

    user_data = load_user_data()
    if not user_data:
        st.info('暂无群体数据，请先完成测评以生成群体画像')
        return

    df = pd.DataFrame(user_data)

    # 总体统计
    st.subheader('总体统计')
    col1, col2, col3, col4 = st.columns(4)

    total_users = len(df)
    avg_total_score = np.mean([x['total_score'] for x in user_data])
    avg_rates = np.mean([x['score_rates'] for x in user_data], axis=0)

    col1.metric('总测评人数', total_users)
    col2.metric('平均综合得分', f'{avg_total_score:.2f}')
    col3.metric('最高得分', f'{max([x["total_score"] for x in user_data]):.2f}')
    col4.metric('最低得分', f'{min([x["total_score"] for x in user_data]):.2f}')

    # 维度得分分布
    st.subheader('各维度得分分布')
    dimension_data = []
    for record in user_data:
        for i, (score, rate) in enumerate(zip(record['scores'], record['score_rates'])):
            dimension_data.append({
                '维度': record['dimension_names'][i],
                '得分': score,
                '得分率': rate
            })

    dimension_df = pd.DataFrame(dimension_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 箱线图
    sns.boxplot(data=dimension_df, x='维度', y='得分率', ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_title('各维度得分率分布')

    # 小提琴图
    sns.violinplot(data=dimension_df, x='维度', y='得分率', ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_title('各维度得分率密度分布')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # 专业类别分析
    st.subheader('按专业类别分析')
    major_data = []
    for record in user_data:
        if 'user_info' in record and 'major' in record['user_info']:
            major_data.append({
                '专业': record['user_info']['major'],
                '综合得分': record['total_score']
            })

    if major_data:
        major_df = pd.DataFrame(major_data)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=major_df, x='专业', y='综合得分', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('各专业类别综合得分分布')
        st.pyplot(fig)
        plt.close(fig)

    # 年级分析
    st.subheader('按年级分析')
    grade_data = []
    for record in user_data:
        if 'user_info' in record and 'grade' in record['user_info']:
            grade_data.append({
                '年级': record['user_info']['grade'],
                '综合得分': record['total_score']
            })

    if grade_data:
        grade_df = pd.DataFrame(grade_data)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=grade_df, x='年级', y='综合得分', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('各年级综合得分分布')
        st.pyplot(fig)
        plt.close(fig)

    # 经验水平分析
    st.subheader('按数据经验分析')
    exp_data = []
    for record in user_data:
        if 'user_info' in record and 'data_exp' in record['user_info']:
            exp_data.append({
                '数据经验': record['user_info']['data_exp'],
                '综合得分': record['total_score']
            })

    if exp_data:
        exp_df = pd.DataFrame(exp_data)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=exp_df, x='数据经验', y='综合得分', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('不同数据经验水平综合得分分布')
        st.pyplot(fig)
        plt.close(fig)

    # 显示当前用户在群体中的位置
    if 'current_record' in st.session_state:
        st.subheader('您在群体中的位置')
        current_score = st.session_state.current_record['total_score']
        all_scores = [x['total_score'] for x in user_data]
        percentile = np.sum(np.array(all_scores) <= current_score) / len(all_scores) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.metric('您的综合得分', f'{current_score:.2f}')
            st.metric('超过的用户比例', f'{percentile:.1f}%')

        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(all_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(current_score, color='red', linestyle='--', linewidth=2, label='您的得分')
            ax.set_xlabel('综合得分')
            ax.set_ylabel('人数')
            ax.legend()
            ax.set_title('综合得分分布')
            st.pyplot(fig)
            plt.close(fig)


# ----------------- 主路由 -----------------
def main():
    st.set_page_config(page_title='大学生数据素养测评系统', page_icon='📊', layout='wide')
    st.title('📊 大学生数据素养测评系统')

    with st.sidebar:
        choice = st.radio('导航', ['指标权重', '开始测评', '查看结果', '群体画像'])

    if choice == '指标权重':
        show_weight_page()
    elif choice == '开始测评':
        show_test_page()
    elif choice == '查看结果':
        show_result_page()
    else:
        show_group_portrait()


if __name__ == '__main__':
    main()
