# main.py
# 극지식물 생육 차이 분석 Streamlit 대시보드
# 인천송천고 실제 CSV 열 이름에 맞춰 작성 · chardet 의존 제거

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# 1. 인코딩 감지 후 CSV 읽기  (chardet 없이)
# ============================================================
def read_csv_auto_encoding(binary_bytes: bytes) -> pd.DataFrame:
    """utf-8-sig → cp949 → euc-kr 순서로 디코딩 시도"""
    for enc in ('utf-8-sig', 'cp949', 'euc-kr'):
        try:
            text = binary_bytes.decode(enc)
            return pd.read_csv(io.StringIO(text))
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError('CSV 읽기에 실패했습니다. 파일 형식을 확인해 주세요.')

# ============================================================
# 2. 생육 CSV 열 이름 정규화  ★ 실제 데이터 기준 보강
# ============================================================
def normalize_growth_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    실제 인천송천고 생육 CSV 열:
      개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm),
      지하부질량(g), 지상부질량(g)
    ※ 생중량(g)은 없을 수 있음 → 지상부+지하부로 합산 열 추가
    ※ 지하부길이(mm)는 띄어쓰기 없이 붙어 올 수 있음
    """
    col_map = {
        # 개체번호
        '개체번호': '개체번호',
        # 잎 수
        '잎 수(장)': '잎 수(장)',
        '잎 수 (장)': '잎 수(장)',
        '잎수(장)': '잎 수(장)',
        '잎수 (장)': '잎 수(장)',
        '잎수': '잎 수(장)',
        # 지상부 길이
        '지상부 길이(mm)': '지상부 길이(mm)',
        '지상부길이(mm)': '지상부 길이(mm)',
        '지상부 길이 (mm)': '지상부 길이(mm)',
        '지상부길이 (mm)': '지상부 길이(mm)',
        '지상부길이': '지상부 길이(mm)',
        '지상부 길이': '지상부 길이(mm)',
        # 지하부 길이  ← 실제 파일에 '지하부길이(mm)'(띄어쓰기 X)로 들어옴
        '지하부길이(mm)': '지하부 길이(mm)',
        '지하부 길이(mm)': '지하부 길이(mm)',
        '지하부길이 (mm)': '지하부 길이(mm)',
        '지하부 길이 (mm)': '지하부 길이(mm)',
        '지하부길이': '지하부 길이(mm)',
        '지하부 길이': '지하부 길이(mm)',
        # 지상부 질량
        '지상부질량(g)': '지상부질량(g)',
        '지상부 질량(g)': '지상부질량(g)',
        '지상부질량 (g)': '지상부질량(g)',
        '지상부 질량 (g)': '지상부질량(g)',
        '지상부질량': '지상부질량(g)',
        '지상부 질량': '지상부질량(g)',
        # 지하부 질량
        '지하부질량(g)': '지하부질량(g)',
        '지하부 질량(g)': '지하부질량(g)',
        '지하부질량 (g)': '지하부질량(g)',
        '지하부 질량 (g)': '지하부질량(g)',
        '지하부질량': '지하부질량(g)',
        '지하부 질량': '지하부질량(g)',
        # 생중량(있을 수도, 없을 수도 있음)
        '생중량(g)': '생중량(g)',
        '생중량 (g)': '생중량(g)',
        '생중량': '생중량(g)',
    }
    rename = {c: col_map[c] for c in df.columns if c in col_map}
    if rename:
        df = df.rename(columns=rename)

    required = ['개체번호', '잎 수(장)', '지상부 길이(mm)', '지하부 길이(mm)',
                '지상부질량(g)', '지하부질량(g)']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f'생육 데이터에서 필수 열이 없습니다: {missing}')
        return pd.DataFrame()

    df = df[required].reset_index(drop=True)

    # 생중량(g) 열이 없으면 지상부+지하부로 만들어 넣음
    if '생중량(g)' not in df.columns:
        df['생중량(g)'] = pd.to_numeric(df['지상부질량(g)'], errors='coerce') \
                        + pd.to_numeric(df['지하부질량(g)'], errors='coerce')
        st.info('📌 "생중량(g)" 열이 없어 지상부질량 + 지하부질량으로 계산했습니다.')

    return df

# ============================================================
# 3. 환경 CSV 열 이름 정규화  (형식 A / B)
# ============================================================
def normalize_env_columns(df: pd.DataFrame) -> pd.DataFrame:
    """형식 A (time, temperature, …) / 형식 B (categories, 온도, …) → 표준 이름"""
    col_map = {
        'time': 'time', 'Time': 'time', '일시': 'time',
        'categories': 'time', '카테고리': 'time', '시간': 'time',
        'temperature': 'temperature', 'Temperature': 'temperature',
        '온도': 'temperature', '온도(℃)': 'temperature',
        'humidity': 'humidity', 'Humidity': 'humidity',
        '습도': 'humidity', '습도(%)': 'humidity',
        'ph': 'ph', 'PH': 'ph', 'pH': 'ph', '수소이온농도': 'ph',
        'ec': 'ec', 'EC': 'ec', '전기전도도': 'ec',
        'water_temp': 'water_temp', 'water temperature': 'water_temp',
        '수온': 'water_temp',
        'co2': 'co2', 'CO2': 'co2', '이산화탄소': 'co2',
    }
    rename = {c: col_map[c] for c in df.columns if c in col_map}
    if rename:
        df = df.rename(columns=rename)
    return df

def detect_time_column(env_df: pd.DataFrame) -> str | None:
    for col in ['time', 'categories', '시간', '일시']:
        if col in env_df.columns:
            return col
    for col in env_df.columns:
        if 'time' in col.lower() or '일시' in col or '카테고리' in col.lower():
            return col
    return None

# ============================================================
# 4. 통계 헬퍼
# ============================================================
def stat_summary(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) == 0:
        return {}
    return {
        'n':      len(s),
        'mean':   s.mean(),
        'std':    s.std(ddof=1),
        'median': s.median(),
        'max':    s.max(),
        'min':    s.min(),
    }

# ============================================================
# 5. 통계 용어 설명  (고등학생용)
# ============================================================
TERM_DESC = {
    '평균 (mean)':     '모든 값을 더해 개수로 나눈 값. 자료의 중심을 나타낸다.',
    '표준편차 (std)':  '값이 평균에서 얼마나 흩어져 있는지를 나타내는 값. '
                       '클수록 개체 간 차이가 크다.',
    '중앙값 (median)': '값을 크기순으로 나열했을 때 한가운데 있는 값. '
                       '극단적으로 크거나 작은 값의 영향을 덜 받는다.',
    '상관계수 (r)':    '두 변수의 직선 관계 강도. −1 ~ +1 사이 값을 가진다. '
                       '+1에 가까우면 함께 증가하는 관계, −1이면 하나가 증가할 때 다른 하나가 감소.',
    'Boxplot (상자 그림)': '상자 아랫줄·윗줄은 중간 50% 범위, 상자 안의 선은 중앙값, '
                           '점(선 밖)은 범위를 벗어난 개체를 나타낸다.',
}

# ============================================================
# 6. 메인 대시보드
# ============================================================
def main():
    st.set_page_config(
        page_title='극지식물 생육 차이 분석',
        page_icon='🌿',
        layout='wide',
        initial_sidebar_state='expanded',
    )

    # ── 사이드바 ────────────────────────────────────────────────────────
    st.sidebar.title('📂 데이터 업로드')
    st.sidebar.info('학교마다 열 이름과 인코딩이 다를 수 있습니다. '
                    '어떤 파일이든 자동으로 맞춰 읽습니다.')

    env_file    = st.sidebar.file_uploader('환경 데이터 (CSV)', type='csv',
                                           help='형식 A/B 모두 지원 · 센서가 빈 기간이어도 OK · '
                                                '없으면 생육 분석만 실행됨')
    growth_file = st.sidebar.file_uploader('생육 결과 데이터 (CSV)', type='csv',
                                           help='개체번호, 잎 수, 길이, 질량 열이 있으면 자동 인식 · '
                                                '생중량(g)이 없으면 자동 합산')

    st.sidebar.markdown('---')
    st.sidebar.caption('인천송천고 극지연구소 교육용 대시보드')

    # ── 데이터 로딩 ─────────────────────────────────────────────────────
    env_df, growth_df = None, None

    if env_file is not None:
        raw = env_file.read()
        try:
            env_df = read_csv_auto_encoding(raw)
            env_df = normalize_env_columns(env_df)
        except Exception as e:
            st.sidebar.error(f'환경 데이터 읽기 오류: {e}')

    if growth_file is not None:
        raw = growth_file.read()
        try:
            growth_df = read_csv_auto_encoding(raw)
            growth_df = normalize_growth_columns(growth_df)
        except Exception as e:
            st.sidebar.error(f'생육 데이터 읽기 오류: {e}')

    # ── 타이틀 ──────────────────────────────────────────────────────────
    st.title('🌿 극지식물 생육 차이 분석 대시보드')
    st.caption('인천송천고 극지연구소 · 고등학생 통계 교육용으로 제작됨')

    if growth_df is None or growth_df.empty:
        st.warning('⚠️ 생육 결과 데이터 CSV를 업로드해 주세요.\n'
                   '오른쪽 사이드바에서 파일을 선택하면 분석이 시작됩니다.')
        st.markdown('''
        **기대하는 열 이름 (예시)**
        | 열 이름 | 의미 |
        |---|---|
        | 개체번호 | 개체 식별 코드 (예: A1, B3 또는 1, 2) |
        | 잎 수(장) | 잎 개수 (단위: 장) |
        | 지상부 길이(mm) | 땅 위 부분 길이 (단위: mm) |
        | 지하부 길이(mm) | 땅 아래 부분 길이 (단위: mm) — **지하부길이(mm**처럼 붙여 써도 OK) |
        | 지상부질량(g) | 땅 위 부분 질량 (단위: g) |
        | 지하부질량(g) | 땅 아래 부분 질량 (단위: g) |
        | 생중량(g) | 전체 질량 (없을 경우 지상부+지하부로 자동 계산) |
        ''')
        return

    # ============================================================
    # 본문 ① 개요 KPI
    # ============================================================
    st.markdown('## 📊 생육 데이터 요약')
    st.caption(f'총 **{len(growth_df)}개체**의 생육 측정값')

    num_cols = [c for c in growth_df.columns if c != '개체번호']
    col_display = {
        '잎 수(장)':        '잎 수 (장)',
        '지상부 길이(mm)': '지상부 길이 (mm)',
        '지하부 길이(mm)': '지하부 길이 (mm)',
        '지상부질량(g)':   '지상부 질량 (g)',
        '지하부질량(g)':   '지하부 질량 (g)',
        '생중량(g)':        '생중량 (g)',
    }
    col_unit = {
        '잎 수(장)':        '장',
        '지상부 길이(mm)': 'mm',
        '지하부 길이(mm)': 'mm',
        '지상부질량(g)':   'g',
        '지하부질량(g)':   'g',
        '생중량(g)':        'g',
    }

    kpi_cols = st.columns(min(5, len(num_cols)))
    for i, col in enumerate(num_cols):
        with kpi_cols[i]:
            s = stat_summary(growth_df[col])
            if not s:
                continue
            unit = col_unit.get(col, '')
            st.metric(
                label=col_display.get(col, col),
                value=f'{s["mean"]:,.1f}',
                delta=f'범위: {s["min"]:,.1f} ~ {s["max"]:,.1f} {unit}',
                delta_help='최솟값 - 최댓값 범위',
            )

    # ============================================================
    # 본문 ② 통계 지표 표
    # ============================================================
    st.markdown('## 📈 개체별 생육 통계')
    st.caption('각 지표가 개체들 사이에서 어떤 모양인지 살펴본다.')

    stats_rows = []
    for col in num_cols:
        s = stat_summary(growth_df[col])
        if not s:
            continue
        stats_rows.append({
            '지표':            col_display.get(col, col),
            '개체 수 (n)':     f'{s["n"]}개체',
            '평균 (mean)':     f'{s["mean"]:,.1f}',
            '표준편차 (std)':  f'{s["std"]:,.1f}',
            '중앙값 (median)': f'{s["median"]:,.1f}',
            '최댓값 (max)':    f'{s["max"]:,.1f}',
            '최솟값 (min)':    f'{s["min"]:,.1f}',
        })

    st.dataframe(
        pd.DataFrame(stats_rows),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander('📖 통계 용어 설명 (고등학생용)'):
        for term, desc in TERM_DESC.items():
            st.markdown(f'**{term}** — {desc}')

    # ============================================================
    # 본문 ③ Boxplot
    # ============================================================
    st.markdown('## 📦 개체별 생육 지표 분포')
    st.caption('상자 그림(Boxplot): 상자 안은 중간 50% 범위, '
               '가운데 선은 중앙값, 점(선 밖)은 범위를 벗어난 개체를 나타낸다. '
               '상자가 길수록 개체 간 차이가 크다는 뜻이다.')

    box_data = growth_df.melt(id_vars='개체번호', value_vars=num_cols,
                              var_name='지표', value_name='값')
    label_reversemap = {v: k for k, v in col_display.items()}
    box_data['지표'] = box_data['지표'].map(lambda x: label_reversemap.get(x, x))

    fig_box = px.box(
        box_data, x='지표', y='값', color='지표',
        title='개체별 생육 지표 분포 (Boxplot)',
        labels={'지표': '생육 지표', '값': '측정값'},
    )
    fig_box.update_layout(
        height=440,
        margin=dict(l=48, r=24, t=48, b=48),
        template='plotly_white',
        showlegend=False,
    )
    fig_box.update_traces(boxmean=True, meanline_visible=True)
    st.plotly_chart(fig_box, use_container_width=True)

    # ============================================================
    # 본문 ④ 상관관계
    # ============================================================
    st.markdown('## 🔗 생육 지표 간 상관관계')
    st.caption('상관계수(r)는 −1에서 +1 사이. '
               '+1이면 두 지표가 완전히 함께 증가, '
               '−1이면 하나가 증가할 때 다른 하나가 감소, '
               '0이면 직선 관계가 없다고 본다.')

    if len(num_cols) >= 2:
        corr = growth_df[num_cols].corr()
        label_idx = [col_display.get(c, c) for c in corr.columns]
        fig_corr = px.imshow(
            corr.values,
            x=label_idx, y=label_idx,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title='생육 지표 간 상관계수 행렬',
            aspect='auto',
        )
        fig_corr.update_layout(
            height=400,
            margin=dict(l=60, r=24, t=48, b=48),
            template='plotly_white',
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        corr_labeled = corr.copy()
        corr_labeled.index = [col_display.get(c, c) for c in corr.index]
        corr_labeled.columns = [col_display.get(c, c) for c in corr.columns]
        st.caption('대각선은 자기 자신과의 상관계수라 항상 1이다.')
        st.dataframe(corr_labeled.round(3), use_container_width=True, hide_index=False)

        st.markdown('### 개체별 산점도 행렬')
        st.caption('각 작은 그래프는 두 지표 사이의 관계. '
                   '점이 오른쪽 위로 퍼지면 함께 증가하는 경향.')
        fig_scatter = px.scatter_matrix(
            growth_df, dimensions=num_cols,
            color='개체번호',
            title='생육 지표 간 산점도 행렬',
        )
        fig_scatter.update_layout(
            height=620,
            margin=dict(l=48, r=24, t=48, b=48),
            template='plotly_white',
            showlegend=False,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info('상관계수를 계산하려면 숫자 열이 2개 이상 필요하다. '
                f'현재 숫자 열: {len(num_cols)}개')

    # ============================================================
    # 본문 ⑤ 개별 개체 상세
    # ============================================================
    st.markdown('## 🔍 개별 개체 상세 정보')
    st.caption('드롭다운에서 개체번호를 선택하면 해당 개체의 모든 생육 값을 확인할 수 있다.')

    all_ids = sorted(growth_df['개체번호'].astype(str).tolist())
    selected_id = st.selectbox(
        '개체번호 선택',
        options=all_ids,
        index=0,
        help='극지식물 개체를 하나씩 살펴본다.',
    )
    row = growth_df[growth_df['개체번호'].astype(str) == selected_id].iloc[0]

    detail_cols = st.columns(3)
    for i, col in enumerate(num_cols):
        with detail_cols[i % 3]:
            label = col_display.get(col, col)
            unit = col_unit.get(col, '')
            val = row[col]
            st.metric(label=label, value=f'{val:,.1f} {unit}')

    st.caption('선택한 개체의 원시 데이터 행:')
    st.dataframe(
        growth_df[growth_df['개체번호'].astype(str) == selected_id],
        use_container_width=True,
        hide_index=True,
    )

    # ============================================================
    # 본문 ⑥ 환경 데이터 (있을 때만)
    # ============================================================
    if env_df is not None and not env_df.empty:
        st.markdown('## 🌡️ 환경 데이터')
        st.caption(f'총 **{len(env_df)}행**의 환경 측정값')

        time_col = detect_time_column(env_df)
        if time_col:
            st.caption(f'시간 열: **{time_col}** (학교마다 이름이 다를 수 있음)')

        env_kpi = [
            ('temperature', '평균 온도', '℃'),
            ('온도',        '평균 온도', '℃'),
            ('humidity',   '평균 습도', '%'),
            ('습도',        '평균 습도', '%'),
            ('ph',         '평균 pH',   ''),
            ('PH',         '평균 pH',   ''),
            ('ec',         '평균 EC',   'dS/m'),
            ('EC',         '평균 EC',   'dS/m'),
            ('water_temp', '평균 수온', '℃'),
            ('co2',        '평균 CO₂',  'ppm'),
        ]
        shown = 0
        env_kpi_cols = st.columns(5)
        for col, label, unit in env_kpi:
            if col in env_df.columns and shown < 5:
                s = stat_summary(env_df[col])
                if s:
                    with env_kpi_cols[shown]:
                        st.metric(label=label, value=f'{s["mean"]:,.1f} {unit}')
                    shown += 1

        if time_col:
            plot_candidates = ['temperature','온도','humidity','습도',
                               'ph','PH','ec','EC','water_temp','co2']
            avail = [c for c in plot_candidates if c in env_df.columns]
            if avail:
                fig_env = go.Figure()
                for col in avail:
                    yvals = pd.to_numeric(env_df[col], errors='coerce')
                    fig_env.add_trace(go.Scatter(
                        x=env_df[time_col],
                        y=yvals,
                        mode='lines+markers',
                        name=col,
                        line=dict(width=1.2),
                        marker=dict(size=3),
                    ))
                fig_env.update_layout(
                    height=320,
                    margin=dict(l=48, r=24, t=48, b=48),
                    xaxis_title='시간',
                    yaxis_title='값',
                    legend=dict(orientation='h', yanchor='bottom',
                                y=1.02, xanchor='right', x=1),
                    template='plotly_white',
                )
                st.plotly_chart(fig_env, use_container_width=True)

    # ── 하단 안내 ───────────────────────────────────────────────────────
    st.markdown('---')
    env_rows = len(env_df) if env_df is not None else 0
    growth_rows = len(growth_df)
    st.caption(
        f'데이터 기준: 환경 데이터 {env_rows:,}행 · 생육 데이터 {growth_rows:,}개체  |  '
        f'인천송천고 극지연구소 교육용 대시보드'
    )
    st.caption(
        '💡 환경 데이터와 생육 데이터는 측정 시간이 다를 수 있어 '
        '직접 연결보다는 각각의 분포를 비교하는 방식이 권장된다. '
        '정확한 상관 분석이 필요하면 두 데이터의 시간 열을 맞춰 전처리해야 한다.'
    )

if __name__ == '__main__':
    main()
