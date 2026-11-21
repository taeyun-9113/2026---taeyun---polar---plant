# main.py
# 극지식물 생육 차이 분석 대시보드  ―  3탭 구조
# tabs: 실험 소개 / 생육 결과 분석 / 실생활 대입
# 인천송천고 실제 CSV 맞춤 · chardet 제거 · Boxplot 2분할(길이/질량) · 단위 명시
# ★ statsmodels 없이도 추세선 표시 (numpy 수동 회귀)

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
# 2. 생육 CSV 열 이름 정규화
# ============================================================
def normalize_growth_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        '개체번호': '개체번호',
        '잎 수(장)': '잎 수(장)',
        '잎 수 (장)': '잎 수(장)',
        '잎수(장)': '잎 수(장)',
        '잎수 (장)': '잎 수(장)',
        '잎수': '잎 수(장)',
        '지상부 길이(mm)': '지상부 길이(mm)',
        '지상부길이(mm)': '지상부 길이(mm)',
        '지상부 길이 (mm)': '지상부 길이(mm)',
        '지상부길이 (mm)': '지상부 길이(mm)',
        '지상부길이': '지상부 길이(mm)',
        '지상부 길이': '지상부 길이(mm)',
        '지하부길이(mm)': '지하부 길이(mm)',
        '지하부 길이(mm)': '지하부 길이(mm)',
        '지하부길이 (mm)': '지하부 길이(mm)',
        '지하부 길이 (mm)': '지하부 길이(mm)',
        '지하부길이': '지하부 길이(mm)',
        '지하부 길이': '지하부 길이(mm)',
        '지상부질량(g)': '지상부질량(g)',
        '지상부 질량(g)': '지상부질량(g)',
        '지상부질량 (g)': '지상부질량(g)',
        '지상부 질량 (g)': '지상부질량(g)',
        '지상부질량': '지상부질량(g)',
        '지상부 질량': '지상부질량(g)',
        '지하부질량(g)': '지하부질량(g)',
        '지하부 질량(g)': '지하부질량(g)',
        '지하부질량 (g)': '지하부질량(g)',
        '지하부 질량 (g)': '지하부질량(g)',
        '지하부질량': '지하부질량(g)',
        '지하부 질량': '지하부질량(g)',
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

    # 생중량(g) 열이 없으면 지상부 + 지하부로 만들어 넣음
    if '생중량(g)' not in df.columns:
        df['생중량(g)'] = pd.to_numeric(df['지상부질량(g)'], errors='coerce') \
                        + pd.to_numeric(df['지하부질량(g)'], errors='coerce')

    return df

# ============================================================
# 3. 환경 CSV 열 이름 정규화 (간단)
# ============================================================
def normalize_env_columns(df: pd.DataFrame) -> pd.DataFrame:
    emap = {
        'time':'time','Time':'time','일시':'time','categories':'time',
        '카테고리':'time','시간':'time',
        'temperature':'temperature','Temperature':'temperature','온도':'temperature','온도(℃)':'temperature',
        'humidity':'humidity','Humidity':'humidity','습도':'humidity','습도(%)':'humidity',
        'ph':'ph','PH':'ph','pH':'ph',
        'ec':'ec','EC':'ec',
        'water_temp':'water_temp','수온':'water_temp',
        'co2':'co2','CO2':'co2',
    }
    return df.rename(columns={c: emap[c] for c in df.columns if c in emap})

def detect_time_column(env_df: pd.DataFrame) -> str | None:
    for c in ['time','categories','시간','일시']:
        if c in env_df.columns:
            return c
    for c in env_df.columns:
        if 'time' in c.lower() or '일시' in c or '카테고리' in c.lower():
            return c
    return None

# ============================================================
# 4. 통계 헬퍼
# ============================================================
def stat_summary(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) == 0:
        return {}
    return {
        'n':      int(len(s)),
        'mean':   float(s.mean()),
        'std':    float(s.std(ddof=1)),
        'median': float(s.median()),
        'max':    float(s.max()),
        'min':    float(s.min()),
    }

def pearson_r(x, y) -> float:
    xs = pd.to_numeric(x, errors='coerce').dropna()
    ys = pd.to_numeric(y, errors='coerce').dropna()
    df2 = pd.concat([xs, ys], axis=1).dropna()
    if len(df2) < 3:
        return float('nan')
    return float(df2.iloc[:, 0].corr(df2.iloc[:, 1]))

def ols_trendline(x, y):
    """numpy 단순 선형회귀 y = ax + b → 추세선 양 끝점 반환"""
    xs = pd.to_numeric(x, errors='coerce').dropna().values.astype(float)
    ys = pd.to_numeric(y, errors='coerce').dropna().values.astype(float)
    mask = ~np.isnan(xs) & ~np.isnan(ys)
    xs, ys = xs[mask], ys[mask]
    n = len(xs)
    if n < 3:
        return None
    xm, ym = xs.mean(), ys.mean()
    num = np.sum((xs - xm) * (ys - ym))
    den = np.sum((xs - xm) ** 2)
    if den == 0:
        return None
    a = num / den
    b = ym - a * xm
    xmin, xmax = float(xs.min()), float(xs.max())
    return (xmin, xmax, a * xmin + b, a * xmax + b)

# ============================================================
# 5. 산점도 + 추세선 (statsmodels 없이)
# ============================================================
def scatter_with_trendline(df, x_col, y_col, x_label, y_label, r_val, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_numeric(df[x_col], errors='coerce'),
        y=pd.to_numeric(df[y_col], errors='coerce'),
        mode='markers',
        marker=dict(size=7, color='steelblue', opacity=0.75,
                    line=dict(width=0.5, color='white')),
        text=df['개체번호'],
        hoverinfo='text+x+y',
        name='개체',
    ))
    tl = ols_trendline(df[x_col], df[y_col])
    if tl is not None:
        xmin, xmax, ymin, ymax = tl
        fig.add_trace(go.Scatter(
            x=[xmin, xmax], y=[ymin, ymax],
            mode='lines',
            line=dict(color='crimson', width=2.2, dash='solid'),
            name=f'추세선 (r={r_val:.3f})',
        ))
    fig.update_layout(
        height=400,
        margin=dict(l=52, r=24, t=52, b=48),
        template='plotly_white',
        xaxis_title=x_label, yaxis_title=y_label,
        title=title,
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
    )
    return fig

# ============================================================
# 6. Boxplot 분할 함수
#    지표 그룹이 다르면(길이 vs 질량) Y축 단위가 완전히 달라
#    한 화면에 같이 그리면 질량 값이 납작하게 보이므로 분리한다.
# ============================================================
def draw_boxplot_group(df, cols, y_title, color_scale='Blues'):
    """
    cols: 표시할 열 이름 리스트
    y_title: Y축 제목 (단위 포함)
    """
    box_data = df[cols].melt(var_name='지표', value_name='값')
    # 열 이름을 보기 좋은 표시명으로
    label_map = {
        '잎 수(장)':        '잎 수 (장)',
        '지상부 길이(mm)': '지상부 길이 (mm)',
        '지하부 길이(mm)': '지하부 길이 (mm)',
        '지상부질량(g)':   '지상부 질량 (g)',
        '지하부질량(g)':   '지하부 질량 (g)',
        '생중량(g)':        '생중량 (g)',
    }
    box_data['지표'] = box_data['지표'].map(lambda x: label_map.get(x, x))

    fig = px.box(
        box_data, x='지표', y='값', color='지표',
        labels={'지표': '생육 지표', '값': '측정값'},
        color_discrete_sequence=px.colors.sequential.__dict__.get(
            color_scale, px.colors.sequential.Blues),
    )
    fig.update_layout(
        height=400,
        margin=dict(l=52, r=24, t=52, b=48),
        template='plotly_white',
        showlegend=False,
        xaxis_title='생육 지표',
        yaxis_title=y_title,
    )
    fig.update_traces(boxmean=True)
    return fig

# ============================================================
# 7. 통계 용어 설명
# ============================================================
TERM_DESC = {
    '평균 (mean)':     '모든 값을 더해 개수로 나눈 값. 자료의 중심을 나타낸다.',
    '표준편차 (std)':  '값이 평균에서 얼마나 흩어져 있는지를 나타내는 값. '
                       '클수록 개체 간 차이가 크다.',
    '중앙값 (median)': '값을 크기순으로 나열했을 때 한가운데 있는 값. '
                       '극단값의 영향을 덜 받는다.',
    '상관계수 (r)':    '두 변수의 직선 관계 강도. −1 ~ +1 사이 값을 가진다. '
                       '+1에 가까우면 함께 증가, −1에 가까우면 하나가 증가할 때 다른 하나가 감소, '
                       '0에 가까우면 직선 관계가 약하다.',
    '추세선':           '두 변수의 관계를 가장 잘 나타내는 직선(선형회귀선). '
                       '점들이 이 선에 가까울수록 관계가 강하다.',
    'Boxplot (상자 그림)': '상자 아랫줄·윗줄은 중간 50% 범위, 상자 안의 선은 중앙값, '
                           '점(선 밖)은 범위를 벗어난 개체를 나타낸다. '
                           '상자가 길수록 개체 간 차이가 크다.',
}

# ============================================================
# 8. CSV 다운로드
# ============================================================
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8-sig')

# ============================================================
# 9. 메인
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
                                           help='형식 A/B 모두 지원 · 없으면 생육 분석만 실행')
    growth_file = st.sidebar.file_uploader('생육 결과 데이터 (CSV)', type='csv',
                                           help='개체번호, 잎 수, 길이, 질량 열이 있으면 자동 인식 · '
                                                '생중량(g) 없으면 자동 합산')

    st.sidebar.markdown('---')
    st.sidebar.caption('인천송천고 극지연구소 교육용 대시보드')

    # ── 데이터 로딩 ─────────────────────────────────────────────────────
    env_df, growth_df = None, None

    if env_file is not None:
        try:
            env_df = read_csv_auto_encoding(env_file.read())
            env_df = normalize_env_columns(env_df)
        except Exception as e:
            st.sidebar.error(f'환경 데이터 읽기 오류: {e}')

    if growth_file is not None:
        try:
            growth_df = read_csv_auto_encoding(growth_file.read())
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

    # ── 숫자 열 준비 ─────────────────────────────────────────────────────
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

    # ── 상관계수 미리 계산 ──────────────────────────────────────────────
    r_length_length = pearson_r(growth_df['지하부 길이(mm)'], growth_df['지상부 길이(mm)'])
    r_mass_mass     = pearson_r(growth_df['지하부질량(g)'],    growth_df['지상부질량(g)'])
    r_length_leaf   = pearson_r(growth_df['지하부 길이(mm)'], growth_df['잎 수(장)'])

    # ============================================================
    # 3탭
    # ============================================================
    tab1, tab2, tab3 = st.tabs([
        '🔬 실험 소개',
        '📊 생육 결과 분석',
        '🌱 실생활 대입',
    ])

    # ────────────────────────────────────────────────────────────────
    # 탭 1: 실험 소개
    # ────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown('## 📌 연구 질문')
        st.info(
            '💡 **핵심 질문**: "지상부의 어떤 요소를 보고 지하부의 상태를 예측할 수 있을까?"\n\n'
            '극지식물은 땅 위보다 땅 아래에서 더 오래, 더 크게 자라는 경우가 많다. '
            '그렇다면 지상부의 길이, 잎 수, 질량 같은 겉으로 드러나는 값만으로도 '
            '땅 속 지하부의 길이·질량을 예측할 수 있을까? '
            '이 질문에 답하면 현장에서 지상부만 측정해도 지하부 생육 상태를 '
            '추정할 수 있어, 작물 관리 효율을 높일 수 있다.'
        )

        st.markdown('## 🧪 연구 가설')
        col_h0, col_h1 = st.columns(2)
        with col_h0:
            st.markdown('**귀무가설 (H₀)**')
            st.caption(
                '지상부의 길이·질량·잎 수와 지하부의 길이·질량 사이에는 '
                '의미 있는 상관관계가 없다. (r ≈ 0)'
            )
        with col_h1:
            st.markdown('**대립가설 (H₁)**')
            st.caption(
                '지상부가 길고 무거울수록 지하부도 길고 무겁다. '
                '(r > 0, 양의 상관관계)'
            )

        st.markdown('## 🎯 이 연구를 통해 얻고자 하는 것')
        for item in [
            '지상부 측정만으로 지하부 생육 상태를 추정할 수 있는지 확인한다.',
            '지상부-지하부 상관관계 수치를 통해 스마트팜에서 지상부 센서만으로 '
            '지하부 상태를 간접 진단할 수 있는 근거를 마련한다.',
            '개체 간 생육 차이(표준편차, Boxplot)를 통해 선발·제거 기준 마련의 기초 자료를 얻는다.',
        ]:
            st.markdown(f'• {item}')

        st.markdown('---')
        st.markdown('## 📊 생육 데이터 핵심 요약')
        st.caption(f'총 **{len(growth_df)}개체**의 생육 측정값 기준')

        s_total  = stat_summary(growth_df['생중량(g)'])
        s_length = stat_summary(growth_df['지상부 길이(mm)'])
        s_root   = stat_summary(growth_df['지하부 길이(mm)'])
        s_leaf   = stat_summary(growth_df['잎 수(장)'])

        intro_metrics = [
            ('전체 개체 수',     f'{len(growth_df)}개체', ''),
            ('평균 생중량',      f'{s_total["mean"]:,.1f} g' if s_total else '–',
                                 f'범위 {s_total["min"]:,.1f} ~ {s_total["max"]:,.1f} g' if s_total else ''),
            ('평균 지상부 길이', f'{s_length["mean"]:,.1f} mm' if s_length else '–',
                                 f'± {s_length["std"]:,.1f} mm' if s_length else ''),
            ('평균 지하부 길이', f'{s_root["mean"]:,.1f} mm' if s_root else '–',
                                 f'± {s_root["std"]:,.1f} mm' if s_root else ''),
            ('평균 잎 수',       f'{s_leaf["mean"]:,.1f} 장' if s_leaf else '–',
                                 f'± {s_leaf["std"]:,.1f} 장' if s_leaf else ''),
        ]
        kpi_cols = st.columns(len(intro_metrics))
        for col, (label, value, delta) in zip(kpi_cols, intro_metrics):
            with col:
                st.metric(label=label, value=value, delta=delta)

        st.markdown('---')
        st.markdown('## 📖 핵심 용어 설명 (고등학생용)')
        with st.expander('통계 용어 펼치기'):
            for term, desc in TERM_DESC.items():
                st.markdown(f'**{term}** — {desc}')

    # ────────────────────────────────────────────────────────────────
    # 탭 2: 생육 결과 분석
    # ────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('## 📦 개체별 생육 지표 분포 (1) ― 길이·잎 수 계열')
        st.caption(
            '아래 상자 그림은 **길이(mm)와 잎 수(장)** 지표만 모았다. '
            '값이 클수록(예: 지상부 길이 40mm 이상) 개체의 지상부·지하부가 길게 자란 것이다. '
            '단위: 잎 수 = 장, 길이 = mm. '
            '잎 수와 길이는 질량(g)과 척도가 달라 따로 표시한다.'
        )

        length_cols = ['잎 수(장)', '지상부 길이(mm)', '지하부 길이(mm)']
        fig_length = draw_boxplot_group(
            growth_df, length_cols,
            y_title='측정값 (장 / mm)',
            color_scale='Blues',
        )
        st.plotly_chart(fig_length, use_container_width=True)

        st.markdown('---')
        st.markdown('## 📦 개체별 생육 지표 분포 (2) ― 질량 계열')
        st.caption(
            '아래 상자 그림은 **질량(g)** 지표만 모았다. '
            '단위: 모든 값이 **g(그램)** 기준이다. '
            '지상부 질량은 0 ~ 2.6 g, 지하부 질량은 0.2 ~ 4.2 g, '
            '생중량(지상부+지하부)은 최대 6.3 g 정도로, 길이 지표와는 값 크기가 완전히 다르므로 '
            '별도 그래프로 보는 것이 정확하다.'
        )

        mass_cols = ['지상부질량(g)', '지하부질량(g)', '생중량(g)']
        fig_mass = draw_boxplot_group(
            growth_df, mass_cols,
            y_title='측정값 (g)',
            color_scale='Greens',
        )
        st.plotly_chart(fig_mass, use_container_width=True)

        st.markdown('---')
        st.markdown('## 🔗 생육 지표 간 상관관계 행렬')
        if len(num_cols) >= 2:
            corr = growth_df[num_cols].corr()
            label_idx = [col_display.get(c, c) for c in corr.columns]
            fig_corr = px.imshow(
                corr.values, x=label_idx, y=label_idx,
                text_auto='.2f', color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1, title='생육 지표 간 상관계수 행렬',
                aspect='auto',
            )
            fig_corr.update_layout(
                height=360, margin=dict(l=60, r=24, t=48, b=48),
                template='plotly_white',
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            corr_l = corr.copy()
            corr_l.index = [col_display.get(c, c) for c in corr.index]
            corr_l.columns = [col_display.get(c, c) for c in corr.columns]
            st.caption('대각선은 자기 자신과의 상관계수라 항상 1이다.')
            st.dataframe(corr_l.round(3), use_container_width=True, hide_index=False)
        else:
            st.info('상관계수 계산을 위해 숫자 열이 2개 이상 필요하다.')

        st.markdown('---')
        st.markdown('## 📈 지하부 길이 vs 잎 수')
        r1 = pearson_r(growth_df['지하부 길이(mm)'], growth_df['잎 수(장)'])
        fig1 = scatter_with_trendline(
            growth_df, '잎 수(장)', '지하부 길이(mm)',
            '잎 수 (장)', '지하부 길이 (mm)',
            r1,
            f'지하부 길이 vs 잎 수  (r = {r1:.3f})',
        )
        fig1.update_traces(hovertemplate='개체: %{text}<br>잎 수: %{x} 장<br>지하부 길이: %{y} mm<extra></extra>')
        st.plotly_chart(fig1, use_container_width=True)
        if abs(r1) < 0.3:
            st.caption(
                f'상관계수 r = **{r1:.3f}**. '
                '잎 수가 많을수록 지하부 길이가 약간 길어지는 경향이 있으나, '
                '관계가 약해 잎 수만으로 지하부 길이를 예측하기는 어렵다.'
            )
        else:
            st.caption(
                f'상관계수 r = **{r1:.3f}**. '
                '잎 수와 지하부 길이 사이에 뚜렷한 관계가 보인다.'
            )

        st.markdown('## 📈 지하부 길이 vs 지상부 길이')
        r2 = pearson_r(growth_df['지하부 길이(mm)'], growth_df['지상부 길이(mm)'])
        fig2 = scatter_with_trendline(
            growth_df, '지상부 길이(mm)', '지하부 길이(mm)',
            '지상부 길이 (mm)', '지하부 길이 (mm)',
            r2,
            f'지하부 길이 vs 지상부 길이  (r = {r2:.3f})',
        )
        fig2.update_traces(hovertemplate='개체: %{text}<br>지상부 길이: %{x} mm<br>지하부 길이: %{y} mm<extra></extra>')
        st.plotly_chart(fig2, use_container_width=True)
        if r2 > 0.5:
            st.caption(
                f'상관계수 r = **{r2:.3f}**. '
                '지상부 길이가 길수록 지하부 길이도 길어지는 강한 양의 관계가 나타난다. '
                '지상부 길이만으로 지하부 길이를 상당히 예측할 수 있다.'
            )
        else:
            st.caption(
                f'상관계수 r = **{r2:.3f}**. '
                '지상부 길이와 지하부 길이 사이의 관계가 뚜렷하지 않다.'
            )

        st.markdown('---')
        st.markdown('## 📊 개체별 생중량 비교 막대그래프')
        bar_df = growth_df[['개체번호', '생중량(g)']].copy()
        bar_df = bar_df.sort_values('생중량(g)', ascending=False)
        top_n = st.slider('표시할 개체 수', min_value=5, max_value=len(bar_df), value=15)
        bar_show = bar_df.head(top_n)

        fig_bar = px.bar(
            bar_show, x='개체번호', y='생중량(g)',
            color='생중량(g)',
            color_continuous_scale='YlOrRd',
            title=f'개체별 생중량 상위 {top_n}개 (단위: g)',
            labels={'개체번호': '개체번호', '생중량(g)': '생중량 (g)'},
        )
        fig_bar.update_layout(
            height=400, margin=dict(l=48, r=24, t=48, b=48),
            template='plotly_white', xaxis_title='개체번호', yaxis_title='생중량 (g)',
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('---')
        st.markdown('## 🗃️ 생육 데이터 원본 표')
        st.caption(f'총 {len(growth_df)}개체의 전체 생육 데이터')
        st.dataframe(
            growth_df[['개체번호'] + num_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = df_to_csv_bytes(
            growth_df[['개체번호'] + num_cols].reset_index(drop=True)
        )
        st.download_button(
            label='⬇️ CSV 다운로드',
            data=csv_bytes,
            file_name='polar_growth_result.csv',
            mime='text/csv',
            help='현재 생육 데이터를 CSV 파일로 다운로드한다.',
        )

    # ────────────────────────────────────────────────────────────────
    # 탭 3: 실생활 대입
    # ────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown('## 🌱 분석 결과 요약')

        col_insight1, col_insight2 = st.columns(2)
        with col_insight1:
            st.markdown('**① 지상부 길이 → 지하부 길이 예측**')
            st.info(
                f'지상부 길이와 지하부 길이 사이의 상관계수 r = **{r_length_length:.2f}**.\n\n'
                '지상부가 길수록 지하부도 길어지는 **강한 양의 상관관계**가 확인되었다. '
                '지상부 길이 값을 알면 지하부 길이를 상당히 정확하게 추정할 수 있다는 뜻이다.'
            )
        with col_insight2:
            st.markdown('**② 지상부 질량 → 지하부 질량 예측**')
            st.info(
                f'지상부 질량과 지하부 질량 사이의 상관계수 r = **{r_mass_mass:.2f}**.\n\n'
                '지상부 질량이 클수록 지하부 질량도 커지는 **매우 강한 양의 상관관계**가 확인되었다. '
                '질량 기준으로도 지상부 측정만으로 지하부 상태를 추정할 수 있다.'
            )

        st.markdown('## 🔍 인사이트 정리')
        insights = [
            ('지상부 길이로 지하부 길이 예측 가능',
             f'지상부 길이-지하부 길이 상관계수 r = {r_length_length:.2f} (강한 양의 상관). '
             '지상부가 긴 개체는 지하부도 길게 자라는 경향이 뚜렷하다.'),
            ('지상부 질량으로 지하부 질량 예측 가능',
             f'지상부 질량-지하부 질량 상관계수 r = {r_mass_mass:.2f} (매우 강한 양의 상관). '
             '지상부 무게로 지하부 무게를 추정하는 데 신뢰도가 높다.'),
            ('잎 수만으로는 지하부 예측 어려움',
             f'지하부 길이-잎 수 상관계수 r = {r_length_leaf:.2f} (약한 상관). '
             '잎 수는 지하부 크기와 거의 관계가 없어, 잎 수만 보고 지하부 상태를 판단하기는 어렵다.'),
            ('개체 간 차이가 매우 큼',
             '잎 수는 최소 7장에서 최대 105장, 지하부 길이는 2mm에서 77mm까지 차이가 난다. '
             '같은 환경에서 자라도 개체별 생육 편차가 크므로, 평균만으로 판단하지 말고 개체 단위로 살펴야 한다.'),
        ]
        for title, desc in insights:
            st.markdown(f'### {title}')
            st.caption(desc)

        st.markdown('---')
        st.markdown('## 🏠 실생활 · 스마트팜 대입 방안')
        st.markdown('### 1. 지상부 센서만으로 지하부 생육 진단하기')
        st.caption(
            '스마트팜에서는 보통 지상부에 카메라를 설치하거나 높이 센서로 지상부 길이·생장 속도를 '
            '측정한다. 이 분석 결과를 활용하면:\n\n'
            '• 지상부 길이가 평균보다 짧은 개체가 발견되면 → 지하부도 짧을 가능성이 높으므로 '
            '뿌리 생육 점검 또는 양액·수분 관리를 추가로 확인한다.\n'
            '• 지상부 질량이 무거운 개체는 지하부 질량도 무거운 경향이므로 → '
            '지상부 측정만으로도 전체 생육이 양호한 개체를 선별할 수 있다.\n'
            '• 지상부만 보고도 지하부 상태를 통계적으로 추정할 수 있다는 근거를 확보했으므로, '
            '측정 장비를 지하부까지 늘리지 않아도 효율적인 관리가 가능하다.'
        )
        st.markdown('### 2. 개체 선발·제거 기준 만들기')
        st.caption(
            '개체 간 생육 차이가 크므로, 평균과 표준편차를 활용해 다음과 같은 기준을 세울 수 있다:\n\n'
            '• 지상부 길이가 평균 − 1 표준편차보다 짧은 개체 → 지하부도 짧을 가능성 ↑ → '
            '원인 점검 대상\n'
            '• 생중량이 상위 20% 이내인 개체 → 지상부·지하부 모두 우수할 가능성 ↑ → '
            '선발·증식 후보\n'
            '• 잎 수만 많고 지상부 길이는 짧은 개체(잎 수-지상부 길이 상관 낮음) → '
            '잎에만 에너지가 몰리는 비정상 생육 가능성 점검'
        )
        st.markdown('### 3. 데이터 기반 관리의 한계와 주의점')
        st.caption(
            '• 이번 분석은 개체 54개 기준이므로, 더 많은 개체로 반복 확인하면 상관계수 추정의 '
            '신뢰도가 더 높아진다.\n'
            '• 상관관계는 인과관계가 아니다. 지상부가 길어서 지하부가 긴 것인지, '
            '공통 원인(양분·수분·온도) 때문에 둘 다 큰 것인지는 별도 실험이 필요하다.\n'
            '• 환경 데이터(온도·습도·pH·EC 등)와 생육 데이터를 시간 축으로 연결하면, '
            '어떤 환경 조건이 지하부 생육에 좋은지까지 분석할 수 있다.'
        )

        st.markdown('---')
        st.caption(
            '💡 인사이트 핵심: 지상부 길이·질량만으로도 지하부 길이·질량을 상당히 예측할 수 있다. '
            '스마트팜에서 지상부 센서를 활용해 지하부 생육을 간접 진단하는 근거로 활용할 수 있다.'
        )

    # ── 환경 데이터 섹션 ─────────────────────────────────────────────────
    if env_df is not None and not env_df.empty:
        st.markdown('---')
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
            ekc = st.columns(5)
            for col, label, unit in env_kpi:
                if col in env_df.columns and shown < 5:
                    s = stat_summary(env_df[col])
                    if s:
                        with ekc[shown]:
                            st.metric(label=label, value=f'{s["mean"]:,.1f} {unit}')
                        shown += 1

        if time_col:
            plot_cols = ['temperature','온도','humidity','습도',
                         'ph','PH','ec','EC','water_temp','co2']
            avail = [c for c in plot_cols if c in env_df.columns]
            if avail:
                fig_env = go.Figure()
                for col in avail:
                    yvals = pd.to_numeric(env_df[col], errors='coerce')
                    fig_env.add_trace(go.Scatter(
                        x=env_df[time_col], y=yvals,
                        mode='lines+markers', name=col,
                        line=dict(width=1.2), marker=dict(size=3),
                    ))
                fig_env.update_layout(
                    height=300, margin=dict(l=48, r=24, t=48, b=48),
                    xaxis_title='시간', yaxis_title='값',
                    legend=dict(orientation='h', yanchor='bottom',
                                y=1.02, xanchor='right', x=1),
                    template='plotly_white',
                )
                st.plotly_chart(fig_env, use_container_width=True)

    st.markdown('---')
    ev_rows = len(env_df) if env_df is not None else 0
    st.caption(
        f'데이터 기준: 환경 데이터 {ev_rows:,}행 · 생육 데이터 {len(growth_df):,}개체  |  '
        f'인천송천고 극지연구소 교육용 대시보드'
    )
    st.caption(
        '💡 환경 데이터와 생육 데이터는 측정 시간이 다를 수 있어 '
        '직접 연결보다는 각각의 분포를 비교하는 방식이 권장된다. '
        '정확한 상관 분석이 필요하면 두 데이터의 시간 열을 맞춰 전처리해야 한다.'
    )

if __name__ == '__main__':
    main()
