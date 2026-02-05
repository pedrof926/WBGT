# -*- coding: utf-8 -*-
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
import requests

# ======================
# ⚙️ CONFIGURAÇÕES
# ======================
url = "https://api.open-meteo.com/v1/forecast"
color_map = {
    "Normal": "rgb(226,240,217)",
    "Atenção": "rgb(255,242,204)",
    "Alerta": "rgb(248,203,173)",
    "Perigo": "rgb(255,102,102)",
    "Extremo": "rgb(153,0,0)"
}

horarios_filtros = [6, 9, 12, 15, 18, 21]
agora = datetime.now()
hora_atual = agora.hour
data_atual = agora.date()

# ======================
# 📍 CAPITAIS
# ======================
capitais_df = pd.read_excel("./lat_lon_capitais_br.xlsx")

# ======================
# 🔬 FÍSICA DO GLOBO NEGRO
# ======================
SIGMA = 5.670374419e-8   # Stefan-Boltzmann [W m-2 K-4]
EPS   = 0.95             # emissividade do globo preto
ALPHA = 0.95             # absortância para curta-onda (pintura preta)
D     = 0.15             # diâmetro do globo [m]
AP_AS = 0.25             # razão área projetada/área de superfície (=1/4)

def _hc_sphere(wind_ms: float) -> float:
    v = max(wind_ms, 0.1)
    return 1.4 * np.sqrt(v)

def tg_black_globe(Ta_C, GHI_Wm2, wind_ms, longwave_K=None, max_iter=50, tol=1e-3):
    Ta_K = Ta_C + 273.15
    T_sur_K = Ta_K if longwave_K is None else longwave_K
    q_sw = ALPHA * GHI_Wm2 * AP_AS
    Tg_K = Ta_K
    for _ in range(max_iter):
        h_c = _hc_sphere(wind_ms)
        F   = q_sw + EPS*SIGMA*(T_sur_K**4 - Tg_K**4) - h_c*(Tg_K - Ta_K)
        dF  = -4.0*EPS*SIGMA*(Tg_K**3) - h_c
        step = -F / dF
        Tg_K_new = Tg_K + step
        if abs(step) < tol:
            Tg_K = Tg_K_new
            break
        Tg_K = Tg_K_new
    return float(Tg_K - 273.15)

# ======================
# 🛁 COLETA DOS DADOS DO OPEN-METEO
# ======================
def coletar_dados():
    dados = []
    for _, row in capitais_df.iterrows():
        nome = row["Capital"]
        lat = row["Latitude"]
        lon = row["Longitude"]
        try:
            response = requests.get(
                url,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": "temperature_2m,wet_bulb_temperature_2m,shortwave_radiation,wind_speed_10m",
                    "timezone": "America/Sao_Paulo"
                },
                verify=False
            )
            result = response.json()
            df = pd.DataFrame(result["hourly"])
            df["Capital"]  = nome
            df["Latitude"] = lat
            df["Longitude"]= lon

            df = df.rename(columns={
                "temperature_2m": "Ta",               # °C
                "wet_bulb_temperature_2m": "Tw",      # °C
                "shortwave_radiation": "GHI",         # W m-2
                "wind_speed_10m": "Wind"              # m s-1
            })

            # ✅ SOMENTE WBGT EXTERNO
            df["Tg_out"] = [
                tg_black_globe(Ta, ghi, v)
                for Ta, ghi, v in zip(df["Ta"].values, df["GHI"].values, df["Wind"].values)
            ]
            df["WBGT_out"] = (0.7*df["Tw"] + 0.2*df["Tg_out"] + 0.1*df["Ta"]).round(1)
            df["WBGT"] = df["WBGT_out"]

            dados.append(df)
        except Exception as e:
            print(f"Erro em {nome}: {e}")
    return pd.concat(dados, ignore_index=True)

df_previsao = coletar_dados()
df_previsao["time"] = pd.to_datetime(df_previsao["time"])
df_previsao["Data"] = df_previsao["time"].dt.date
df_previsao["Hora"] = df_previsao["time"].dt.hour
df_previsao["Hora_str"] = df_previsao["time"].dt.strftime("%Hh")

# ======================
# 🧭 CLASSIFICAÇÃO DE RISCO (ISO 7243)
# ======================
def classificar_risco(wbgt):
    """Apenas EXTERNO (com radiação solar direta)."""
    if wbgt < 27.8:
        return "Normal"
    elif wbgt < 29.4:
        return "Atenção"
    elif wbgt < 31.1:
        return "Alerta"
    elif wbgt < 32.2:
        return "Perigo"
    else:
        return "Extremo"

df_previsao["Risco"] = df_previsao["WBGT"].apply(classificar_risco)

RECOMENDACOES = {
    "Normal": "Hidrate-se regularmente e planeje pausas. Observe grupos sensíveis.",
    "Atenção": "Aumente pausas em sombra/área fresca; reforçar hidratação; monitorar sintomas iniciais.",
    "Alerta": "Pausas frequentes, reduzir intensidade/esforço, supervisão ativa; ajustar horários.",
    "Perigo": "Restringir atividades intensas ao ar livre; priorizar ambientes climatizados; vigilância de sinais de estresse térmico.",
    "Extremo": "Suspender atividades físicas; remover exposição ao calor; acionar protocolos de emergência."
}

# ======================
# 🌍 APP DASH (RESPONSIVO)
# ======================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Painel WBGT"

# Helpers de estilo (para evitar px fixo e ficar bem no mobile)
GRAPH_STYLE_MAPA  = {"height": "62vh", "width": "100%"}
GRAPH_STYLE_BARRA = {"height": "62vh", "width": "100%"}
CARD_STYLE        = {"minHeight": "90px"}

app.layout = dbc.Container([
    html.H2("Painel de Risco Térmico (WBGT) - Capitais do Brasil", className="text-center mb-3"),

    # 🔎 Filtros (responsivos)
    dbc.Row([
        dbc.Col([
            html.Label("Capital", className="fw-bold mb-1"),
            dcc.Dropdown(
                id='filtro-capital',
                options=[{"label": c, "value": c} for c in sorted(df_previsao["Capital"].unique())],
                value="Brasília",
                clearable=False
            )
        ], xs=12, sm=12, md=4, lg=4),

        dbc.Col([
            html.Label("Data", className="fw-bold mb-1"),
            dcc.DatePickerSingle(
                id='filtro-data',
                min_date_allowed=df_previsao['Data'].min(),
                max_date_allowed=df_previsao['Data'].max(),
                date=data_atual
            )
        ], xs=12, sm=6, md=4, lg=3),

        dbc.Col([
            html.Label("Hora", className="fw-bold mb-1"),
            dcc.Dropdown(
                id='filtro-hora',
                options=[{"label": f"{h:02d}:00", "value": h} for h in horarios_filtros],
                placeholder="Escolha uma hora"
            )
        ], xs=12, sm=6, md=4, lg=3),
    ], className="g-2 mb-2", align="end", justify="center"),

    # 🏷️ Legendas (responsivas)
    dbc.Row([
        dbc.Col([
            html.Div("Risco do WBGT:", style={"fontSize": "18px", "marginBottom": "5px", "fontWeight": "bold", "textAlign": "center"}),
            html.Div([
                html.Span(" Normal ",  style={"backgroundColor": color_map["Normal"],  "padding": "5px", "marginRight": "10px", "borderRadius": "5px"}),
                html.Span(" Atenção ", style={"backgroundColor": color_map["Atenção"], "padding": "5px", "marginRight": "10px", "borderRadius": "5px"}),
                html.Span(" Alerta ",  style={"backgroundColor": color_map["Alerta"],  "padding": "5px", "marginRight": "10px", "borderRadius": "5px"}),
                html.Span(" Perigo ",  style={"backgroundColor": color_map["Perigo"],  "padding": "5px", "marginRight": "10px", "borderRadius": "5px"}),
                html.Span(" Extremo ", style={"backgroundColor": color_map["Extremo"], "padding": "5px", "color": "white", "borderRadius": "5px"})
            ], style={"textAlign": "center", "marginBottom": "8px"})
        ], xs=12, md=5),

        dbc.Col([
            html.Div([
                html.Span("● ", style={"color": color_map["Normal"], "fontSize": "20px"}),
                html.Span("Normal  ", style={"marginRight": "15px"}),
                html.Span("● ", style={"color": color_map["Atenção"], "fontSize": "20px"}),
                html.Span("Atenção  ", style={"marginRight": "15px"}),
                html.Span("● ", style={"color": color_map["Alerta"], "fontSize": "20px"}),
                html.Span("Alerta  ", style={"marginRight": "15px"}),
                html.Span("● ", style={"color": color_map["Perigo"], "fontSize": "20px"}),
                html.Span("Perigo  ", style={"marginRight": "15px"}),
                html.Span("● ", style={"color": color_map["Extremo"], "fontSize": "20px"}),
                html.Span("Extremo")
            ], style={"textAlign": "center", "marginBottom": "8px", "fontWeight": "bold"})
        ], xs=12, md=7)
    ], className="g-2"),

    # 🧾 Card recomendação
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Recomendações para a faixa atual (WBGT Externo)"),
                dbc.CardBody(id="card-recomendacao", style=CARD_STYLE)
            ])
        ], xs=12)
    ], className="mb-3"),

    # 📌 Área principal (mapa + gráfico) -> responsivo:
    # - Mobile: empilha (mapa em cima, gráfico em baixo)
    # - Desktop: lado a lado
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='mapa-wbgt',
                style=GRAPH_STYLE_MAPA,
                config={"responsive": True}
            )
        ], xs=12, lg=7),

        dbc.Col([
            dcc.Graph(
                id='grafico-horario',
                style=GRAPH_STYLE_BARRA,
                config={"responsive": True}
            )
        ], xs=12, lg=5),
    ], className="g-2")

], fluid=True)


@app.callback(
    Output("mapa-wbgt", "figure"),
    [Input("filtro-data", "date"),
     Input("filtro-hora", "value")]
)
def atualizar_mapa(data, hora):
    data = pd.to_datetime(data).date()
    if hora is None:
        hora = hora_atual

    df_dia = df_previsao[(df_previsao["Data"] == data) & (df_previsao["Hora"] == hora)].copy()

    df_dia["WBGT"] = df_dia["WBGT_out"]
    df_dia["Risco"] = df_dia["WBGT"].apply(classificar_risco)

    fig = px.scatter_geo(
        df_dia,
        lat="Latitude",
        lon="Longitude",
        text="Capital",
        color="Risco",
        size="WBGT",
        size_max=8,
        color_discrete_map=color_map,
        hover_data={"Capital": True, "WBGT": True, "time": True}
    )
    fig.update_traces(
        marker=dict(line=dict(color='black', width=0.5)),
        textposition="middle right",
        textfont=dict(size=9)
    )
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=True,
        legend_title_text="Risco:",
        legend=dict(
            orientation="v",
            y=0.5,
            yanchor="middle",
            x=1.02,
            xanchor="left",
            title_text="Risco"
        ),
        geo=dict(
            resolution=50,
            showcountries=True,
            countrycolor="Gray",
            showsubunits=True,
            subunitcolor="Black",
            lataxis_range=[-35, 5],
            lonaxis_range=[-75, -30],
            showland=True,
            landcolor="rgb(240, 240, 240)",
            showcoastlines=False
        )
    )
    return fig


@app.callback(
    Output("grafico-horario", "figure"),
    [Input("filtro-data", "date"),
     Input("filtro-capital", "value")]
)
def atualizar_grafico(data, capital):
    data = pd.to_datetime(data).date()
    df_capital = df_previsao[(df_previsao["Data"] == data) & (df_previsao["Capital"] == capital)].copy()

    df_capital["WBGT"] = df_capital["WBGT_out"]
    df_capital["Risco"] = df_capital["WBGT"].apply(classificar_risco)

    fig = px.bar(
        df_capital,
        x="Hora_str",
        y="WBGT",
        color="Risco",
        color_discrete_map=color_map
    )
    fig.update_xaxes(title="Horas", categoryorder="array", categoryarray=[f"{h:02d}h" for h in range(24)])
    fig.update_layout(
        title={
            "text": capital,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18}
        },
        yaxis_title="WBGT",
        yaxis=dict(range=[0, 40]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"r":10,"t":55,"l":10,"b":10},
        height=None  # deixa o style (vh) mandar na altura
    )
    return fig


@app.callback(
    Output("card-recomendacao", "children"),
    [Input("filtro-data", "date"),
     Input("filtro-capital", "value"),
     Input("filtro-hora", "value")]
)
def atualizar_recomendacao(data, capital, hora):
    data = pd.to_datetime(data).date()
    if hora is None:
        hora = hora_atual

    df_sel = df_previsao[
        (df_previsao["Data"] == data) &
        (df_previsao["Hora"] == hora) &
        (df_previsao["Capital"] == capital)
    ].copy()

    if df_sel.empty:
        return "Sem dados para o filtro selecionado."

    wbgt_val = float(df_sel["WBGT_out"].iloc[0])
    risco = classificar_risco(wbgt_val)
    rec = RECOMENDACOES[risco]

    return html.Div([
        html.P([
            html.Strong(f"{capital} – {data} {hora:02d}:00  "),
            "WBGT: ",
            html.Strong(f"{wbgt_val:.1f}"),
            "  |  Risco: ",
            html.Span(
                risco,
                style={"backgroundColor": color_map[risco], "padding": "3px 6px", "borderRadius": "4px"}
            )
        ], style={"marginBottom": "8px"}),
        html.P(rec, style={"marginBottom": 0})
    ])


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)








































