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
    "Perigo": "rgb(255,0,0)"
}

horarios_filtros = [6, 9, 12, 15, 18, 21]
agora = datetime.now()
hora_atual = agora.hour
data_atual = agora.date()

# ======================
# 📍 CAPITAIS
# ======================
capitais_df = pd.read_excel("./lat_lon_capitais_br.xlsx")

# (Opcional) Albedo por capital — ajuste se quiser. Se não estiver no dict, usa o padrão.
ALBEDO_PADRAO = 0.22  # urbano claro típico; asfalto ~0.12–0.15, concreto claro ~0.25–0.30
ALBEDO_POR_CAPITAL = {
    # "Brasília": 0.24,
    # "São Paulo": 0.20,
    # "Rio de Janeiro": 0.22,
}

# ======================
# 🔬 FÍSICA DO GLOBO NEGRO (Tg)
# ======================
SIGMA = 5.670374419e-8   # Stefan-Boltzmann [W m-2 K-4]
EPS   = 0.95             # emissividade do globo preto
ALPHA_DEFAULT = 0.95     # absortância do globo para curta-onda
AP_AS = 0.25             # área projetada / área de superfície da esfera
# D (diâmetro) não entra explicitamente nesta forma (agregado nos coeficientes)

def _hc_sphere(wind_ms: float) -> float:
    """
    Coeficiente de convecção convectivo simplificado para esfera em ar (W m-2 K-1).
    Robusto para uso operacional em painel.
    """
    v = max(float(wind_ms), 0.1)  # evita singularidade
    return 1.4 * np.sqrt(v)

def tg_black_globe(
    Ta_C: float,
    GHI_Wm2: float,
    wind_ms: float,
    lw_down_Wm2: float = None,   # Onda longa descendente (W/m²) — opcional
    albedo: float = ALBEDO_PADRAO,
    alpha_sw: float = ALPHA_DEFAULT,
    max_iter: int = 60,
    tol: float = 1e-3
) -> float:
    """
    Tg (°C) via balanço de energia do globo negro:
      curto-onda (incidente + refletido pelo solo via albedo) +
      longa-onda descendente (atmosfera/nuvens) -
      emissão do globo -
      convecção com o ar
    = 0
    Resolve por Newton-Raphson.
    """
    Ta_K = Ta_C + 273.15

    # Curto-onda absorvido (incidente + reflexão do solo por albedo)
    ghi = max(float(GHI_Wm2), 0.0)
    ghi_total = ghi * (1.0 + max(float(albedo), 0.0))
    q_sw = alpha_sw * ghi_total * AP_AS  # W/m²

    # Longa-onda descendente: se não houver, usa fallback σ·Ta^4
    if (lw_down_Wm2 is None) or (not np.isfinite(lw_down_Wm2)):
        lw_down_Wm2 = SIGMA * Ta_K**4

    # Iteração Newton-Raphson
    Tg_K = Ta_K
    for _ in range(max_iter):
        hc = _hc_sphere(wind_ms)
        # Balanço radiativo-convectivo:
        # curto-onda + LW↓ - emissão globo (εσTg^4) - convecção (h_c (Tg - Ta)) = 0
        F  = q_sw + EPS*lw_down_Wm2 - EPS*SIGMA*Tg_K**4 - hc*(Tg_K - Ta_K)
        dF = -4.0*EPS*SIGMA*(Tg_K**3) - hc
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
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        albedo_local = ALBEDO_POR_CAPITAL.get(nome, ALBEDO_PADRAO)
        try:
            # Tentamos pedir também a "terrestrial_radiation" (LW↓) — o provedor pode aceitar ou ignorar.
            response = requests.get(
                url,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": "temperature_2m,wet_bulb_temperature_2m,shortwave_radiation,wind_speed_10m,terrestrial_radiation",
                    "timezone": "America/Sao_Paulo"
                },
                verify=False
            )
            result = response.json()
            df = pd.DataFrame(result["hourly"])
            df["Capital"]   = nome
            df["Latitude"]  = lat
            df["Longitude"] = lon
            df["Albedo"]    = albedo_local

            # Renomear para nomes curtos
            rename_map = {
                "temperature_2m": "Ta",                # °C
                "wet_bulb_temperature_2m": "Tw",       # °C
                "shortwave_radiation": "GHI",          # W/m²
                "wind_speed_10m": "Wind",              # m/s
            }
            # LW↓: vários candidatos possíveis (depende do provedor)
            for cand in ["surface_thermal_radiation_downwards", "terrestrial_radiation",
                         "longwave_radiation_downwelling", "lw_down"]:
                if cand in df.columns:
                    rename_map[cand] = "LW_down"

            df = df.rename(columns=rename_map)

            # Se a coluna LW_down não existir, cria vazia (fallback dentro da função cuidará)
            if "LW_down" not in df.columns:
                df["LW_down"] = np.nan

            # ======================
            # 🔢 CÁLCULO DO Tg
            # ======================

            # Externo (com sol): usa GHI real + reflexão por albedo
            df["Tg_out"] = [
                tg_black_globe(Ta, ghi, v, lw_down_Wm2=lw, albedo=alb)
                for Ta, ghi, v, lw, alb in zip(
                    df["Ta"].values, df["GHI"].values, df["Wind"].values, df["LW_down"].values, df["Albedo"].values
                )
            ]

            # Interno (sombra): GHI = 0, sem reflexão; ainda pode considerar LW↓
            df["Tg_in"] = [
                tg_black_globe(Ta, 0.0, v, lw_down_Wm2=lw, albedo=0.0)
                for Ta, v, lw in zip(
                    df["Ta"].values, df["Wind"].values, df["LW_down"].values
                )
            ]

            # ======================
            # 🧮 WBGT — FÓRMULAS OFICIAIS
            # ======================
            # Externo (sol): 0.7*Tw + 0.2*Tg + 0.1*Ta
            df["WBGT_out"] = (0.7*df["Tw"] + 0.2*df["Tg_out"] + 0.1*df["Ta"]).round(1)
            # Interno (sombra): 0.7*Tw + 0.3*Tg
            df["WBGT_in"]  = (0.7*df["Tw"] + 0.3*df["Tg_in"]).round(1)

            # Coluna padrão (usada pelo mapa inicialmente)
            df["WBGT"] = df["WBGT_out"]

            dados.append(df)

        except Exception as e:
            print(f"Erro em {nome}: {e}")

    if not dados:
        raise RuntimeError("Nenhum dado retornado pela API. Verifique a conexão/variáveis.")
    return pd.concat(dados, ignore_index=True)

df_previsao = coletar_dados()
df_previsao["time"] = pd.to_datetime(df_previsao["time"])
df_previsao["Data"] = df_previsao["time"].dt.date
df_previsao["Hora"] = df_previsao["time"].dt.hour
df_previsao["Hora_str"] = df_previsao["time"].dt.strftime("%Hh")

# ======================
# 🧭 CLASSIFICAÇÃO DE RISCO
# ======================
def classificar_risco(wbgt):
    if wbgt < 26:
        return "Normal"
    elif wbgt < 28:
        return "Atenção"
    elif wbgt < 30:
        return "Alerta"
    else:
        return "Perigo"

df_previsao["Risco"] = df_previsao["WBGT"].apply(classificar_risco)

RECOMENDACOES = {
    "Normal": "Hidrate-se regularmente e planeje pausas. Observe grupos sensíveis.",
    "Atenção": "Aumente pausas em sombra/área fresca; reforçar hidratação; monitorar sintomas iniciais.",
    "Alerta": "Pausas frequentes, reduzir intensidade/esforço, supervisão ativa; ajustar horários.",
    "Perigo": "Restringir atividades intensas ao ar livre; priorizar ambientes climatizados; vigilância de sinais de estresse térmico."
}

# ======================
# 🌍 APP DASH
# ======================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Painel WBGT"

app.layout = dbc.Container([
    html.H2("Painel de Risco Térmico (WBGT) - Capitais do Brasil", className="text-center mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='filtro-capital',
                options=[{"label": c, "value": c} for c in sorted(df_previsao["Capital"].unique())],
                value="Brasília",
                clearable=False
            )
        ], width=3),

        dbc.Col([
            dcc.DatePickerSingle(
                id='filtro-data',
                min_date_allowed=df_previsao['Data'].min(),
                max_date_allowed=df_previsao['Data'].max(),
                date=data_atual
            )
        ], width=2),

        dbc.Col([
            dcc.Dropdown(
                id='filtro-hora',
                options=[{"label": f"{h:02d}:00", "value": h} for h in horarios_filtros],
                placeholder="Escolha uma hora"
            )
        ], width=2),

        dbc.Col([
            dcc.RadioItems(
                id="filtro-ambiente",
                options=[
                    {"label": "Externo", "value": "out"},
                    {"label": "Interno", "value": "in"},
                ],
                value="out",
                inline=True
            )
        ], width=5)
    ], justify="center", className="mb-2"),

    dbc.Row([
        dbc.Col([
            html.Div("Risco do WBGT:", style={"fontSize": "18px", "marginBottom": "5px", "fontWeight": "bold", "textAlign": "center"}),
            html.Div([
                html.Span(" Normal ",  style={"backgroundColor": color_map["Normal"],  "padding": "5px", "marginRight": "10px", "borderRadius": "5px"}),
                html.Span(" Atenção ", style={"backgroundColor": color_map["Atenção"], "padding": "5px", "marginRight": "10px", "borderRadius": "5px"}),
                html.Span(" Alerta ",  style={"backgroundColor": color_map["Alerta"],  "padding": "5px", "marginRight": "10px", "borderRadius": "5px"}),
                html.Span(" Perigo ",  style={"backgroundColor": color_map["Perigo"],  "padding": "5px", "color": "white", "borderRadius": "5px"})
            ], style={"textAlign": "center", "marginBottom": "10px"})
        ], width=5),

        dbc.Col([
            html.Div([
                html.Span("● ", style={"color": color_map["Normal"], "fontSize": "20px"}),
                html.Span("Normal  ", style={"marginRight": "15px"}),
                html.Span("● ", style={"color": color_map["Atenção"], "fontSize": "20px"}),
                html.Span("Atenção  ", style={"marginRight": "15px"}),
                html.Span("● ", style={"color": color_map["Alerta"], "fontSize": "20px"}),
                html.Span("Alerta  ", style={"marginRight": "15px"}),
                html.Span("● ", style={"color": color_map["Perigo"], "fontSize": "20px"}),
                html.Span("Perigo")
            ], style={"textAlign": "center", "marginBottom": "20px", "fontWeight": "bold"})
        ], width=7)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Recomendações para a faixa atual"),
                dbc.CardBody(id="card-recomendacao", style={"minHeight": "80px"})
            ])
        ], width=12)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([ dcc.Graph(id='grafico-horario', style={"height": "700px"}) ], width=5),
        dbc.Col([ dcc.Graph(id='mapa-wbgt',    style={"height": "600px"}) ], width=7)
    ])
], fluid=True)

@app.callback(
    Output("mapa-wbgt", "figure"),
    [Input("filtro-data", "date"),
     Input("filtro-hora", "value"),
     Input("filtro-ambiente", "value")]
)
def atualizar_mapa(data, hora, ambiente):
    data = pd.to_datetime(data).date()
    if hora is None:
        hora = hora_atual
    df_dia = df_previsao[(df_previsao["Data"] == data) & (df_previsao["Hora"] == hora)].copy()

    col = "WBGT_out" if ambiente == "out" else "WBGT_in"
    df_dia["WBGT"] = df_dia[col]
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
        legend=dict(orientation="h", y=-0.2, x=0.25),
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
     Input("filtro-capital", "value"),
     Input("filtro-ambiente", "value")]
)
def atualizar_grafico(data, capital, ambiente):
    data = pd.to_datetime(data).date()
    df_capital = df_previsao[(df_previsao["Data"] == data) & (df_previsao["Capital"] == capital)].copy()

    col = "WBGT_out" if ambiente == "out" else "WBGT_in"
    df_capital["WBGT"] = df_capital[col]
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
        yaxis_title="WBGT (°C)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500
    )
    return fig

@app.callback(
    Output("card-recomendacao", "children"),
    [Input("filtro-data", "date"),
     Input("filtro-capital", "value"),
     Input("filtro-hora", "value"),
     Input("filtro-ambiente", "value")]
)
def atualizar_recomendacao(data, capital, hora, ambiente):
    data = pd.to_datetime(data).date()
    if hora is None:
        hora = hora_atual
    col = "WBGT_out" if ambiente == "out" else "WBGT_in"

    df_sel = df_previsao[
        (df_previsao["Data"] == data) &
        (df_previsao["Hora"] == hora) &
        (df_previsao["Capital"] == capital)
    ].copy()

    if df_sel.empty:
        return "Sem dados para o filtro selecionado."

    wbgt_val = float(df_sel[col].iloc[0])
    risco = classificar_risco(wbgt_val)
    rec = RECOMENDACOES[risco]

    return html.Div([
        html.P([
            html.Strong(f"{capital} – {data} {hora:02d}:00  "),
            f"WBGT ({'Externo' if ambiente=='out' else 'Interno'}): ",
            html.Strong(f"{wbgt_val:.1f} °C"),
            "  |  Risco: ",
            html.Span(risco, style={"backgroundColor": color_map[risco], "padding": "3px 6px", "borderRadius": "4px"})
        ], style={"marginBottom": "8px"}),
        html.P(rec, style={"marginBottom": 0})
    ])

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)



































