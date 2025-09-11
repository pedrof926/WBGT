import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from datetime import datetime
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
            response = requests.get(url, params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,wet_bulb_temperature_2m,shortwave_radiation,wind_speed_10m",
                "timezone": "America/Sao_Paulo"
            }, verify=False)
            result = response.json()
            df = pd.DataFrame(result["hourly"])
            df["Capital"] = nome
            df["Latitude"] = lat
            df["Longitude"] = lon

            # === CÁLCULOS WBGT ===
            # Renomear para nomes curtos (apenas para facilitar leitura)
            df = df.rename(columns={
                "temperature_2m": "Ta",
                "wet_bulb_temperature_2m": "Tw",
                "shortwave_radiation": "GHI",
                "wind_speed_10m": "Wind"
            })

            # Interno / Sombra (aproximação ISO: Tg ~ Ta): WBGT_in = 0.7*Tw + 0.3*Ta
            df["WBGT_in"]  = (0.7 * df["Tw"] + 0.3 * df["Ta"]).round(1)

            # Externo / Sol (mantendo sua ideia original com termo de radiação):
            # WBGT_out = 0.7*Tw + 0.2*Ta + 0.1*(GHI/100)
            df["WBGT_out"] = (0.7 * df["Tw"] + 0.2 * df["Ta"] + 0.1 * (df["GHI"] / 100.0)).round(1)

            # Compat: manter coluna "WBGT" padrão (usaremos conforme o ambiente escolhido)
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

# Recomendações por faixa
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

        # >>> NOVO: seletor de ambiente
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

    # 🔷 LEGENDA MAIOR DO "RISCO DO WBGT" (ESQUERDA) + LEGENDA DO MAPA (DIREITA)
    dbc.Row([
        dbc.Col([
            html.Div("Risco do WBGT:", style={"fontSize": "18px", "marginBottom": "5px", "fontWeight": "bold", "textAlign": "center"}),
            html.Div([
                html.Span(" Normal ", style={"backgroundColor": color_map["Normal"], "padding": "5px", "marginRight": "10px", "borderRadius": "5px"}),
                html.Span(" Atenção ", style={"backgroundColor": color_map["Atenção"], "padding": "5px", "marginRight": "10px", "borderRadius": "5px"}),
                html.Span(" Alerta ", style={"backgroundColor": color_map["Alerta"], "padding": "5px", "marginRight": "10px", "borderRadius": "5px"}),
                html.Span(" Perigo ", style={"backgroundColor": color_map["Perigo"], "padding": "5px", "color": "white", "borderRadius": "5px"})
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

    # >>> NOVO: CARD DE RECOMENDAÇÕES
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Recomendações para a faixa atual"),
                dbc.CardBody(id="card-recomendacao", style={"minHeight": "80px"})
            ])
        ], width=12)
    ], className="mb-3"),

    # 🔷 GRÁFICO E MAPA
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='grafico-horario', style={"height": "700px"})
        ], width=5),

        dbc.Col([
            dcc.Graph(id='mapa-wbgt', style={"height": "600px"})
        ], width=7)
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

    # Seleciona WBGT conforme ambiente
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
        yaxis_title="WBGT",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500
    )
    return fig

# >>> NOVO: recomendação dinâmica
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

































