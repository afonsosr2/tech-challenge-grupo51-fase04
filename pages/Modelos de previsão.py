import streamlit as st
import pandas as pd
from prophet.plot import plot_plotly
from prophet.serialize import model_from_json
from plotly import graph_objs as go
from utils import atualizando_dados_ipea

dados = pd.read_csv(atualizando_dados_ipea())

###### Modelos ######

# Previsão com Prophet
@st.cache_data
def prophet_prediction(periodo_previsao):
    # Carregando o modelo
    with open('serialized_model.json', 'r') as fin:
        m = model_from_json(fin.read())  # Load model
    # m = pd.read_pickle('modelo/Prophet.pkl')

    future = m.make_future_dataframe(periods=periodo_previsao, freq="B")
    forecast = m.predict(future)
    forecast_resumo = forecast[["ds", "yhat"]].rename(columns=
                                                      {"ds": "Data", 
                                                       "yhat": "Preço - petróleo bruto - Brent (FOB)"})

    # Show and plot forecast
    st.subheader('Previsão')
    st.dataframe(forecast_resumo.tail())
        
    st.write(f'### Gráfico de previsão em {periodo_previsao} dias')
    plot_prev_prophet = plot_plotly(m, forecast)
    st.plotly_chart(plot_prev_prophet)

def modelo_1_prediction(periodo_previsao):
    st.text('Em construção...')


###### Criando a página do Streamlit ######
    
#### Sidebar ####
    
st.sidebar.title('Parâmetros do Modelo')
with st.sidebar.expander('Período de Previsão', True):
    periodo = st.slider('Selecione o período de previsão:', 1, 365, 7)

with st.sidebar.expander('Modelo de Machine Learning', True):
    input_modelo = st.selectbox('Selecione o modelo que deseja utilizar:', ['Modelo_1', 'Prophet'], 0)


#### Página dos modelos de previsão do petróleo Brent ####
st.write('# :oil_drum: Análise de preços do Petróleo Brent')

st.metric('Data da última atualização dos dados:', dados["Data"].max())
st.metric('Preço da última atualização dos dados:', dados['Preço - petróleo bruto - Brent (FOB)'].iloc[-1])

st.subheader('Últimos 5 dias')
st.write(dados.tail())

# Gráfico dos dados atuais
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados['Data'], y=dados['Preço - petróleo bruto - Brent (FOB)'], name="Preço do Petróleo Brent"))
    fig.layout.update(title_text='Preço do Petróleo Brent (FOB)', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

if(input_modelo == "Prophet"):
    prophet_prediction(periodo)
if(input_modelo == "Modelo_1"):
    modelo_1_prediction(periodo)

