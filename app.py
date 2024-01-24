import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go



dados = pd.read_csv("dados/ipea.csv")

# Criando a página do Streamlit

# Página dos modelos de previsão do petróleo Brent
st.write("# \U0001f6e2\uFE0F Análise de preços do Petróleo Brent")

st.write("### Selecione o modelo desejado para a previsão:")
input_modelo = st.selectbox("Qual o modelo que deseja utilizar?", ["Modelo_1", "Prophet"])

st.write("### Período da previsão")
semanas = st.slider('Semanas de previsão:', 1, 52)
periodo = semanas * 7

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
    # Previsão com Prophet
    df_train = dados[['Data','Preço - petróleo bruto - Brent (FOB)']]
    df_train = df_train.rename(columns={"Data": "ds", "Preço - petróleo bruto - Brent (FOB)": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=periodo)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Previsão')
    st.write(forecast.tail())
        
    st.write(f'### Gráfico de previsão em {periodo} dias')
    plot_prev_prophet = plot_plotly(m, forecast)
    st.plotly_chart(plot_prev_prophet)
if(input_modelo == "Modelo_1"):
    st.text('Em construção...')