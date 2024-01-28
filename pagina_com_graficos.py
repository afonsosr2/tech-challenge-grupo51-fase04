import pandas as pd 
import plotly.express as px
import requests
import streamlit as st
from io import StringIO
from grafico import plotar_graf


st.title('Analise variação preço Petróleo:oil_drum:')


st.text(''' O preço do petróleo pode ser influenciado por vários fatores ao longo do tempo.
Para entender o aumento do preço do petróleo desde 1980 até 2024, é necessário considerar 
alguns fatores-chave:''')

st.subheader('Demanda crescente:')

st.text('''A demanda global por petróleo aumentou significativamente ao longo das 
décadas devido ao crescimento econômico, urbanização e aumento da população mundial.
Países emergentes, como a China e a Índia, experimentaram um rápido crescimento
econômico, aumentando sua demanda por energia, incluindo o petróleo.''')

st.subheader('Restrições de oferta:')  

st.text(''' A disponibilidade de petróleo é limitada e, em certos momentos, 
a oferta pode ser restringida devido a vários fatores.
Conflitos geopolíticos, instabilidade em países produtores 
de petróleo e interrupções na produção podem levar a uma 
diminuição da oferta,essas restrições podem levar a aumentos de preços.''')

st.subheader('OPEP:')
st.text(''' A Organização dos Países Exportadores de Petróleo (OPEP) é uma 
influente organização que agrupa países produtores de petróleo.
Ela pode adotar políticas que afetam a oferta global de petróleo, 
como reduzir a produção para aumentar os preços.As decisões 
da OPEP podem influenciar significativamente o preço do petróleo.''')

st.subheader('Flutuações cambiais:')
st.text('''As mudanças nas taxas de câmbio entre as moedas podem afetar o 
preço do petróleo, especialmente em relação ao dólar americano.
Como o petróleo é cotado em dólares americanos, alterações nas 
taxas de câmbio podem ter um impacto no preço para os países que 
não utilizam o dólar como moeda local.''')
        
st.subheader('Incertezas geopolíticas:') 

st.text('''Eventos geopolíticos, como guerras, sanções econômicas e tensões 
regionais podem ter um impacto significativo no preço do petróleo.
Essas incertezas podem gerar preocupações sobre a oferta futura e 
levar a aumentos nos preços.''')

graf_1= plotar_graf(1)

st.plotly_chart(graf_1)


st.subheader("Produção de petróleo por Região :")
st.text ("""Podemos notar que tivemos um crescimento exponencial de produção 
apartir do ano de 1990. As regiões que mais produzem Petróleo são Oriente Médio , 
America do Norte e a Comunidade dos Estados idependentes (Cis) que é composta 
por paises da antiga União Soviética.

A Região (CIS) Comunidade dos Estados Idependentes e o Oriente Médio são regiões 
de conflitos com alta representatividade em produção , os conflitos impactam nos 
preços de produção e logistica.

Fonte : https://www.investopedia.com/articles/investing/012216/how-opec-and-nonopec-production-affects-oil-prices.asp
""")

graf_2 = plotar_graf(2)

st.plotly_chart(graf_2)

st.subheader("Produção de petróleo por OPEP e Não OPEP:")

st.text('''Os preços do petróleo são influenciados por vários fatores, incluindo 
oferta e demanda. 
Os países membros da Organização dos Países Exportadores de Petróleo (OPEP) 
produzem cerca de 40% do petróleo bruto mundial.As exportações de petróleo da OPEP 
representam aproximadamente 60% do total de petróleo negociado internacionalmente. 

A OPEP, especialmente a Arábia Saudita, tem a vantagem na determinação da direção 
dos preços do petróleo, mas a Rússia também se tornou um ator-chave.

Fonte : https://www.investopedia.com/articles/investing/012216/how-opec-and-nonopec-production-affects-oil-prices.asp''')

text_tabela = '''Country	Oil exports (bbl/day)	Date of information
 Saudi Arabia (OPEC)	7,364,000	2022 est.
 Russia	4,780,000	2022 est.
 Iraq (OPEC)	3,712,000	2022 est.
 United States	3,604,000	2022
 Canada	3,350,000	2022
 United Arab Emirates (OPEC)	2,717,000	2022
 Kuwait (OPEC)	1,879,000	2022
 Norway	1,558,000	2022
 Nigeria (OPEC)	1,388,000	2022
 Brazil	1,346,417	2022 est.'''
 
tabela_opep = StringIO(text_tabela)
tabela_opep=pd.read_csv(tabela_opep,sep="\t")

st.dataframe(tabela_opep)




graf_3 = plotar_graf(3)
st.plotly_chart(graf_3)

st.subheader("Consumo Barril de petróleo por região")

st.text('''Podemos notar que as regiões que mais consomem Petróleo 
são a Ásia , América do Norte e Europa Asia produziu cerca de 2.4 bilhões de barris
em 2022 consumiu 11.4 bilhões em 2022 e a Europa produziu 1 bilhão de barris e consumiu 4.7 
a Asia e Europa nescessitam de importações e como vimos os maiores Exportadores são 
os Paises da OPEP que em sua maioria estão no Oriente Médio que é uma região com muitos
conflitos e que possui grande poder na decisão no preço de venda.

Notamos que o consumo de Petróleo ficou em torno de 33 bilhões de barris em 2022 e a 
produção total ficou em torno de 30 bilhões a demanda está maior que a oferta puxando 
os preçoa para cima .''')

graf_4 = plotar_graf(4)

st.plotly_chart(graf_4)