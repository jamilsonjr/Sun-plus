## Inteligência Artificial

### Modelo de Machine Learning
O Modelo de Machine Learning tem como objetivo a previsão do consumo da instituição, bem como a energia solar produzida, numa dada hora do carregamento pretendida pelo proprietario do veículo eletrico.
Estes modelos foram desenvolvidos em:
- Python 3.8;
- Keras, Pandas and SKLearn (bibliotecas de Python).

### Dados utilizados
  Com recurso ao website "Open Data" da EDP, tivemos acesso a dados de radiação de 2017 em Faro. De seguida, com o histórico de consumo de uma instituição conseguimos ter dados de treino para o nosso modelo, de modo a obter a previsão num dado dia desejado.
  Com os dados de radiação, através de um método linear, conseguimos obter o output dos paineis.
  
### Métodos utilizados:
Para fazer este "forecast" utlilizamos 2 métodos distintos:
- AutoRegressão: método de regressão linear;
- Long Short-Term Memory: algoritmo de Machine Learning. 

## Resultados

Em baixo, temos o resultado da previsão de consumo de uma instituição para um dado dia. É possível ver também o valor real.

<p align="center">
  <img height="350px" src="https://user-images.githubusercontent.com/47533831/115143096-322a1500-a03d-11eb-8b66-418e99f25b4c.png" />
</p>

De seguida, apresentamos todos os resultados do nosso modelo:
- Previsão da produção PV;
- Previsão do consumo da instituição;
- Surplus existente nesse dia.

<p align="center">
  <img height="350px" src="https://user-images.githubusercontent.com/47533831/115143124-5a197880-a03d-11eb-957f-bfab48a51ffc.png" />
</p>
