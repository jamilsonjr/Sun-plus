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
<p align="center">
  <img height="400px" src="![image](https://user-images.githubusercontent.com/47533831/115143010-b203af80-a03c-11eb-88f8-b91d439fd2fd.png)" />
</p>
