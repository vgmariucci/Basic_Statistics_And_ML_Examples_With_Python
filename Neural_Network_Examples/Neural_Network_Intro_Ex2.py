##############################################################################################################################
#
# Agora que já sambemos os processos que ocorrem em uma rede neural, podemos criar uma classe para a mesma.
# Classes são os principais blocos de construção da chamada programação orientada a objetos (OOP - Objetct Oriented Programming). 
# 
# A classe RedeNeural irá gerar aleatoriamente valores inicias para as variáveis pesos e bias. Além disso, ao instanciar 
# o objeto RedeNeural, precisamos passar como parâmetro a taxa de aprendizagem alfa. Iremos escrever uma função para fazer predição 
# chamada predicao(), bem como métodos para calcular as derivadas _calcula_derivadas() e pra atualizar os parâmetros 
# __atualiza_parametros().
#  
################################################################################################################################
# Importando as bibliotecas usadas no exemplo
import numpy as np
import matplotlib.pyplot as plt

print("\n================================================================================================================")
print("\n                                        CRIANDO A CLASSE RedeNeural                                             ")
print("\n================================================================================================================")

class RedeNeural:
    
    # Método construtor
    def __init__(self, alfa):
        self.pesos = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.taxa_de_aprendizado = alfa
    
    # Método da função sigmoidal    
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Método que calcula a derivada da função sigmoidal
    def __derivada_da_sigmoid(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))
    
    # Função para realizar a predição a partir do vetores de entrada, pesos e o valor do bias
    def predicao(self, v_entrada):
        camada_1 = np.dot(v_entrada, self.pesos) + self.bias
        camada_2 = self.__sigmoid(camada_1)
        predicao = camada_2
        return predicao
    
    # Método para calcular o gradiente descendente 
    def __calcula_gradiente(self, v_entrada, v_respostas_corretas):
        camada_1 = np.dot(v_entrada, self.pesos) + self.bias
        camada_2 = self.__sigmoid(camada_1)
        predicao = camada_2
        
        derivada_erro_predicao_camada_2 = 2 * (predicao - v_respostas_corretas)
        
        derivada_predicao_camada_1 = self.__derivada_da_sigmoid(camada_1)
        
        derivada_bias_camada_1 = 1
        
        derivada_pesos_camada_1 = (0 * self.pesos) + (1 * v_entrada)
        
        valor_de_ajuste_do_bias = derivada_erro_predicao_camada_2 * derivada_predicao_camada_1 * derivada_bias_camada_1
        
        valor_de_ajuste_dos_pesos = derivada_erro_predicao_camada_2 * derivada_predicao_camada_1 * derivada_pesos_camada_1
        
        return valor_de_ajuste_do_bias, valor_de_ajuste_dos_pesos
    
    
    def __atualiza_parametros(self, valor_de_ajuste_dos_bias, valor_de_ajuste_dos_pesos):
        
        self.bias = self.bias - (valor_de_ajuste_dos_bias * self.taxa_de_aprendizado)
        
        self.pesos = self.pesos - (valor_de_ajuste_dos_pesos * self.taxa_de_aprendizado)
        
    
    def treinamento(self, v_entradas, v_respostas_corretas, numero_de_iteracoes):
        
        erros_cumulativos = []
        
        for i in range(numero_de_iteracoes):
            # Seleciona aleatoriamente um dado de treinamento
            indice_aleatorio = np.random.randint(len(v_entradas))
            
            vetor_entrada = v_entradas[indice_aleatorio]
            vetor_resposta_correta = v_respostas_corretas[indice_aleatorio]
        
            # Calcula o gradiente e atualiza os pesos e o bias
            valor_de_ajuste_do_bias, valor_de_ajuste_dos_pesos = self.__calcula_gradiente(vetor_entrada, vetor_resposta_correta)
        
            self.__atualiza_parametros(valor_de_ajuste_do_bias, valor_de_ajuste_dos_pesos)
        
            # Se a iteração é múltipla de 100, executamos as linhas de código a seguir para ver como o erro se comporta a cada 100 iterações
            if i % 100 == 0:
                
                erro_cumulativo = 0
            
                # Varre todos os valores de erros gerados em cada iteração
                for indice_de_dado in range(len(v_entradas)):
                    
                    dado_de_entrada = v_entradas[indice_de_dado]
                    resposta_correta = v_respostas_corretas[indice_de_dado]
                    
                    predicao = self.predicao(dado_de_entrada)
                    erro = np.square(predicao - resposta_correta)
                    
                    erro_cumulativo = erro_cumulativo + erro
                    
                erros_cumulativos.append(erro_cumulativo)
        
        return erros_cumulativos    
        
##############################################################################################################################
#
# Uma vez que a nossa classe RedeNeural está criada, basta criar uma instancia da mesma e chamar a função .predicao() para
# que a rede realize uma predição para as respostas de saída em função das variáveis de entrada
#
##############################################################################################################################

# Antes de instanciar a nossa Rede Neural, primeiramente escolhemos o valor da taxa de aprendizagem alfa
alfa = 0.1

# Instanciando a nossa Rede Neural
rede_neural = RedeNeural(alfa)

# Declarando os vetores de entrada:
v_entrada = np.array([1.5 , 2])

valor_predito = rede_neural.predicao(v_entrada)

print(f"\n O valor predito pela rede neural foi: {valor_predito}")

################################################################################################################################
#
# Apesar da nossa rede neural estar realizando as predições, ainda precisamo treiná-la. O objetivo é fazer a rede aprender a 
# detectar padrões de respostas corretas em função dos dados de entrada. Isso significa que a rede neural precisa saber se 
# adaptar aos novos dados de entrada que possuam a mesma distribuição de probabilidade que os dados de treinamento.
#
# O Gradiente Descendente Estocástico é uma técnica em que, a cada iteração, o modelo (rede neural) realiza uma predição
# para dados de treinamento selecionados aleatoriamente, ou seja, de maneira estocástica, calculando o erro e atualiza os
# parâmetros.
#
# Para isso, precisamos criar um método para a classe RedeNeural que irá treinar a nossa rede neural com dados de treinamento.
#
# Também iremos salvar os erros de cada iteração com o objetivo de mostrar através de um gráfico como o erro se comporta 
# conforme o número de iterações aumenta.
###############################################################################################################################
print("\n================================================================================================================")
print("\n                                    TREINANDO A REDE NEURAL COM MAIS DADOS                                      ")
print("\n================================================================================================================")

# Dataset 1
# Valores de entrda
v_entrada = np.array(
                    [
                        [3, 1.5], 
                        [2, 1], 
                        [4, 1.5], 
                        [3, 4], 
                        [3.5, 0.5], 
                        [2, 0.5], 
                        [5.5, 1], 
                        [1, 1]
                    ]
                   )

# Valores de saída
v_respostas_corretas = np.array([0, 1, 0, 1, 0, 1, 1, 0])

# Dataset 2
# Valores de entrda
# v_entrada = np.array(
#                     [
#                         [3, 1.5], 
#                         [2, 1], 
#                         [4, 1.5], 
#                         [3, 4], 
#                         [3.5, 0.5], 
#                         [2, 0.5], 
#                         [5.5, 1], 
#                         [1, 1],
#                         [4, 2.5], 
#                         [2.5, 1], 
#                         [2.5, 1.5], 
#                         [1, 0.7], 
#                         [1, 1], 
#                         [2.2, 0.5], 
#                         [3.7, 2], 
#                         [3, 3]
#                     ]
#                    )

# Valores de saída
# v_respostas_corretas = np.array([0, 1, 0, 1, 0, 1, 1, 0,
#                                  1, 0, 0, 0, 0, 0, 1, 1])

numero_de_iteracoes = 10000

erro_de_treinamento = rede_neural.treinamento(v_entrada, v_respostas_corretas, numero_de_iteracoes )

plt.plot(erro_de_treinamento)
plt.xlabel("Iterações")
plt.ylabel("Erro durante as iterações")
plt.show()

##########################################################################################################
# 
# A partir do gráfico do Erro em função das iterações, podemos ver que o erro total começa com um valor
# elevado e tende a um valor relativamente menor e oscila ao redor de um valor médio. 
# Essa oscilações bruscas decorrem da seleção aleatória dos dados de treinamento, bem como, por termos uma 
# quantidade pequena de dados.
# 
# Não é recomendado usar os dados de treinamento para avaliar a performance da rede neural, pois são dados
# que após o treinameto ela já estará sabendo como responder em função das entradas. Isso pode levar ao caso
# de overfiting, quando a rede neural fica tão boa em prever os dados de treinamento que não é capaz de 
# generalizar para dados novos.
#
# Neste exemplo o objetivo principal é entender os fundamentos básicos de construção de uma rede neural e
# por isso usamos um conjunto reduzido de dados. Geralmente, modelos de aprendizado profundo (deep learning)
# precisam de uma quantidade grande de dados devido às complexidades de certos problemas, por exemplo, o
# reconhecimento de imagens ou sinais de áudio, entre outros. Devido aos diferentes níveis de complexidade,
# usar apenas uma ou duas camadas na rede neural não é sufiente, de modo que chamamos de aprendizado profundo
# justamente o fato da rede neural ser composta por muitas camadas. 
#
# Adicionando mais camadas à rede neural e vários tipos de funções de ativação, aumentamos o poder de
# predição da mesma. Um exemplo de aplicação desse nível de complexidade é o reconhecimento facial, como
# alguns celulares possuem ao desbloquear a tela quando reconhce pela imagem quem é o dono ou quem foi 
# cadastrado para poder usar.
############################################################################################################
