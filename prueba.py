#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRABAJO FINAL 
Nombre Estudiante: José María Borrás Serrano
"""

from numba import jit
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Fijamos la semilla 
semilla=1234
random.seed(semilla, version=1)

# Número de coeficientes con el que trabajamos
NUM_COEF = 6
# Cotas para los coeficientes
COTA_INF = 100
COTA_SUP = -100
# Número de puntos a calcular para cada función
NUM_PTOS = 100000
# Número de decimales a los que redondeamos el ángulo al obtener los puntos del contorno
DEC = 2



########################################################
######### FUNCIONES PARA GENERAR GRÁFICAS ##############
########################################################

# Muestra una gŕafica usando plt.plot
# Lo vamos a utilizar para dibujar el contorno de los puntos de la imagen y la bola contenida en ella
def grafica_imagen_y_bola(figura1, figura2, color1='blue', color2='red',
            leyenda1 = "Frontera de la imagen.",
            leyenda2 = "Bola contenida en la imagen.",
            titulo = "Fontera de la imagen y bola de mayor radio contenida.",
            eje_x = "Eje real.", eje_y = "Eje imaginario.",
            mostrar = True,
            archivo_guardar = None):
    """Parámetros:
       figura1, figura2: figuras a dibujar, deben ser un vector de puntos
       color1, color2: color de los puntos de la gráfica
       leyenda1, leyenda2: leyenda de cada figura
       titulo: titulo del grafico 
       eje_x, eje_y: etiquetas para los ejes
       mostrar: booleano para mostrar o no la imagen, por defecto True (porque a veces queremos guardar la imagen pero no mostrarla) 
       archivo_guardar: ruta del archivo donde guardar la imagen
    """
    
    # Establecemos el tamaño de la imagen
    plt.figure(figsize = (8, 8))
    # Establecemos los límites del plot
    min_x, max_x = np.min(figura1[:, 0]), np.max(figura1[:, 0])
    min_y, max_y = np.min(figura1[:, 1]), np.max(figura1[:, 1])
    min_eje = min(min_x, min_y)
    max_eje = max(max_x, max_y)
    escala = (max_eje - min_eje)/10 # dejamos un 10% extra a la izquierda y a la derecha
    plt.xlim(min_eje - escala, max_eje + escala)
    plt.ylim(min_eje - escala, max_eje + escala)
    # Ponemos la leyenda y el título a la imagen y los ejes 
    plt.xlabel(eje_x)
    plt.ylabel(eje_y)
    plt.title(titulo)
    ley_1 = mpatches.Patch(color = color1, label = leyenda1)
    ley_2 = mpatches.Patch(color = color2, label = leyenda2)
    plt.legend(handles=[ley_1, ley_2])
    # Dibujamos las figuras
    plt.plot(figura1[:,0], figura1[:,1], c = color1) #no pone la leyenda
    plt.plot(figura2[:,0], figura2[:,1], c = color2)
    # Guardamos la imagen
    if archivo_guardar is not None:
        plt.savefig(archivo_guardar, dpi=80)
    # Mostramos la imagen
    if mostrar:
        plt.show()
    
    plt.close()
    
def grafica_figura(figura,  color1='blue',
            titulo = "Fontera de la imagen y bola de mayor radio contenida.",
            eje_x = "Eje real.", eje_y = "Eje imaginario.",
            mostrar = True,
            archivo_guardar = None):
    """Parámetros:
       x: datos de la gráfica
       color: color de los puntos de la gráfica
       y: etiquetas de los datos (por defecto None)
       etiquetas_ws: etiquetas de las rectas (por defecto None)
       titulo: titulo del grafico (por defecto None)
       ejes: título de los ejes x, y (por defecto None), 
             tiene que ser un vector con 2 elementos.
    """
    
    # Establecemos el tamaño de la imagen
    plt.figure(figsize = (8, 8))
    # Establecemos los límites del plot
    min_x, max_x = np.min(figura[:, 0]), np.max(figura[:, 0])
    min_y, max_y = np.min(figura[:, 1]), np.max(figura[:, 1])
    min_eje = min(min_x, min_y)
    max_eje = max(max_x, max_y)
    escala = (max_eje - min_eje)/10 # dejamos un 10% extra a la izquierda y a la derecha
    plt.xlim(min_eje - escala, max_eje + escala)
    plt.ylim(min_eje - escala, max_eje + escala)
    # Ponemos la leyenda y el título a la imagen y los ejes 
    plt.xlabel(eje_x)
    plt.ylabel(eje_y)
    plt.title(titulo)
    # Dibujamos las figuras
    plt.plot(figura[:,0], figura[:,1], c = color1) #no pone la leyenda
    # Guardamos la imagen
    if archivo_guardar is not None:
        plt.savefig(archivo_guardar, dpi=80)
    # Mostramos la imagen
    if mostrar:
        plt.show()
    
    plt.close()
    
def grafica_adicional(figura,  color1='blue',
            titulo = "Fontera de la imagen y bola de mayor radio contenida.",
            eje_x = "Eje real.", eje_y = "Eje imaginario.",
            mostrar = True,
            archivo_guardar = None):
    """Parámetros:
       x: datos de la gráfica
       color: color de los puntos de la gráfica
       y: etiquetas de los datos (por defecto None)
       etiquetas_ws: etiquetas de las rectas (por defecto None)
       titulo: titulo del grafico (por defecto None)
       ejes: título de los ejes x, y (por defecto None), 
             tiene que ser un vector con 2 elementos.
    """
    
    # Establecemos el tamaño de la imagen
    plt.figure(figsize = (8, 8))
    # Establecemos los límites del plot
    min_x, max_x = np.min(figura[:, 0]), np.max(figura[:, 0])
    min_y, max_y = np.min(figura[:, 1]), np.max(figura[:, 1])
    min_eje = min(min_x, min_y)
    max_eje = max(max_x, max_y)
    escala = (max_eje - min_eje)/10 # dejamos un 10% extra a la izquierda y a la derecha
    plt.xlim(min_eje - escala, max_eje + escala)
    plt.ylim(min_eje - escala, max_eje + escala)
    # Ponemos la leyenda y el título a la imagen y los ejes 
    plt.xlabel(eje_x)
    plt.ylabel(eje_y)
    plt.title(titulo)
    # Dibujamos las figuras
    plt.plot(figura[:,0], figura[:,1], c = color1) #no pone la leyenda
    
########################################################
######### FUNCIONES AUXILIARES #########################
########################################################

# devuelve el módulo del punto complejo
def obtener_modulo(punto):
    """Parámetros:
       punto: punto complejo del que obtener el módulo
    """
    return math.sqrt(punto.real**2 + punto.imag**2)

# devuelve la distancia entre dos puntos
@jit
def obtener_distancia(p0, p1):
    """Parámetros:
       p0, p1: puntos de los que obtener la distancia euclídea
    """
    return math.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)

# devuelve el baricentro de un conjunto de puntos  
@jit
def baricentro(puntos):
    """Parámetros:
       puntos: vector de puntos de los que obtener el baricentro 
    """
    b = np.zeros(2)
    for i in range(len(puntos)):
        b = b + puntos[i]
        
    return b/len(puntos)

# devuelve el ángulo polar entre dos puntos en sentido antihorario
@jit
def angulo_ah(p0, p1):
    """Parámetros:
       p0, p1: puntos de los que obtener el ángulo
    """
    angulo = math.atan2(p1[1]-p0[1], p1[0]-p0[0])
    # math.atan2 devuelve el angulo en el intervalo [-pi, pi], lo pasamos a [0, 2pi]
    if angulo < 0:
        angulo = 2*math.pi + angulo
        
    return angulo
   
########################################################################
#################### Algoritmo de ordenación ##########################
########################################################################

def primer_valor(elemento):
    return elemento[0]

def ordenar(puntos):
    ordenados = np.array(sorted(puntos, key=primer_valor))
    
    return ordenados

##########################################################################
####################  Trabajar con las funciones de F ####################
##########################################################################
    
# devuelve un string de la función
def mostrar_funcion(coef):
    """Parámetros:
       coef: vector de coeficientes de la función
    """
    string_f = str(coef[0]) + " + " + str(coef[1]) + "*z" + " + "
    for i in range(2, len(coef) - 1):
        string_f = string_f + str(coef[i]) + "*z^" + str(i) + " + "
    string_f = string_f + str(coef[len(coef) - 1]) + "*z^" + str(len(coef) - 1)
    
    return string_f

# calculamos un punto de la función
@jit # para que el cálculo sea más rápido
def calcular_punto(punto, coef):
    """Parámetros:
       punto: punto del que calcular su imagen por la función
       coef: vector de coeficientes de la función
    """
    z = complex(punto[0],punto[1])
    imagen = complex(0,0)
    for i in range(1, len(coef)): # porque el primer coeficiente siempre es 0
        imagen = imagen + coef[i]*(z**i)
        
    return imagen

# calculamos puntos de la imagen de la función
@jit 
def calcular_puntos_imagen(funcion, puntos):
    """Parámetros:
       funcion: vector de coeficientes de la función
       puntos: vector de puntos de los que calcular su imagen por la función
    """
    puntos_imagen = np.empty([len(puntos), 2])
    for i in range(len(puntos)):
        pto_im = calcular_punto(puntos[i], funcion)
        puntos_imagen[i][0] = pto_im.real
        puntos_imagen[i][1] = pto_im.imag
        
    return puntos_imagen

# obtenemos aleatoriamente los coeficientes (complejos) de la función
def obtener_funcion():
    coeficientes = np.empty(NUM_COEF, dtype=np.complex_)
    coeficientes[0] = 0 # ya que sólo sirve para trasladar la imagen
    coeficientes[1] = 1 # 1 para que la derivada en 0 valga 1
    for i in range(2, NUM_COEF):
        coeficientes[i] = complex(random.uniform(COTA_INF, COTA_SUP),random.uniform(COTA_INF, COTA_SUP))
        #coeficientes[i] = complex(random.uniform(COTA_INF, COTA_SUP),0)
        
    return coeficientes

# obtenemos aleatoriamente los coeficientes (complejos) de la función para los términos elevados a una potencia par
# para los términos con potencia impar (salvo el z) el coeficiente es 0
def obtener_funcion_2():
    coeficientes = np.empty(NUM_COEF, dtype=np.complex_)
    coeficientes[0] = 0 # ya que sólo sirve para trasladar la imagen
    coeficientes[1] = 1 # 1 para que la derivada en 0 valga 1
    for i in range(2, NUM_COEF):
        if (i % 2) == 0:
            coeficientes[i] = 0
        else:
        #coeficientes[i] = complex(random.uniform(COTA_INF, COTA_SUP),random.uniform(COTA_INF, COTA_SUP))
            coeficientes[i] = complex(random.uniform(COTA_INF, COTA_SUP),0)
        
    return coeficientes

# obtenemos aleatoriamente los coeficientes (complejos) de la función y 
# el coeficiente i-ésimo se encuentra en el intervalo [COTA_INF/i, COTA_SUP/i]
def obtener_funcion_3():
    coeficientes = np.empty(NUM_COEF, dtype=np.complex_)
    coeficientes[0] = 0 # ya que sólo sirve para trasladar la imagen
    coeficientes[1] = 1 # 1 para que la derivada en 0 valga 1
    for i in range(2, NUM_COEF):
        coeficientes[i] = complex(random.uniform(COTA_INF/i, COTA_SUP/i), random.uniform(COTA_INF/i, COTA_SUP/i))
        #coeficientes[i] = complex(random.uniform(COTA_INF/i, COTA_SUP/i),0)
        
    return coeficientes

##########################################################################
####################  Funciones específicas ##############################
##########################################################################

def funcion_prueba_1():
    NUM_COEF = 6
    coeficientes = np.empty(NUM_COEF, dtype=np.complex_)
    coeficientes[0] = 0
    coeficientes[1] = 1
    coeficientes[2] = 14
    coeficientes[3] = 8
    coeficientes[4] = -2
    coeficientes[5] = 6
    
    return coeficientes

def funcion_prueba_2():
    NUM_COEF = 6
    coeficientes = np.empty(NUM_COEF, dtype=np.complex_)
    coeficientes[0] = 0
    coeficientes[1] = 1
    coeficientes[2] = -93.29070713842776
    coeficientes[3] = 11.853480164929465
    coeficientes[4] = 98.50170598828257
    coeficientes[5] = -82.19519248982482
    
    
    return coeficientes

##########################################################################
###############  Funciones para obtener puntos del disco #################
##########################################################################

# obtenemos un punto aleatorio del disco unidad
def obtener_punto_disco():
    punto = np.empty(2)
    punto[0] = random.uniform(-1, 1) # parte real
    max_abs = math.sqrt(1 - punto[0]**2)  # máximo valor absoluto que puede tomar punto[1] tras obtener punto[0]
    punto[1] = random.uniform(-max_abs, max_abs) # parte imaginaria
    
    return punto

# obtenemos un array de puntos aleatorios del disco unidad
def puntos_aleatorios_disco(num_ptos):
    """Parámetros:
       num_ptos: número entero de puntos que obtener del disco unidad
    """
    puntos = np.empty([num_ptos,2])
    for i in range(num_ptos):
        puntos[i] = obtener_punto_disco()
        
    return puntos

# obtenemos un array de puntos de la frontera del disco unidad
@jit 
def puntos_frontera_disco(num_ptos):
    """Parámetros:
       num_ptos: número entero de puntos que obtener de la frontera disco unidad
    """
    puntos = np.empty([num_ptos,2])
    
    theta = 2*math.pi/num_ptos # ángulo que rotamos cada punto
    
    puntos[0] = np.array([1.0, 0.0])
    
    #cada nuevo punto es el anterior rotado theta radianes en sentido horario
    # seguimos: x' = xcos(theta)+  ysen(theta)
    #           y' = -xsen(theta) + ycos(theta)
    for i in range(1, num_ptos):
        parte_real = math.cos(i*theta) 
        parte_im = -math.sin(i*theta)
        puntos[i] = np.array([parte_real, parte_im])
    
    return puntos

# devuelve un array de puntos de la frontera de la bola que se le pase como argumento
def obtener_puntos_frontera_disco(bola, num_ptos):
    """Parámetros:
        bola: bola de la que obtener los puntos de su frontera, tiene que ser un array donde el primer elemento es el centro
            y el segundo elemento es el radio
       num_ptos: número entero de puntos que obtener de la frontera de la bola
    """
    centro = bola[0]
    radio = bola[1]
    puntos = np.empty([num_ptos,2])
    
    theta = 2*math.pi/num_ptos # ángulo que rotamos cada punto
    
    punto_inicial = np.array([centro[0] + radio, centro[1]])
    puntos[0] = punto_inicial
    
    for i in range(1, num_ptos):
        parte_real = radio*math.cos(i*theta) + centro[0] 
        parte_im = -radio*math.sin(i*theta) + centro[1]
        puntos[i] = np.array([parte_real, parte_im])
    
    return puntos

    
###############################################################################
######### Algoritmo para obtener los puntos del contorno de una figura ########
###############################################################################
    
# redondeo es el número de decimales a los que redondeamos el ángulo
def contorno(puntos, redondeo = 2, centro = None):
    """Parámetros:
        puntos: vector de puntos de la figura de la que obtener el contorno
        redondeo: número de cifras decimales a los que redondear los ángulos entre puntos 
        centro: punto central desde el que obtener los ángulos de los puntos
    """
    num_ptos = len(puntos)
    puntos_angulos = np.empty([num_ptos, 3]) # el primer valor es el ángulo que forma con el centro, el segundo la distancia al centro y el tercero es el índice del punto
    if (centro is None):
        #centro = baricentro(puntos)
        centro = np.array([0, 0])
    
    #quizás habría que quitar el punto si es el mismo que el centro
    for i in range(num_ptos):
        pto = puntos[i]
        if (redondeo is not None):
            puntos_angulos[i] = np.array([round(angulo_ah(centro, pto), redondeo), obtener_distancia(centro, pto), i])
        else:
            puntos_angulos[i] = np.array([angulo_ah(centro, pto), obtener_distancia(centro, pto), i])
    
    ptos_ordenados = ordenar(puntos_angulos)
    ptos_contorno = []
    angulo = ptos_ordenados[0][0]
    distancia = ptos_ordenados[0][1]
    indice = ptos_ordenados[0][2]
    
    for i in range(1, num_ptos):
        pto = ptos_ordenados[i]
        if angulo == pto[0]:
            if distancia < pto[1]:
                distancia = pto[1]
                indice = pto[2]
        else:
            ptos_contorno.append(puntos[int(indice)])
            angulo = pto[0]
            distancia = pto[1]
            indice = pto[2]
            
    ptos_contorno.append(puntos[int(indice)])
        
    return np.array(ptos_contorno)

###############################################################################
### Algoritmos para obtener la bola de mayor tamaño contenida en la figura ####
###############################################################################
    

# devuelve la distancia entre un punto (centro) y un conjunto de puntos (puntos)
def obtener_distancia_min(centro, puntos):
    """Parámetros:
        centro: punto central del cual calculamos su distancia al conjunto de puntos
        puntos: vector de puntos 
    """
    d_min = math.inf
    for punto in puntos:
        distancia = obtener_distancia(centro, punto)
        if distancia < d_min:
            d_min = distancia
            
    return d_min

# Genera los vecinos para la función obtener_bola_max
def genera_vecinos(bola, num_angulos=20, num_distancias=5):
    """Parámetros:
        bola: disco del obtener los centros vecinos, tiene que ser un array donde el primer elemento es el centro
            y el segundo elemento es el radio       
        num_angulos: número de ángulos para los vecinos
        num_distancias: número de distancias para los vecinos
    """
    centro = bola[0]
    radio = bola[1]
    vecinos = []
    
    theta = 2*math.pi/num_angulos # ángulo que rotamos cada punto
    d = radio/num_distancias # distancia que separa cada punto del centro de la bola
    
    for i in range(num_angulos):
        angulo = random.uniform(i*theta, (i+1)*theta)
        giro_parte_real = math.cos(angulo)
        giro_parte_im = -math.sin(angulo)
        for j in range(num_distancias):
            distancia = random.uniform(j*d, (j+1)*d)
            parte_real = distancia*giro_parte_real + centro[0]
            parte_im = distancia*giro_parte_im + centro[1]
            vecinos.append(np.array([parte_real, parte_im]))
        # añadimos punto cuya distancia al radio sea una centésima parte del radio
        distancia = radio/100
        parte_real = distancia*giro_parte_real + centro[0]
        parte_im = distancia*giro_parte_im + centro[1]
        vecinos.append(np.array([parte_real, parte_im]))  
    
    return np.array(vecinos)

# Realizamos una búsqueda local de la bola de mayor radio contenida en la imagen
def obtener_bola_max(puntos, num_angulos = 20, num_distancias = 5, centro = None):
    #Procedimiento Búsqueda Local del Mejor
    """Parámetros:
        puntos: vector puntos de la figura de la que obtener la bola máxima contenida     
        num_angulos: número de ángulos para los vecinos
        num_distancias: número de distancias para los vecinos
        centro: punto central desde el que iniciar la búsqueda
    """
    
    if(centro is None):
        centro = baricentro(puntos)
    #centro = np.array([0,0]) # otra opción sería usar como centro [0,0] = f(0)
    radio = obtener_distancia_min(centro, puntos)
    bola_max = np.array([centro, radio])
    fin = False
    
    while (not fin):
        fin = True
        vecinos = genera_vecinos(bola_max, num_angulos, num_distancias)
        for vecino in vecinos:
            radio_vecino = obtener_distancia_min(vecino, puntos)
            if radio_vecino > radio:
                radio = radio_vecino
                centro = vecino
                fin = False
        bola_max = np.array([centro, radio])
                 
    return bola_max

# Emplea una búsqueda global para obtener la bola de mayor radio
def obtener_bola_max_seguridad(puntos, bola, num_angulos = 20, num_distancias = 5, num_busquedas_nuevas = 30, t=0.8):
    """Parámetros:
        puntos: vector puntos de la figura de la que obtener la bola máxima contenida     
        bola: disco desde el que iniciar la búsqueda
        num_angulos: número de ángulos para los vecinos
        num_distancias: número de distancias para los vecinos
        num_busquedas_nuevas: número de búsquedas locales que llegar a realizar como máximo
        t: cercanía a los puntos de la frontera, 1 es el máximo, 0 es el mínimo
    """
    centros_iniciales = []
    long = int(len(puntos)/num_busquedas_nuevas)
    for i in range(num_busquedas_nuevas):
        indice = random.randint(i*long, (i+1)*long - 1)
        centro_nuevo = [t * puntos[indice][0], t*puntos[indice][1]] 
        if obtener_distancia(centro_nuevo, bola[0]) > bola[1]:
            centros_iniciales.append(centro_nuevo)
    
    bola_mejor = bola
    for centro in centros_iniciales:
        bola_nueva = obtener_bola_max(puntos, num_angulos, num_distancias, centro)
        if bola_mejor[1] < bola_nueva[1]:
            bola_mejor = bola_nueva 
            
    #Mostrar los nuevos_centros para la memoria
                 
    return bola_mejor


# Mostramos la circuferencia unidad, la imagen de dos funciones y el uso del contorno
print("Mostramos la circunferencia unidad, su imagen mediante la función f y el resultado del algoritmo para obtener el contorno.\n ")
ptos_frontera = puntos_frontera_disco(NUM_PTOS)
grafica_figura(ptos_frontera, titulo = "Circunferencia unidad.")

f = funcion_prueba_1()
ptos_imagen = calcular_puntos_imagen(f, ptos_frontera) 
grafica_figura(ptos_imagen, titulo = "Imagen de la circuferencia mediante la función f.")
print("La función es f(z) = " + mostrar_funcion(f))

ptos_contorno_f = contorno(ptos_imagen, 2)
grafica_figura(ptos_contorno_f, titulo = "Contorno de la imagen de la circuferencia mediante la función f.")

input("\n--- Pulsar tecla para continuar ---\n")

#print("El número de puntos en el contorno de la imagen es :" + str(len(ptos_contorno_f)))
print("Mostramos la imagen de la circunferencia unidad mediante la función g y el resultado del algoritmo para obtener el contorno.\n ")

g = funcion_prueba_2()
ptos_imagen = calcular_puntos_imagen(g, ptos_frontera) 
grafica_figura(ptos_imagen, titulo = "Imagen de la circuferencia mediante la función g.")
print("La función es g(z) = " + mostrar_funcion(g))

ptos_contorno_g = contorno(ptos_imagen, 2)
grafica_figura(ptos_contorno_g, titulo = "Contorno de la imagen de la circuferencia mediante la función g.")
#print("La función es g(z) = " + mostrar_funcion(g))

#print("El número de puntos en el contorno de la imagen es :" + str(len(ptos_contorno_g)))

input("\n--- Pulsar tecla para continuar ---\n")

# Mostramos 2 ejemplos de vecinos
print("Mostramos dos ejemplos de generación de vecinos para la bola con radio 3 y centro (4,4).\n")
bola = [[4, 4], 3]
centro = [4, 4]
ptos_bola = obtener_puntos_frontera_disco(bola, 1000)

vecinos = genera_vecinos(bola)
grafica_adicional(ptos_bola, titulo = "Ejemplo 1 de vecinos del disco.")
plt.scatter(vecinos[:, 0], vecinos[:, 1], c='red')
plt.scatter(centro[0], centro[1], c='green')
ley_1 = mpatches.Patch(color = 'blue', label = "Frontera del disco.")
ley_2 = mpatches.Patch(color = 'red', label = "Vecinos del centro del disco.")
ley_3 = mpatches.Patch(color = 'green', label = "Centro del disco.")
plt.legend(handles=[ley_1, ley_2, ley_3])
plt.show()

vecinos = genera_vecinos(bola)
grafica_adicional(ptos_bola, titulo = "Ejemplo 2 de vecinos del disco.")
plt.scatter(vecinos[:, 0], vecinos[:, 1], c='red')
plt.scatter(centro[0], centro[1], c='green')
ley_1 = mpatches.Patch(color = 'blue', label = "Frontera del disco.")
ley_2 = mpatches.Patch(color = 'red', label = "Vecinos del centro del disco.")
ley_3 = mpatches.Patch(color = 'green', label = "Centro del disco.")
plt.legend(handles=[ley_1, ley_2, ley_3])
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Mostramos la bola obtenida de la imagen de f mediante obtener_bola_max
print("Mostramos el resultado del algoritmo para obtener las bolas de mayor tamaño contenidas en la imágenes de f y g.\n ")
bola_max_f = obtener_bola_max(ptos_contorno_f)
ptos_bola_max_f = obtener_puntos_frontera_disco(bola_max_f, 1000)
radio_redondeado = round(bola_max_f[1], 5) # redodeamos a 5 cifras decimales
titulo = "Frontera de la imagen y bola contenida con radio = " + str(radio_redondeado) + "."
grafica_imagen_y_bola(ptos_contorno_f, ptos_bola_max_f, titulo = titulo, mostrar = True)
            
# Mostramos la bola obtenida de la imagen de g mediante obtener_bola_max
bola_max_g = obtener_bola_max(ptos_contorno_g)
ptos_bola_max_g = obtener_puntos_frontera_disco(bola_max_g, 1000)
radio_redondeado = round(bola_max_g[1], 5) # redodeamos a 5 cifras decimales
titulo = "Frontera de la imagen y bola contenida con radio = " + str(radio_redondeado) + "."
grafica_imagen_y_bola(ptos_contorno_g, ptos_bola_max_g, titulo = titulo, mostrar = True)
  