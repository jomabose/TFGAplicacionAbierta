#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRABAJO FINAL 
Nombre Estudiante: José María Borrás Serrano
"""

from numba import jit
import warnings
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

import os # para crear directorios donde guardar las imagenes
from datetime import date # para obtener la fecha
from datetime import datetime # para obtener la fecha y la hora


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

# devuelve el primer valor del elemento
def primer_valor(elemento):
    """Parámetros:
       elemento: vector del que devolvemos el primer valor 
    """
    return elemento[0]

# devuelve el conjunto de puntos ordenados según el primer valor de cada elemento de menor a mayor 
# para ordenar se utiliza el algoritmo que ya viene implementado por defecto en python
def ordenar(puntos):
    """Parámetros:
       puntos: vector de puntos que ordenar
    """
    
    return np.array(sorted(puntos, key=primer_valor))

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


def funcion1():
    # El radio de la mayor bola contenida en la imagen es: 0.77245...
    NUM_COEF = 6
    coeficientes = np.empty(NUM_COEF, dtype=np.complex_)
    coeficientes[0] = 0
    coeficientes[1] = 1
    coeficientes[2] = 0.03673432230816914
    coeficientes[3] = -0.09692491870164482
    coeficientes[4] = -0.2425685851909108
    coeficientes[5] = 0.04050226114774763
    
    return coeficientes

def funcion2():
    # El radio de la mayor bola contenida en la imagen es: 0.77273...
    NUM_COEF = 6
    coeficientes = np.empty(NUM_COEF, dtype=np.complex_)
    coeficientes[0] = 0
    coeficientes[1] = 1
    coeficientes[2] = -0.11265218483690814
    coeficientes[3] = -0.22485422962175083
    coeficientes[4] = -0.18456528508162684
    coeficientes[5] = 0.14577651034488337
    
    return coeficientes

def funcion3():
    # El radio de la mayor bola contenida en la imagen es: 0.77031...
    NUM_COEF = 6
    coeficientes = np.empty(NUM_COEF, dtype=np.complex_)
    coeficientes[0] = 0
    coeficientes[1] = 1
    coeficientes[2] = -0.03158188502790604
    coeficientes[3] = -0.14420708054309922
    coeficientes[4] = -0.20299965516529406
    coeficientes[5] = 0.06085143927231301
    
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

############################################################################################
### Algoritmo para obtener una estimación de la cota superior de la constante de Landau ####
############################################################################################
    

def cota_constante_landau(num_funciones, mostrar=False, guardar=False):
    """Parámetros:
        num_funciones: número de funciones a emplear para estimar la cota
        mostrar: booleano para determinar si mostrar las imágenes de cada función
        guardar: booleano para determinar si guardar las imágenes de cada función,
                se guardarían siguiendo la ruta "imagenes/Landau/" + str(fecha) + "/" + str(hora)
    """
    ptos_disco = puntos_frontera_disco(NUM_PTOS)
    radio_min = math.inf
    if guardar:
        fecha = date.today().strftime("%d-%m-%Y")
        hora = datetime.now().strftime("%H:%M:%S")
        directorio = "imagenes/Landau/" + str(fecha) + "/" + str(hora)
        os.makedirs(directorio)
        texto = ("Semilla que hemos fijado al principio: " + str(semilla) + 
                "\nNúmero de coeficientes con los que trabajamos: " + str(NUM_COEF) +
                "\nIntervalo para el valor de cada coeficiente: (" + str(COTA_INF) + "," + str(COTA_SUP) + ")" + 
                "\nNúmero de puntos a calcular para cada función: " + str(NUM_PTOS) +
                "\nNúmero de decimales a los que redondeamos el ángulo al obtener los puntos del contorno: " + str(DEC))
    else:
        archivo = None
    for i in range(num_funciones):
        f = obtener_funcion() # cada función se obtiene con coeficientes aleatorios en un intervalo
        ptos_imagen = calcular_puntos_imagen(f, ptos_disco) 
        ptos_contorno = contorno(ptos_imagen, DEC)
        bola_max = obtener_bola_max(ptos_contorno, NUM_ANGULOS, NUM_DISTANCIAS)
        radio = bola_max[1]
        if radio < radio_min:
            radio_min = radio
            bola_cota = bola_max
            funcion_cota = f
        if mostrar or guardar:
            if guardar:
                archivo = directorio + "/" + str(i) + ".png"
                texto = texto + "\n\nFunción " + str(i) + ": " + mostrar_funcion(f) + ". \nEl radio de la mayor bola contenida en la imagen es: " + str(radio) + "."
            ptos_bola_max = obtener_puntos_frontera_disco(bola_max, 1000)
            radio_redondeado = round(radio, 5) # redodeamos a 5 cifras decimales
            titulo = "Frontera de la imagen y bola contenida con radio = " + str(radio_redondeado) + "."
            grafica_imagen_y_bola(ptos_contorno, ptos_bola_max, titulo = titulo, mostrar = mostrar, archivo_guardar = archivo)
            if mostrar:
                print("El radio de la bola 'máxima' contenida en la imagen es: " + str(radio) + ".\n\n")
            plt.close()
    if guardar:
        texto = ("La función de la cual hemos obtenido la mejor cota es f(z) = " + mostrar_funcion(funcion_cota)
                + "\nLa cota ha sido: " + str(radio_min) + "\n\n" + texto)
        archivo_texto = open(directorio + "/archivo.txt", "a") # se le pasa el argumento 'a' para que añada el texto y cree el archivo si no existe
        archivo_texto.write(texto)
        archivo_texto.close()
    return [bola_cota, funcion_cota]



if __name__ == '__main__':
    
    # Fijamos la semilla 
    semilla=1234
    random.seed(semilla, version=1)

    # Número de coeficientes con el que trabajamos
    NUM_COEF = 10
    # Cotas para los coeficientes
    COTA_INF = 0.5
    COTA_SUP = -0.5
    # Número de puntos a calcular para cada función
    NUM_PTOS = 100000
    # Número de decimales a los que redondeamos el ángulo al obtener los puntos del contorno
    DEC = 2
    # Número de ángulos y de distancias para la generación de vecinos
    NUM_ANGULOS = 20
    NUM_DISTANCIAS = 5
    # número de funciones de las que calcular la bola de mayor radio contenida en su imagen
    NUM_FUNCIONES = 100

    warnings.filterwarnings('ignore') #para ignorar los warnings que muestra @jit

    mostrar_todas = False #para mostrar todas las imágenes generadas por pantalla
    mostrar_mejor = False #para mostrar la imagen de la función de la que se ha obtenido una mejor cota
    guardar = False #para guardar las imágenes que se van generando
    refinar_disco = True #para realizar la búsqueda global del mayor disco y así refinar la solución obtenida

    
    inicio = time.time()
    bola, funcion = cota_constante_landau(NUM_FUNCIONES, mostrar = mostrar_todas, guardar = guardar)
    fin = time.time()
    radio = bola[1]
    print("La función de la cual hemos obtenido la mejor cota es f(z) = " + mostrar_funcion(funcion))
    print("La cota ha sido: " + str(radio))
    print("Tiempo que hemos empleado en calcular la cota: " + str(fin-inicio) + " segundos.")
    print("Números de funciones que hemos usado: " + str(NUM_FUNCIONES))

    if mostrar_mejor:
        ptos_disco = puntos_frontera_disco(NUM_PTOS)
        ptos_imagen = calcular_puntos_imagen(funcion, ptos_disco) 
        ptos_contorno = contorno(ptos_imagen, DEC)
        bola = obtener_bola_max(ptos_contorno, NUM_ANGULOS, NUM_DISTANCIAS)
        radio = bola[1]
        print("La cota ha sido: " + str(radio))
        ptos_bola = obtener_puntos_frontera_disco(bola, 1000)
        radio_redondeado = round(radio, 5) # redodeamos a 5 cifras decimales
        titulo = "Frontera de la imagen y bola contenida con radio = " + str(radio_redondeado) + "."
        grafica_imagen_y_bola(ptos_contorno, ptos_bola, titulo = titulo, mostrar = True)

    if refinar_disco:
        if not mostrar_mejor: # no hace falta si ya los hemos obtenido en la parte de mostra_mejor
            ptos_disco = puntos_frontera_disco(NUM_PTOS)
            ptos_imagen = calcular_puntos_imagen(funcion, ptos_disco) 
            ptos_contorno = contorno(ptos_imagen, DEC)
            bola = obtener_bola_max(ptos_contorno, NUM_ANGULOS, NUM_DISTANCIAS)
        inicio = time.time()
        bola_max = obtener_bola_max_seguridad(ptos_contorno, bola, NUM_ANGULOS, NUM_DISTANCIAS)
        fin = time.time()
        radio_max = bola_max[1]
        print("La cota ha sido: " + str(radio))
        print("El radio de la bola 'máxima' contenida en la imagen es: " + str(radio_max) + ".\n\n")
        print("Tiempo que hemos empleado en calcular la bola: " + str(fin-inicio) + " segundos.")
        ptos_bola_max = obtener_puntos_frontera_disco(bola_max, 1000)
        radio_redondeado = round(radio_max, 5) # redodeamos a 5 cifras decimales
        titulo = "Frontera de la imagen y bola contenida con radio = " + str(radio_redondeado) + "."
        grafica_imagen_y_bola(ptos_contorno, ptos_bola_max, titulo = titulo, mostrar = True)

        titulo = "Imagen y bola contenida con radio = " + str(radio_redondeado) + "."
        grafica_imagen_y_bola(ptos_imagen, ptos_bola_max, titulo = titulo, leyenda1 = "Imagen.", mostrar = True)
