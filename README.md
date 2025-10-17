# TP4 SIA - Aprendizaje No Supervisado

## Introducción

Trabajo práctico para la materia Sistemas de Inteligencia Artificial con el
objetivo de evaluar Métodos de Aprendizaje No Supervisado usando Modelos de Redes Neuronales

[Enunciado](Enunciado.pdf)

[Presentación](Presentacion.pdf) (FALTA AGREGAR)

### Requisitos

- Python3
- pip3
- [pipenv](https://pypi.org/project/pipenv/)

### Instalación

Parado en la carpeta del tp4 ejecutar

```sh
  pipenv install -r requirements.txt
```

Para instalar las dependencias necesarias en el ambiente virtual

## Ejecución
Para ejecutar el algoritmo
```
pipenv run python main.py --config-file <config-path>
```

Por ejemplo:
```
pipenv run python main.py --config-file ./config/kohonen/config.json
```

En el archivo `config.json` se encuentran todos los **hiperparámetros** que controlan el comportamiento del algoritmo.

## Uso
1. Editar el archivo `config.json` con los valores deseados.
No modificar él `csv_file` ni él `algorithm` ya que este define él .csv y el algoritmo a utilizar.
2. Ejecutar el programa principal del algoritmo.  
   El programa leerá automáticamente esta configuración y aplicará los parámetros.

---

### Algoritmo
- **`algorithm`**: algoritmo a utilizar. Opciones:
  - `kohonen`
  - `oja`

---

### Archivo de Datos
- **`csv_file`**: ruta del archivo csv con los datos de entrada.

---

### Factor de Epocas
- **`epochs_factor`**: factor de cantidad maxima de épocas (iteraciones).

---

### Radio
- **`r`**: valor del radio del vecindario inicial.

---

### Radio Constante
- **`r_constant`**: booleano para mantener el radio constante o no.

---

### Dimensión de la Grilla
- **`k`**: dimensión de la grilla (k x k).

---

### Similitud
- **`similarity_metric`**: medida de similitud. Opciones:
  - `euclidean`
  - `exponential`

---

### Epocas
- **`epochs`**: cantidad maxima de épocas (iteraciones).

---

### Tasa de aprendizaje
- **`n`**: valor de la tasa de aprendizaje, debe ser menor o igual a `0.5`