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
  pipenv install
```

Para instalar las dependencias necesarias en el ambiente virtual

## Ejecución
Para ejecutar el algoritmo
```
pipenv run python main.py --config-file ./config/config.json
```

En el archivo `config.json` se encuentran todos los **hiperparámetros** que controlan el comportamiento del algoritmo.

## Uso
1. Editar el archivo `config.json` con los valores deseados.
No modificar él `csv_file` ya que este define él .csv a utilizar.
2. Ejecutar el programa principal del algoritmo.  
   El programa leerá automáticamente esta configuración y aplicará los parámetros.

---

### Archivo de Datos
- **`csv_file`**: ruta del archivo csv con los datos de entrada.

---

### Epocas
- **`epochs`**: cantidad maxima de épocas (iteraciones)

---

### Radio
- **`r`**: valor del radio del vecindario inicial.

---

### Dimensión de la Grilla
- **`k`**: dimensión de la grilla (k x k).

---

### Tasa de Aprendizaje
- **`n`**: valor de la tasa de aprendizaje inicial