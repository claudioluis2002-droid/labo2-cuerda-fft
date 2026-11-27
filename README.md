# Excitación modal y observabilidad en una cuerda 1D (Método implícito y FFT)

Trabajo Final - Laboratorio II  
Autor: Claudio Luis Campero  
Facultad de Ciencias Exactas - Universidad Nacional de Salta (UNSa)  
2026

## Descripción del problema

Este repositorio contiene el código de simulación, los datos generados, las figuras y el informe del estudio sobre la vibración transversal libre de una cuerda tensa homogénea de longitud L = 1.00 m con extremos fijos, velocidad inicial nula y desplazamiento inicial triangular de amplitud A = 5.0 mm (A << L) con vértice en x = a.

Se analiza cómo varía la distribución de amplitud y energía entre los modos normales al desplazar el punto de pulsación (a/L = 1/2, 1/3, 2/5) y se distingue mediante análisis espectral (FFT) en sensores virtuales (x/L = 1/2, 1/3, 3/11) qué armónicos están ausentes por falta de excitación (Bn = 0) frente a cuáles resultan invisibles por coincidencia nodal del sensor.

- Informe completo: informe/informe.pdf (fuente en informe/informe.tex)

## Estructura del repositorio

- README.md: Documentación y guía de ejecución.
- codigo/simulacion_cuerda.py: Solver implícito tridiagonal, verificación y análisis FFT.
- resultados/: Archivos CSV y resumen.json con métricas, convergencia, coeficientes modales, señales temporales y espectros FFT.
- figuras/: Gráficos generados en formato PDF y PNG.
- informe/: Código fuente LaTeX (informe.tex) y documento compilado (informe.pdf).

## Requisitos y ejecución

Requiere Python 3 con las bibliotecas numpy, scipy y matplotlib.

Para ejecutar la simulación y regenerar todos los datos y figuras desde la raíz del proyecto:

    python3 codigo/simulacion_cuerda.py

Para recompilar el informe PDF desde LaTeX:

    cd informe && pdflatex informe.tex && pdflatex informe.tex

## Limitaciones del modelo

- El modelo lineal desprecia el amortiguamiento viscoso con el aire, las pérdidas en los anclajes y la rigidez a la flexión.
- Los armónicos de alta frecuencia presentan un leve retardo de fase por dispersión numérica cerca del vértice discontinuo, conservando la energía discreta.
