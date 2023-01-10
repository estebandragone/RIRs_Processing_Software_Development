Código para el cálculo de los siguientes parámetros acústicos, segun norma ISO 3382: -EDT -T20 -T30 -C50 -C80 -IACC(early) -Tt (tiempo de transición) -EDTt (early decay time en el tiempo de transición

Para la ejecución correcta del mismo, se deberá contener en el mismo directorio los archivos .py siguientes: Main.py GUI.py AcousticalParameters.py

El código sólo acepta archivos .wav monofónicos y estereofónicos. Se visualiza gráfico y tabla de resultados por octava y tercio de octava (filtros IIR 2do orden butter. de acuerdo a IEC 6120) Se permite exportar los resultados en un csv.
