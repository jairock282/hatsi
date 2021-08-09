"""
 __| |______________________________________________________________________________________________________________________________| |__
(__   ______________________________________________________________________________________________________________________________   __)
   | |                                                                                                                              | |
   | |                                                          Clean data                                                          | |
   | |                                                                                                                              | |
   | |                                                  Cleans all the data frames                                                  | |
   | |  average_2_pts(): fills up all the zero values with the average of the previous and next point                               | |
   | |  difference_frame_to_frame(): fills up all the zero values with the difference between the current and the previous frame    | |
 __| |______________________________________________________________________________________________________________________________| |__
(__   ______________________________________________________________________________________________________________________________   __)
   | |                                                                                                                              | |
"""
import os
import glob
import numpy as np
import pandas as pd

##Remplace the zero values with the average of 2 points
def average_2_pts(df_origin):
    df = df_origin.copy()
    for i in range(df.shape[1]):
        test = df.iloc[:,[i]]
        index = test[(test==0).all(1)].index.tolist() ##Indexes with zero values
        if len(index) > 0:
            for j in index:
                if j > 0 and j < df.shape[0]-1:
                    pt1 = test.iloc[[j-1]]
                    pt2 = test.iloc[[j+1]]
                    df.iloc[[j],[i]] = np.average([pt1,pt2]) ##Average of the 2 points

                else:
                    if j == 0:  ##First row
                        df.iloc[[j],[i]] = test.iloc[[j+1]]

                    if j == df.shape[0]-1: ##Last row
                        df.iloc[[j],[i]] = test.iloc[[j-1]]
    return df

##Remplace the zero values with the difference between frames
def difference_frame_to_frame(df_origin):
    df = df_origin.copy()
    for i in range(df.shape[1]):
        for j in range(df.shape[0]):
            if j != 0: ##First row
                frame_actual = df_origin.iloc[[j],[i]].values[0][0]
                frame_anterior = df_origin.iloc[[j-1],[i]].values[0][0]

                df.iloc[[j],[i]] = frame_actual - frame_anterior

    df = df.drop([0]) ##Removes the first row because it doesn't have a previous value
    return df

##Saves CSV file
def GuardarArchivos(path,nombre,test):
    df = pd.DataFrame(test)
    df.to_csv(os.path.join(path, nombre))

def main():
    Ruta_origen = r'C:\Users\khmap\depthai-python\Ejemplos_Python\Datos nuevos\Todos' ##Path to the CSV files
    Ruta_Datos_Limpios = r'C:\Users\khmap\depthai-python\Ejemplos_Python\Datos nuevos\Todos\Datos_Limpios' ##Path to the saving point of the CSV files filled with an average
    Ruta_Datos_Movimiento = r'C:\Users\khmap\depthai-python\Ejemplos_Python\Datos nuevos\Todos\Datos_Movimiento' ##Path to the saving point of the CSV files filled with the difference

    Todos_Archivos = glob.glob(Ruta_origen + "/*.csv") ##Gets all the CSV files

    ##Iterates over each CSV file
    for NombreArchivo in Todos_Archivos:
        
        df = pd.read_csv(NombreArchivo,index_col=0)
        test = average_2_pts(df)
        nombre = NombreArchivo.split('\\')
        nombre = nombre.pop() ##Current file name

        print("\nNombre de archivo",nombre)
        test2 = test
        GuardarArchivos(Ruta_Datos_Limpios,nombre,test) ##Save file

        test2 =  difference_frame_to_frame(test2)
        GuardarArchivos(Ruta_Datos_Movimiento,nombre,test2) ##Save file

if __name__ == "__main__":
    main()
