
lista=[1]
while lista[-1]>0:
    n=lista[-1]/2 
    #se usa 2 por que la computadora trabaja con un codigo binario, es decir sus flotantes se encuentran en potencias de 2
    #se puede probar con numeros mas pequeños, pero generaria errores de redondeo
    lista.append(n)
print(f"xmin={lista[-2]}")