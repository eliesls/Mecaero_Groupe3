import numpy as np

NACA = '2415'

file = open(f"Naca.txt","r")

file2 = open("NACA_geo.txt","w")

counter = 0

file2.write('c1=0.01;\n')

for line in file:
    u = line.split(' ')
    x = u[0]
    y = u[len(u)-1][:-2]
    z = 0
    string = f"Point({counter}) = {{{x}, {y}, {z}, c1}};\n"
    file2.write(string)
    counter+=1

counter-=1
L = [counter] + [k for k in range(1,counter+1)]
spline_string = f"Spline(1) = {{{str(L)[1:-1]}}};"
file2.write(spline_string)

