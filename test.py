import numpy as np

indicator = ["" for x in range(62)]
indicator[0] = "F"
indicator[1] = "R"

for i in range(30):
    indicator[2*(i+1)] = indicator[i]+"F"
    indicator[2*(i+1)+1] = indicator[i]+"R"

for i in range(62):
    print("y",i,": ",indicator[i])