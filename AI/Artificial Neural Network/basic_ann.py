import numpy as np
import random

learning_rate = 1 # ogrenme oranimiz
bias = 1 # yanlilik/onyargi
weights = [random.random(), random.random(), random.random()] # agirliklarimiz 2 noron ve 1 bias icin

def perceptron(input1, input2, output) -> None:
    """
    Yapay Sinir Agimizi olusturdugumuz fonksiyonumuz.
    
    Parameters
    ----------
    input1 : int
        1. Girdi degerimiz.
    input2 : int
        2. Girdi degerimiz.
    output : int
        Beklenen cikti degerimiz.
    """

    output_per = input1 * weights[0] + input2 * weights[1] + bias * weights[2]
    
    if output_per > 0: # aktivasyon fonk.
        output_per = 1
    else:
        output_per = 0
    
    error = output - output_per # modelin hatasi
    
    # agirliklari guncelleyelim
    weights[0] += error * input1 * learning_rate
    weights[1] += error * input2 * learning_rate
    weights[2] += error * bias * learning_rate

# ogrenme asamamiz
for i in range(12):
    perceptron(1, 1, 1) # T or T
    perceptron(1, 0, 1) # T or F
    perceptron(0, 1, 1) # F or T
    perceptron(0, 0, 0) # F or F


x = int(input())
y = int(input())

output_per = x * weights[0] + y * weights[1] + bias * weights[2]

if output_per > 0:
    output_per = 1
else:
    output_per = 0

print(x, ' or ', y, ' is: ', output_per)

output_per = 1 / (1 + np.exp(-output_per)) # Sigmoid Aktivasyon Fonksiyonu
