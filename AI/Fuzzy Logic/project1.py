"""
Bu uygulama ile bulanık mantık yapısı kullanılmış ve belirli kriterlere göre hangi oranda bahşiş 
verileceği belirlenmiştir. Bahşiş işlemi yiyeceğin ve servis işleminin kalitesine göre 
verilecektir. Servis ve yiyecek kalitesinin puanlama aralığı [0-10]’dur. Ve belirli kurallara göre 
bahşiş verilecek ve bu bahşişin aralığı ise %0- ve %25 olacaktır.

Girdiler:
- Servis kalitesi
Garsonun servis yapma kalitesini ifade etmektedir (0-10).
Bulanık küme: zayıf, kabul edilebilir, harika
- Yemek kalitesi
Yiyeceğin ne kadar güzel olduğunu ifade etmektedir (0-10).
Bulanık küme: kötü, idare eder, lezzetli
Çıktılar:
- Bahşiş
Yüzde kaç oranında bahşiş verilecek? (%0 - %25)
Bulanık küme: düşük, orta, yüksek
- Kurallar:
Eğer servis iyi ya da yemek kalitesi iyi ise, bahşiş yüksek olacak.
Eğer servis idare eder durumdaysa, bahşiş orta düzeyde olacak.
Eğer servis kötü ya da yemek kalitesi de kötü ise, bahşiş düşük olacak.

"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

kalite = ctrl.Antecedent(np.arange(0, 11, 1), 'kalite')
servis = ctrl.Antecedent(np.arange(0, 11, 1), 'servis')
bahsis = ctrl.Consequent(np.arange(0, 26, 1), 'bahsis')

kalite.automf(number=3)
servis.automf(number=3)

bahsis['dusuk'] = fuzz.trimf(bahsis.universe, [0, 0, 13])
bahsis['orta'] = fuzz.trimf(bahsis.universe, [0, 13, 25])
bahsis['yuksek'] = fuzz.trimf(bahsis.universe, [13, 25, 25])

kalite.view()
bahsis.view()

kural1 = ctrl.Rule(kalite['good'] | servis['good'], bahsis['yuksek'])
kural2 = ctrl.Rule(servis['average'], bahsis['orta'])
kural3 = ctrl.Rule(servis['poor'] | kalite['poor'], bahsis['dusuk'])

bahsis_kontrol = ctrl.ControlSystem([kural1, kural2, kural3])
bahsis_belirle = ctrl.ControlSystemSimulation(bahsis_kontrol)

bahsis_belirle.input['kalite'] = 5.5
bahsis_belirle.input['servis'] = 8.9
bahsis_belirle.compute()
print(bahsis_belirle.output['bahsis'])
