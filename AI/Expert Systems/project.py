""" Expert Systems - Uzman Sistemler
"""

from random import choice
from experta import *


class Isik(Fact):
    pass


class KarsidanKarsiyaGecme(KnowledgeEngine):
    
    @Rule(Isik(renk='yesil'))
    def yesilIsik(self):
        print('Yesil Isik Yandi Gecebilirsiniz...')
    
    @Rule(Isik(renk='sari'))
    def sariIsik(self):
        print('Sari Isik Yandi Lutfen Bekleyin...')
    
    @Rule(Isik(renk='kirmizi'))
    def kirmiziIsik(self):
        print('Kirmizi Isik Yandi Lutfen Gecmeyiniz...')


expert = KarsidanKarsiyaGecme()
expert.reset()
expert.declare(Isik(renk=choice(['kirmizi'])))
expert.run()
