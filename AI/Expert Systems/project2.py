from random import choice
from experta import *

class Dis(Fact):
    """ Dis Ile Ilgili Bilgilerin Bulundugu Sinif """
    pass


class DisDurumu(KnowledgeEngine):
    
    @Rule(Dis(durum='anlik_kanama'))
    def disFircalama(self):
        print('Eğer diş fırçalarken diş eti kanaması olursa, diş hastalığı vardır ve diş hekimine başvur...')
    
    @Rule(Dis(durum='uzun_kanama'))
    def disFircalamaIki(self):
        print('Eğer diş fırçalarken uzun süreli diş eti kanaması olursa, dişeti çekilmesi vardır\
              ve diş hekimine başvur.')  
    
    @Rule(Dis(durum='dis_eti_cekilmesi'))
    def disEtiCekilmesi(self):
        print('Eğer diş eti çekilmesi var ve diş kökü görünüyorsa, dolgu yaptır.')
    
    @Rule(Dis(durum='renk_degisimi'))
    def renkDegisimi(self):
        print('Eğer dişte yiyecek ve içeceklerden oluşan renk değişimi varsa, dişleri temizle.')
    
    @Rule(Dis(durum='morarma'))
    def morarma(self):
        print('Eğer yeni diş çıkarken morarma görünüyorsa, diş hekimine başvur.')
    
    @Rule(Dis(durum='agrisiz_curuk'))
    def agrisizCuruk(self):
        print('Eğer dişte ağrı yapmayan çürük varsa, dolgu yaptır.')
    
    @Rule(Dis(durum='ileri_derecede_curuk'))
    def ileriDereceCuruk(self):
        print('Eğer dişteki çürük ileri derecedeyse, kanal tedavisi ve dolgu yaptır.')


expert = DisDurumu()
expert.reset()
expert.declare(Dis(durum=choice(['agrisiz_curuk'])))
expert.run()
