from dataclasses import dataclass


@dataclass
class Felles:

    def interMetode(self):
        print("fra felles")



class Spesifik(Felles):

    def spesifik(self):
        print("spesifik")

felles = Felles()
felles.interMetode()

spesifik = Spesifik()
spesifik.interMetode()