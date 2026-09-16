# Wstep teoretyczny 
`15-20 MIN`

1. Co to jest ML
  1.1 Sztuczna inteligencja
  1.2 Różnice LLM (chatgpt) a ML
  1.3 Trenowanie vs Inference
  1.4 Supervised / Unsupervised
2. Czym jest Wizja Komputerowa
  2.1 Ogólnie - oczy komputera - przykład
  2.2 Jak to działa prosto
  2.3 Gdzie jest używana
  2.4 Cloud vs Edge Devices, raspberry pi

# Praktyka 
`40-50 MIN`

3. Postawy Pythona - czemu nie cpp
  3.1 zmienne
  3.2 print
  3.3 pętle
  3.4 biblioteki
    * ultralytcis - yolo `modele`
    * opencv, PIL - wyswietlanie
4. Detekcja
  4.1 Odpalenie Modelu + logi 
    - `384x640 1 person, 43.7ms`
    - `preprocess, inference, postprocess`
  4.2 Czym jest bounding box
    - `box = results[0].boxes[0]`
    - format xyxy
    - wyciagniecie klasy + nazwa
    - eksperyment z threshhold
      - gubienie przedmiotow
      - halucynacje
    - rysowanie z opencv

5. Pose-Estimation
  5.1 Punkty przy segmentacji (format wyjscia)
  5.2 Mapa indeksow ciała 
  5.3 Odczyt konkretnego punkt u w petli
    - rysowanie nosem

6. *Segmentacja (tylko jezeli szybko idzie, lub zamiast ostatniego)*


# Praktyczny przyklad, gra we flappy-bird
`15+ MIN`

No co no niech se graja, eweuntualnie cos w kodzie do zmodyfikowania. Przyklady:
- Jest do uzupełnienia logika w
  - podnoszenie reki
  - robienie przysiadow
  - klaskanie
