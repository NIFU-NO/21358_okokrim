# Økonomisk kriminalitet: En oversikt over forskning og virkemidler på feltet
Dette kodedepotet inneholder skript for å trene en modell for binær klassifisering av publikasjoner basert på det tekstlige innholdet i tittel, sammendrag og nøkkelbegreper. [bert_model.py](https://github.com/NIFU-NO/21358_okokrim/blob/main/bert_model.py) inneholder kode for å trene en BERT-modell basert på et utvalg ferdig klassifiserte publikasjoner som så kan brukes til prediksjon på uklassifisert tekst. [bert_predictions.py](https://github.com/NIFU-NO/21358_okokrim/blob/main/bert_predictions.py) inneholder kode for å laste inn den trente modellen og klassifisere nye data.

Modellen ble brukt i prosjektet "Økonomisk kriminalitet: En oversikt over forskning og virkemidler på feltet", utført av NIFU på oppdrag for Justis- og beredskapsdepartmentet i perioden 2022-2023. Metoden er nærmere beskrevet i prosjektrapporten.

# Financial Crime: An Overview of Research and Measures in the Field
This code repository contains scripts to train a model for binary classification of publications based on the textual content of title, abstract, and keywords. [bert_model.py](https://github.com/NIFU-NO/21358_okokrim/blob/main/bert_model.py) contains code to train a BERT model based on a selection of pre-classified publications that can then be used for prediction on unclassified text. [bert_predictions.py](https://github.com/NIFU-NO/21358_okokrim/blob/main/bert_predictions.py) contains code to load the trained model and classify new data.

The model was used in the project "Financial Crime: An Overview of Research and Measures in the Field," conducted by NIFU on behalf of the Ministry of Justice and Public Security during the period of 2022-2023. The method is further described in the project report.
