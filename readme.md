# Opdracht Dominique:

### Programeer in Python een MiniPIA

1. Pak audio file met engelse gesproken tekst
2. Kopieer file naar een AWS s3 bucket
3. Laad dezelfde file van de bucket
4. Maak een transcriptie van de audiofile met een huggingface model
5. Sla de transcriptie op in een DynamoDB
6. Maak een GUI om twee schermen te tonen. Een met een lijst met files en een met de metadata en transcriptie.

### Eventueel extra opdracht:
1. Extra conversie op het audiobestand
2. Extra optimalisatie


### Hierna zalf bedachte "verbeteringen":
1. Extra scherm in GUI om:
    - files te uploaden
    - starten transciptie op specifieke bestanden
    - verwijderen files/transcriptie
    - afspelen van audio + tonen transcriptie
2. Zet bestanden om naar een UUID

### Extra informatie van Nick
1. Installeer
    - python 3.9
    - VSCode
    - AWScli
2. In VSCode installeer de extenties:
    - Jupyter
    - Python3
    - Boto3 (Python3 module voor AWS)
3. Pip install boto3
