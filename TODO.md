### Verloop data in programma:

    1. Audio bestand uploaden naar server [DONE]
    2. Verwerk batch te verwerken bestanden 
    3. Copieer audiobestand naar s3 bucket [DONE]
    4. Verwerk metadata in DynamoDB
        - UUID
        - Filenaam
        - Creatiedatum
        - Uploaddatum
        - Lengte
        - Omschrijving
        - Transcriptie
    5. Laad bestand als bytestream
    6. Converteer audio naar juiste samplerate
    7. Maak een transcriptie met een huggingface model
    8. Sla transcriptie op in de DynamoDB en s3-bucket/uuid.json

