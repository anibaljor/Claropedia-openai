import openai
from approaches.approach import Approach
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from text import nonewlines

# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion 
# (answer) with that prompt.
class RetrieveThenReadApproach(Approach):

    template = \
"Usted es un asistente inteligente que ayuda a los empleados de atención al cliente de la compañía telefónica Claro con sus preguntas sobre los documentos de Claropedia" + \
"Responda la siguiente pregunta usando solo los datos proporcionados en las fuentes a continuación. " + \
"Para información tabular, devuélvala como una tabla html" + \
"Cada fuente tiene un nombre seguido de dos puntos y la información real, siempre incluya el nombre de la fuente para cada hecho que use en la respuesta" + \
"Si no puede responder usando las fuentes a continuación, diga que no sabe" + \
"""

###
Question: 'What is the deductible for the employee plan for a visit to Overlake in Bellevue?'

Sources:
info1.txt: Next Day AMBA (SIM CORPO): Si el pedido se cargó hasta las 16 hs. llegará a destino al día siguiente. Si la carga es posterior a dicho horario llegará a las 48 hs. hábiles.
info2.pdf:  Líneas Corporativas (segmento: N, H, T, O y Pymes) con Cecor asignado: el asesor de Delivery derivará la solicitud de compra a su supervisor vía mail con los datos del pedido (o con el mail adjunto) y cerrará la workflow como PROCESADO OK.
info3.pdf: El supervisor de Delivery es el encargado de notificar al CECOR por las vías correspondientes para que sea el CECOR quien finalmente cargue el pedido.

Respuesta:
El asesor de Delivery derivará la solicitud de compra a su supervisor vía mail con los datos del pedido (o con el mail adjunto) y cerrará la workflow como PROCESADO OK [info2.txt] Si el pedido se cargó hasta las 16 hs. llegará a destino al día siguiente. Si la carga es posterior a dicho horario llegará a las 48 hs. hábiles. [info2.pdf] El supervisor de Delivery es el encargado de notificar al CECOR para que finalice el proceso de carga [info3.pdf].

###
Pregunta: '{q}'?

Fuentes:
{retrieved}

Respuesta:
"""

    def __init__(self, search_client: SearchClient, openai_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def run(self, q: str, overrides: dict) -> any:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        if overrides.get("semantic_ranker"):
            r = self.search_client.search(q, 
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="es-es", 
                                          query_speller="lexicon", 
                                          semantic_configuration_name="default", 
                                          top=top, 
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) for doc in r]
        content = "\n".join(results)

        prompt = (overrides.get("prompt_template") or self.template).format(q=q, retrieved=content)
        completion = openai.Completion.create(
            engine=self.openai_deployment, 
            prompt=prompt, 
            temperature=overrides.get("temperature") or 0.3, 
            max_tokens=1024, 
            n=1, 
            stop=["\n"])

        return {"data_points": results, "answer": completion.choices[0].text, "thoughts": f"Question:<br>{q}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')}
