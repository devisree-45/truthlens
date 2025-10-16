from src.model_handler import OllamaHandler

handler = OllamaHandler()
prompt = """CLASSIFY the following as REAL or FAKE news. Answer ONLY in this format:
CLASSIFICATION: REAL or FAKE
CONFIDENCE: 0-100
REASONING: [explanation]

SHOCKING! Scientists confirm that aliens secretly control world governments! You won't believe what happens next!"""
resp = handler.generate_response(prompt)
print(resp['response'])
