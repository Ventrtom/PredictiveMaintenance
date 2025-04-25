"""
Připravit LLM orchestrator. Ten by měl využívat LLM API (Open AI, Gemini atd.) a sám řídit celý proces predikce

Tedy na základě základních předzpracovaných dat vyhodnotit který z prediktivních přístupů by měl být použit jako první
a spustit jeho provedení, analyzovat výsledky a případně použít jiný model pokud nebudou dostačující.
Stejná logika by pak měla jít použít i namísto adaptive_logic.py kde jsou zatím jen primitivní pravidla 
rozhodující jaké předzpracování dat se má použít, ale lépe by to mohl rozhodnout nějaký LLM možná.
"""