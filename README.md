Projekt je určen pro vytvoření prediktivního modelu údržby na základě dat sbíraných přímo z výrobních strojů. Nasbíraná data v databázi jsou získána z reálného provozu.


Jak používat tento project:
Aktivace venv:
 .venv\Scripts\activate

Instalace knihoven z requirements.txt:
 pip install -r requirements.txt


Stav projektu a návrh dalšího postupu
Současný stav
Projekt je napojen na Oracle databázi pomocí SQLAlchemy. K dispozici jsou funkce pro načtení:

senzorových dat z tabulky FC_OBJECT_DATA

KPI definic z tabulky FC_KPI_DEFINITION

informací o zařízeních z tabulky EQUIPMENT

záznamů o údržbě z tabulky MAINTENANCE_ORDER

Data ze strojů jsou již předzpracovávána pomocí adaptivních strategií (preprocessing.py - OBSOLETE, lze smazat, preprocessing2.py). Vizualizační nástroje (visualization.py) umožňují sledovat trendy, rolling metriky, korelace a chybějící hodnoty.

Identifikované mezery a potřeby
1. Propojení senzorových dat se záznamy údržby
Maintenance order reprezentuje zásah prováděný nad konkrétním zařízením (EQUIPMENT). Každé zařízení je svázáno s FC_OBJECT, a senzorová data jsou vázána právě na id_fc_object. Pro jakoukoliv analýzu a predikci je nutné propojit senzorová data a záznamy údržby skrz tyto vazby.

Akce:

Ověřit a implementovat propojení MAINTENANCE_ORDER → EQUIPMENT → FC_OBJECT → FC_OBJECT_DATA.
Zatím nám v DB chybí data o vazbě mezi EQUIPMENT a FC_OBJECT kde je pouze jediný object ale equipmentů je hned několik. Equipmenty mají ve slouci EQUIPMENT_NUMBER uvedené kódové označení kdy teoreticky přes prefix můžeme vytvořit vazbu equipmentu a konkrétního KPI které zdá se používá stejný prefix? Ale to už je specifický case pro tento konkrétní dataset a my potřebujeme vytvořit univerzální funkci. Proto i kdybychom měli pozměnit tato aktuální data v databázi tak, že nebudou všechna KPI pod jediným objectem ale založíme FC_OBJECT pro každý EQUIPMENT a správně namapujeme FC_KPI_DEFINITION a FC_OBJECT a EQUIPMENT. Tím bychom měli získat správnou podobu datasetu.

Vytvořit funkci fetch_maintenance_with_object_ids().

2. Rozlišení typů údržby
Pro predikční účely nás zajímají především breakdown/corrective typy poruch. Preventivní údržby je třeba ze značení poruch odfiltrovat.

Akce:

Zajistit rozlišení typů v rámci dat z MAINTENANCE_ORDER (podle sloupce type ID_LISTING_VALUE=1 kde 1 reprezentuje typ breakdown, 3 corrective a 101 preventive).

Přidat filtr poruch podle typu.

3. Časová prodleva mezi poruchou a vytvořením/zahájením údržby
V reálném provozu často existuje prodleva mezi skutečným vznikem poruchy a vytvořením/zahájením MAINTENANCE_ORDER. Pro predikci je nejvhodnější orientovat se podle času vzniku nebo vytvoření záznamu, ne podle jeho zahájení.

Akce:

Pracovat s created_at jako s aproximací času poruchy.

Vytvořit časové okno (např. 60–180 minut před created_at) a označit data v tomto okně jako již potenciálně odpovídající poruše tedy že během této doby už porucha probíhá a analyzovat převážně ještě dřívější data.

Doporučený další postup
Fáze 1 – Sloučení dat
Implementovat funkci pro propojení záznamů údržby se senzory na základě equipment_id a id_fc_object.

Vytvořit funkci create_labeled_dataset(sensor_df, maintenance_df), která vytvoří trénovací dataset se sloupci featur a binárním labelem označujícím přítomnost poruchy.

Fáze 2 – Modelování
Použít preprocess_sensor_data2 pro generování vstupních featur.

Vytvořit datové matice X a y pro trénink klasifikačního modelu.

Implementovat základní model (např. RandomForest) v models/random_forest.py.

Fáze 3 – Evaluace a interpretace
Vyhodnotit model pomocí metrik jako přesnost, recall a F1-score.

Vizualizovat význam jednotlivých featur (feature importance).

Vytvořit ROC křivku a matici záměn.

Doporučení k implementaci
Vytvořit nový skript labeled_dataset.py v utils/, který bude:

Spojovat data senzorů a údržby.

Generovat binární label pro přítomnost poruchy.

Přidat notebook notebooks/model_training.ipynb pro experimenty:

Načtení dat.

Preprocessing.

Trénování modelu.

Vizualizace výsledků.