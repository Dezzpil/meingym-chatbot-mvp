from mvp1.app import App
from mvp1.translater import Translater
from pprint import pprint

app = App(Translater())

# Generate search queries
queries = app.generate_search_queries('Собери тренировку ног')
print("\nGenerated search queries:")
pprint(queries)

# Search and collect documents
if queries:
    documents = app.search_and_collect_documents(queries)
    print("\nCollected documents:")
    pprint(documents)
else:
    print("\nNo search queries generated.")
