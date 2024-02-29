import os
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()

from datasets import Dataset

import openai

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.metrics.critique import harmfulness
import openai


os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']
openai.api_key = os.getenv('OPENAI_API_KEY')


from prompts import text_qa_template
from simpleRAG import genaration_qa

import json

if __name__ == "__main__":
    
    # Open the JSON file for reading
    with open('data.json', 'r') as file:
        # Load JSON data from the file
        data_list= json.load(file)

    ds = Dataset.from_list(data_list)
    
    result = evaluate(
        ds,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
    )

    print(result)

