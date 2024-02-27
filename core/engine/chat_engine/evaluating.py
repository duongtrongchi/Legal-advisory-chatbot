import os
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.metrics.critique import harmfulness



os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']
openai.api_key = os.getenv('OPENAI_API_KEY')


from prompts import text_qa_template
from simpleRAG import genaration_qa


data_list = [
    {
        'question': "Tác động hạn chế cạnh tranh là gì",
        'ground_truth': "Tác động hạn chế cạnh tranh là tác động loại trừ, làm giảm, sai lệch hoặc cản trở cạnh tranh trên thị trường.",
        'answer': "Tác động hạn chế cạnh tranh là tác động loại trừ, làm giảm, sai lệch hoặc cản trở cạnh tranh trên thị trường.",
        'contexts': ["France is a country located in Western Europe.", "The Eiffel Tower is located in Paris."]
    },
    {
        'question': "Who wrote 'Romeo and Juliet'?",
        'ground_truth': "William Shakespeare",
        'answer': "William Shakespeare",
        'contexts': ["'Romeo and Juliet' is a tragedy written by William Shakespeare.", "It was first published in 1597."]
    },
    {
        'question': "What is the chemical symbol for water?",
        'ground_truth': "H2O",
        'answer': "H2O",
        'contexts': ["Water is a chemical compound composed of two hydrogen atoms and one oxygen atom.", "It is essential for life on Earth."]
    }
]
ds = Dataset.from_list(data_list)



if __name__ == "__main__":
    result = evaluate(
        ds,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
    )

    result

