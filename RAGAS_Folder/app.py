from datasets import Dataset
import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness

data_samples = {
   'question': [
          'when was the first super bowl ?',
          'who won the most super bowls ?'
   ],
   'answer': [
        'The first superbowl was held on Jan 15 1967',
        'The most super bowls have been won by the New England Patriots'
   ],
   'contexts': [
     [
        'The First AFL-NFL World Championship Game was an american football game played on January 15, 1967 at the Los angles'
     ],
     [
       'The Green Bay Packers...Green Bay, Wisconsin.',
       'The packers compete... Football Conference'
     ]
   ],
   'ground_truth':[
     'The first superbowl was held on January 15, 1967',
     'The New England Patriots have won the superbowl a record six times'
   ]

}

dataset = Dataset.from_dict(data_samples)
score = evaluate(dataset, metrics=[faithfulness, answer_correctness])
df =score.to_pandas()
df.to_csv('score.csv',index=False)