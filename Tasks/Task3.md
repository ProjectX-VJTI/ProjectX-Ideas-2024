# Task 3:

One of the problems in Natural Language Processing (NLP) is, how do you tell whether your solution(model) works?

Say for example, you have a model that converts positive sentences to negative sentences. How will you tell that the model is converting it correctly? Moreover how do you tell how wrong or right it is?

Sure, you could sit manually grading each of the outputs it gives you, but thats way too tedious. Not to mention you can't do that with larger datasets.

So you need a metric, by which you can compare the model's output to what you know is correct.

Naively, one might suggest simply matching word for word and dividing by total number of words.

```bash
Input: This sushi is delicious
Predicted: The the sushi tastes awful
Correct: The sushi tastes awful
```

For the naive approach, this would have a 1/4 score, even though we can see the model is very close to the correct answer.

What about if we have multiple correct answers, here for example:
```bash
Correct: This sushi here does not taste good.
```
is an equally correct response, but our naive method would grade the model poorly.

Thus, all of these various requirements, give rise to a more accurate metric, that is commonly used to grade the outputs of an NLP model that is the **BLEU Score**.

The BLEU score performs n-gram matching instead of word matching and includes many adjustments like a brevity tax.

Your goal is to implement the calculation of the BLEU score metric as a python function having the following signature:

```python
def bleu_score(references, candidate) -> float:
```
where,
```python
references: list(str)
candidate: str
```
the function should return a floating point value, i.e. the bleu score

Here is are two test cases, that you must mandatorily submit output for
```bash
Reference: It is a guide to action that ensures that the military will forever heed Party commands
Reference: It is the guiding principle which guarantees the military forces always being under the command of the Party
Reference: It is the practical guide for the army always to heed the directions of the party
Candidate: It is a guide to action which ensures that the military always obeys the commands of the party
```

```bash
Reference: It is a guide to action that ensures that the military will forever heed Party commands
Reference: It is the guiding principle which guarantees the military forces always being under the command of the Party
Reference: It is the practical guide for the army always to heed the directions of the party
Candidate: It is the to action the troops forever hearing the activity guidebook that party direct
```

Notes:
1. You are **not** allowed to use external libraries (`nltk`, etc) to aid in the calculation, it is expected that you implement the calculation algorithm yourself, only using python libraries like `math` and `collections` for utilities.

2. For the sake of clarity, the BLEU Score you implement, should follow the formula mentioned in section 2.3 of the original paper proposing the bleu score(though we wont link to it, its for you to find :)).

3. Your implementation need not be exact accurate upto the decimals, as long as its approximate, its correct.

4. Only consider upto n=4 n-grams

5. Your submission will consist of your code + the screenshots of what your code gives for the above 2 test cases.

