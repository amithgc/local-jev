"""Turn gold-labelled public datasets into System One questions.

Every example is stored in Jev's wire format -- {"state", "question", "gold"} --
and rendered through the same compiler the server uses, so training and
inference can never drift apart.

The goal is a model that learns *to judge from a description*, not one that
memorises forty label sets. So each source dataset is expanded into many
surface forms: option order is always shuffled, option subsets are sampled,
an "other" option sometimes replaces the true class, keys change style,
descriptions come and go, instructions are paraphrased, states switch between
strings, objects and arrays, and every class can also be asked as a yes/no.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable

MAX_CHARS = 1400
OTHER_KEYS = ["other", "none_of_the_above", "none", "unknown", "something_else", "not_listed"]
OTHER_DESCS = ["Anything that does not fit the other options", "None of the other options apply",
               "Does not match any listed option", ""]


def clip(text, limit=MAX_CHARS) -> str:
    text = re.sub(r"[ \t]+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.75)
    return text[:head] + " [...] " + text[-(limit - head):]


def human(label: str) -> str:
    label = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(label))
    return re.sub(r"[_\-./|]+", " ", label).strip().lower()


def style_key(label: str, pick: float) -> str:
    """One key style per question -- real users are consistent within a question."""
    words = human(label).split()
    if pick < 0.45:
        return "_".join(words)
    if pick < 0.75:
        return " ".join(words)
    if pick < 0.9:
        return " ".join(w.capitalize() for w in words)
    return "-".join(words)


@dataclass
class Task:
    name: str
    source: tuple                       # (hf repo, config, train split, eval split)
    kind: str                           # "classify" | "binary" | "score"
    state: Callable[[dict], object]     # row -> default state (str or dict)
    label: Callable[[dict], object]     # row -> class label / bool / level index
    classes: dict = field(default_factory=dict)      # label -> list of descriptions (may be empty)
    instructions: list = field(default_factory=list)
    noul: list = field(default_factory=list)         # templates with {x}; for binary: full statements
    noul_criteria: list = field(default_factory=list)   # binary only: (true, false) pairs
    levels: list = field(default_factory=list)       # score only: list of alternative level-description lists
    fields: list = field(default_factory=list)       # alternative key names when wrapping a string state
    holdout: bool = False
    label_names: Callable | None = None              # dataset -> list of names, for ClassLabel features
    keep: Callable[[dict], bool] | None = None


def _wrap_state(task: Task, state, rng: random.Random):
    """Vary how the same content is presented: bare string, named field, or array."""
    if not isinstance(state, str):
        return state                     # instructions may name these fields; keep them
    pick = rng.random()
    if pick < 0.6 or not task.fields:
        return state
    if pick < 0.95:
        return {rng.choice(task.fields): state}
    return [state]


def make_choice(task: Task, row: dict, labels: list[str], rng: random.Random) -> dict | None:
    gold = task.label(row)
    if gold not in labels:
        return None
    n = len(labels)
    if n > 26:
        size = rng.choice([4, 6, 8, 12, 16, 20, 26] if rng.random() < 0.85 else [30, 40, 60])
    elif n > 5:
        size = n if rng.random() < 0.5 else rng.randint(3, n)
    else:
        size = n if rng.random() < 0.8 else rng.randint(2, n)
    size = min(size, n)
    others = [l for l in labels if l != gold]
    rng.shuffle(others)
    # "other" must be the right answer only about as often as any other option. An earlier build made it
    # correct 62% of the time it appeared, and the model learned to pick "Other" whenever it saw one.
    use_other = size < n and rng.random() < 0.08
    chosen = others[: size - 1] + ([] if use_other else [gold])
    with_desc = rng.random() < 0.65
    style = rng.random()
    styled = {}
    for label in chosen:
        descs = task.classes.get(label) or []
        styled[label] = (style_key(label, style), rng.choice(descs) if (descs and with_desc) else None)
    if len({k for k, _ in styled.values()}) != len(styled):
        return None
    options = [styled[l] for l in chosen]
    if use_other or (size < n and rng.random() < 0.30):
        options.append((rng.choice(OTHER_KEYS), rng.choice(OTHER_DESCS) or None))
    gold_key = styled[gold][0] if gold in styled else options[-1][0]
    rng.shuffle(options)
    instructions = rng.choice(task.instructions) if task.instructions and rng.random() < 0.92 else None
    question = {"type": "choice", "criteria": {k: d for k, d in options}}
    if instructions:
        question["instructions"] = instructions
    return {"question": question, "gold": [k for k, _ in options].index(gold_key)}


def make_class_noul(task: Task, row: dict, labels: list[str], rng: random.Random) -> dict | None:
    gold = task.label(row)
    if gold not in labels or not task.noul:
        return None
    positive = rng.random() < 0.5
    target = gold if positive else rng.choice([l for l in labels if l != gold])
    question = {"type": "noul", "instructions": rng.choice(task.noul).format(x=human(target))}
    descs = task.classes.get(target) or []
    if descs and rng.random() < 0.3:
        question["criteria"] = {"true": rng.choice(descs)}
    return {"question": question, "gold": 0 if positive else 1}


def make_binary(task: Task, row: dict, rng: random.Random) -> dict:
    truth = bool(task.label(row))
    pick = rng.random()
    if pick < 0.75 or not task.classes:
        question = {"type": "noul", "instructions": rng.choice(task.noul)}
        if task.noul_criteria and rng.random() < 0.35:
            yes, no = rng.choice(task.noul_criteria)
            question["criteria"] = {"true": yes, "false": no} if rng.random() < 0.6 else {"true": yes}
        return {"question": question, "gold": 0 if truth else 1}
    labels = list(task.classes)                       # [positive, negative]
    return make_choice(Task(task.name, task.source, "classify", task.state,
                            lambda r: labels[0] if task.label(r) else labels[1],
                            task.classes, task.instructions), row, labels, rng)


def make_score(task: Task, row: dict, rng: random.Random) -> dict:
    level = int(task.label(row))
    levels = rng.choice(task.levels)
    question = {"type": "score", "instructions": rng.choice(task.instructions), "criteria": list(levels)}
    return {"question": question, "gold": level}


def make_example(task: Task, row: dict, labels: list[str], rng: random.Random) -> dict | None:
    if task.kind == "classify":
        made = (make_class_noul if (task.noul and rng.random() < 0.35) else make_choice)(task, row, labels, rng)
    elif task.kind == "binary":
        made = make_binary(task, row, rng)
    else:
        made = make_score(task, row, rng)
    if made is None:
        return None
    state = task.state(row)
    if isinstance(state, str):
        state = clip(state)
        if len(state) < 3:
            return None
    elif isinstance(state, list):
        state = [clip(v, MAX_CHARS // max(1, len(state))) for v in state]
    else:
        state = {k: clip(v, MAX_CHARS // max(1, len(state))) if isinstance(v, str) else v for k, v in state.items()}
    made.update(state=_wrap_state(task, state, rng), task=task.name)
    return made


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
def _names(column):
    return lambda ds: ds.features[column].names


TOPIC_NOUL = ["This {{noun}} is about {x}.", "Is this {{noun}} about {x}?", "The topic is {x}",
              "The {{noun}} belongs in the category: {x}", "Does the {{noun}} concern {x}?"]


def _topic(noun):
    return [t.replace("{{noun}}", noun) for t in TOPIC_NOUL]


INTENT_NOUL = ["The user wants: {x}", "The user's intent is {x}.", "Is the user asking about {x}?",
               "This request is a '{x}' request"]
INTENT_INSTR = ["What is the user's intent", "Which intent best matches this request", "Classify the request",
                "What does the user want to do?", "Route this request to the right handler"]

SENTIMENT_5 = [
    ["Very negative", "Negative", "Neutral or mixed", "Positive", "Very positive"],
    ["Strongly dislikes it; harsh criticism", "Mostly unhappy; more complaints than praise",
     "Mixed or indifferent", "Mostly happy; more praise than complaints", "Loves it; enthusiastic praise"],
    ["1 star: terrible experience", "2 stars: poor, significant problems", "3 stars: average, some good and some bad",
     "4 stars: good, minor issues at most", "5 stars: excellent, nothing to fault"],
]

TASKS: list[Task] = [
    # ---- topic / category ----------------------------------------------------
    Task("ag_news", ("fancyzhx/ag_news", None, "train", "test"), "classify",
         lambda r: r["text"], lambda r: r["_label"],
         {"world": ["International news, politics, conflict and diplomacy"], "sports": ["Sport, athletes, matches and competitions"],
          "business": ["Companies, markets, the economy and finance"], "sci_tech": ["Science, technology, software and the internet"]},
         ["What is this news article about", "Which section of the newspaper does this belong in", "Classify the news topic"],
         _topic("article"), fields=["article", "text", "headline"], holdout=True,
         label_names=lambda ds: ["world", "sports", "business", "sci_tech"]),
    Task("dbpedia", ("fancyzhx/dbpedia_14", None, "train", "test"), "classify",
         lambda r: {"title": r["title"], "abstract": r["content"].strip()}, lambda r: r["_label"], {},
         ["What kind of thing does this encyclopedia entry describe", "Classify the entity type", "Which category fits this entry"],
         ["The entry describes a {x}.", "Is this entry about a {x}?", "Entity type: {x}"],
         label_names=lambda ds: ["company", "educational institution", "artist", "athlete", "office holder", "means of transportation",
                                 "building", "natural place", "village", "animal", "plant", "album", "film", "written work"]),
    Task("newsgroups", ("SetFit/20_newsgroups", None, "train", "test"), "classify",
         lambda r: r["text"], lambda r: r["label_text"], {},
         ["Which discussion forum was this posted to", "What is this post about", "Pick the best-matching forum topic"],
         _topic("post"), fields=["post", "message", "text"], keep=lambda r: len(r["text"].strip()) > 40),
    Task("bbc_news", ("SetFit/bbc-news", None, "train", "test"), "classify",
         lambda r: r["text"], lambda r: r["label_text"],
         {"tech": ["Technology, gadgets and the internet"], "business": ["Companies, markets and the economy"],
          "sport": ["Sport and athletes"], "entertainment": ["Film, music, TV and celebrities"], "politics": ["Government, elections and policy"]},
         ["What is this article about", "Classify the article"], _topic("article"), fields=["article", "text"]),
    Task("fin_topic", ("zeroshot/twitter-financial-news-topic", None, "train", "validation"), "classify",
         lambda r: r["text"], lambda r: r["_label"], {},
         ["What is this financial news post about", "Classify the finance topic", "Which desk should read this"],
         _topic("post"), fields=["tweet", "post", "text"],
         label_names=lambda ds: ["analyst update", "central banks", "company or product news", "treasuries and corporate debt", "dividend",
                                 "earnings", "energy and oil", "financials", "currencies", "general news or opinion", "gold, metals and materials",
                                 "ipo", "legal and regulation", "mergers, acquisitions and investments", "macro", "markets", "politics",
                                 "personnel change", "stock commentary", "stock movement"]),
    Task("news_cat", ("valurank/News_Articles_Categorization", None, "train", None), "classify",
         lambda r: r["Text"], lambda r: r["Category"], {},
         ["Which category does this article belong to", "Classify the article"], _topic("article"), fields=["article", "text"]),
    Task("student_q", ("SetFit/student-question-categories", None, "train", "test"), "classify",
         lambda r: r["text"], lambda r: r["label_text"], {},
         ["Which school subject is this question from", "Classify the subject"], ["This is a {x} question.", "Subject: {x}"],
         fields=["question"]),
    Task("trec", ("SetFit/TREC-QC", None, "train", "test"), "classify",
         lambda r: r["text"], lambda r: r["label_coarse_text"], {},
         ["What kind of answer is this question looking for", "Classify the expected answer type"],
         ["The question asks for {x}.", "Expected answer type: {x}"], fields=["question"]),
    Task("language_id", ("papluca/language-identification", None, "train", "test"), "classify",
         lambda r: r["text"], lambda r: r["labels"],
         {"ar": ["Arabic"], "bg": ["Bulgarian"], "de": ["German"], "el": ["Greek"], "en": ["English"], "es": ["Spanish"], "fr": ["French"],
          "hi": ["Hindi"], "it": ["Italian"], "ja": ["Japanese"], "nl": ["Dutch"], "pl": ["Polish"], "pt": ["Portuguese"], "ru": ["Russian"],
          "sw": ["Swahili"], "th": ["Thai"], "tr": ["Turkish"], "ur": ["Urdu"], "vi": ["Vietnamese"], "zh": ["Chinese"]},
         ["Which language is this written in", "Detect the language"], [], fields=["text"]),
    # ---- intent -------------------------------------------------------------
    Task("banking77", ("mteb/banking77", None, "train", "test"), "classify",
         lambda r: r["text"], lambda r: r["label_text"], {}, INTENT_INSTR + ["Which banking issue is the customer describing"],
         INTENT_NOUL, fields=["message", "query", "customer_message"], holdout=True),
    Task("massive_intent", ("mteb/amazon_massive_intent", "en", "train", "test"), "classify",
         lambda r: r["text"], lambda r: r["label_text"], {}, INTENT_INSTR, INTENT_NOUL, fields=["utterance", "command", "query"]),
    Task("massive_scenario", ("mteb/amazon_massive_scenario", "en", "train", "test"), "classify",
         lambda r: r["text"], lambda r: r["label_text"], {},
         ["Which assistant skill should handle this", "What domain is this request about"], INTENT_NOUL, fields=["utterance", "query"]),
    Task("clinc", ("clinc/clinc_oos", "plus", "train", "test"), "classify",
         lambda r: r["text"], lambda r: r["_label"], {}, INTENT_INSTR, INTENT_NOUL, fields=["query", "message"],
         label_names=_names("intent")),
    Task("bitext_intent", ("bitext/Bitext-customer-support-llm-chatbot-training-dataset", None, "train", None), "classify",
         lambda r: r["instruction"], lambda r: r["intent"], {},
         ["What does the customer want", "Classify the support request", "Which support workflow applies"], INTENT_NOUL,
         fields=["message", "ticket", "customer_message"]),
    Task("bitext_category", ("bitext/Bitext-customer-support-llm-chatbot-training-dataset", None, "train", None), "classify",
         lambda r: r["instruction"], lambda r: r["category"], {},
         ["Which team should handle this", "Route the ticket to a department", "Determine the broad category of this support ticket"],
         ["This ticket is about {x}.", "Should the {x} team handle this?"], fields=["message", "ticket"]),
    Task("ticket_queue", ("Tobi-Bueck/customer-support-tickets", None, "train", None), "classify",
         lambda r: {"subject": r["subject"] or "", "body": r["body"]}, lambda r: r["queue"], {},
         ["Which team should handle this ticket", "Route this ticket to the right queue"], ["This ticket belongs to {x}."],
         keep=lambda r: r["language"] == "en" and r["body"]),
    Task("ticket_type", ("Tobi-Bueck/customer-support-tickets", None, "train", None), "classify",
         lambda r: {"subject": r["subject"] or "", "body": r["body"]}, lambda r: r["type"],
         {"Incident": ["Something is broken or failing right now"], "Request": ["The customer asks for something to be done or provided"],
          "Problem": ["An underlying or recurring issue that needs investigation"], "Change": ["A request to modify a configuration, plan or setup"]},
         ["What type of ticket is this", "Classify the ticket"], ["This ticket is a {x}."],
         keep=lambda r: r["language"] == "en" and r["body"]),
    # ---- emotion / sentiment ---------------------------------------------
    Task("emotion", ("dair-ai/emotion", "split", "train", "test"), "classify",
         lambda r: r["text"], lambda r: r["_label"], {},
         ["Which emotion does the author express", "What is the writer feeling"],
         ["The author expresses {x}.", "Does the writer feel {x}?"], fields=["text", "message"], holdout=True,
         label_names=_names("label")),
    Task("tweet_sentiment", ("cardiffnlp/tweet_eval", "sentiment", "train", "test"), "classify",
         lambda r: r["text"], lambda r: r["_label"],
         {"negative": ["Unhappy, critical or hostile"], "neutral": ["Factual or without clear feeling"], "positive": ["Happy, approving or excited"]},
         ["What is the sentiment of this tweet", "What is the tone of this message?"], ["The sentiment is {x}.", "Is the tone {x}?"],
         fields=["tweet", "text"], label_names=_names("label")),
    Task("tse_sentiment", ("mteb/tweet_sentiment_extraction", None, "train", "test"), "classify",
         lambda r: r["text"], lambda r: r["label_text"], {},
         ["Classify the sentiment", "How does the author feel"], ["The sentiment is {x}."], fields=["text", "post"]),
    Task("hate_offensive", ("tdavidson/hate_speech_offensive", None, "train", None), "classify",
         lambda r: r["tweet"], lambda r: ["hate speech", "offensive language", "neither"][int(r["class"])],
         {"hate speech": ["Attacks or demeans a group based on identity"], "offensive language": ["Crude or insulting but not targeting a protected group"],
          "neither": ["Not hateful and not offensive"]},
         ["Moderate this post", "Which moderation label applies"], ["This post contains {x}."], fields=["post", "tweet"]),
    # ---- binary -------------------------------------------------------------
    Task("sms_spam", ("ucirvine/sms_spam", None, "train", None), "binary", lambda r: r["sms"], lambda r: int(r["label"]) == 1,
         {"spam": ["Unsolicited advertising, scams or prize notifications"], "legitimate": ["A normal personal or transactional message"]},
         ["Is this message spam", "Classify this text message"],
         ["This message is spam.", "Is this text message spam?", "The message is unsolicited advertising or a scam"],
         [("Unsolicited advertising, prizes, or scams", "A legitimate conversation")], fields=["sms", "message"], holdout=True),
    Task("enron_spam", ("SetFit/enron_spam", None, "train", "test"), "binary",
         lambda r: {"subject": r["subject"] or "", "body": r["message"] or ""}, lambda r: int(r["label"]) == 1,
         {"spam": ["Unsolicited bulk email, scams or junk advertising"], "ham": ["A legitimate work or personal email"]},
         ["Is this email spam", "Classify this email"],
         ["This email is spam.", "Is this email spam or junk?", "This is unsolicited bulk email", "The email is a legitimate message from a real correspondent"],
         [("Unsolicited bulk mail, scams or junk advertising", "A legitimate work or personal email")]),
    Task("phishing", ("zefang-liu/phishing-email-dataset", None, "train", None), "binary",
         lambda r: r["Email Text"], lambda r: r["Email Type"] == "Phishing Email",
         {"phishing": ["Tries to trick the reader into giving money, credentials or personal data"], "safe": ["A legitimate email"]},
         ["Is this email a phishing attempt", "Classify this email"],
         ["This email is a scam or phishing attempt.", "Is this email phishing?"],
         [("It tries to trick the reader into handing over money, credentials or data", "It is an ordinary legitimate email")],
         fields=["email", "body"], holdout=True, keep=lambda r: r["Email Text"] and len(r["Email Text"]) > 30),
    Task("imdb", ("stanfordnlp/imdb", None, "train", "test"), "binary", lambda r: r["text"], lambda r: int(r["label"]) == 1,
         {"positive": ["The reviewer liked the film"], "negative": ["The reviewer disliked the film"]},
         ["Is this review positive or negative", "What is the reviewer's verdict"],
         ["The reviewer liked the film.", "Is this a positive review?", "The review is negative", "The reviewer recommends this movie"],
         [("The reviewer liked it", "The reviewer disliked it")], fields=["review"]),
    Task("sst2", ("stanfordnlp/sst2", None, "train", "validation"), "binary", lambda r: r["sentence"], lambda r: int(r["label"]) == 1,
         {"positive": [], "negative": []}, ["What is the sentiment"], ["The sentiment is positive.", "Is the sentence positive in tone?"],
         [], fields=["sentence", "text"]),
    Task("amazon_polarity", ("mteb/amazon_polarity", None, "train", "test"), "binary", lambda r: r["text"], lambda r: int(r["label"]) == 1,
         {"satisfied": ["The customer is happy with the product"], "dissatisfied": ["The customer is unhappy with the product"]},
         ["Is the customer satisfied"], ["The customer is happy with the product.", "Is this a positive product review?", "The customer is complaining"],
         [("The customer is satisfied", "The customer is dissatisfied")], fields=["review"]),
    Task("offensive", ("cardiffnlp/tweet_eval", "offensive", "train", "test"), "binary", lambda r: r["text"], lambda r: int(r["label"]) == 1,
         {"offensive": ["Contains insults, profanity or targeted abuse"], "not_offensive": ["Civil, even if critical"]},
         ["Moderate this post"], ["This post is offensive.", "Does this post contain offensive language?"], [], fields=["post", "tweet"]),
    Task("irony", ("cardiffnlp/tweet_eval", "irony", "train", "test"), "binary", lambda r: r["text"], lambda r: int(r["label"]) == 1,
         {}, [], ["The author is being ironic or sarcastic.", "Is this tweet sarcastic?"], [], fields=["tweet"]),
    Task("hate", ("cardiffnlp/tweet_eval", "hate", "train", "test"), "binary", lambda r: r["text"], lambda r: int(r["label"]) == 1,
         {}, [], ["This post contains hate speech.", "Does this post attack people based on their identity?"], [], fields=["post"]),
    Task("toxic", ("mteb/toxic_conversations_50k", None, "train", "test"), "binary", lambda r: r["text"], lambda r: int(r["label"]) == 1,
         {"toxic": ["Rude, disrespectful or likely to make someone leave a discussion"], "not_toxic": ["A civil comment"]},
         ["Moderate this comment"], ["This comment is toxic.", "Is this comment rude or disrespectful?", "The comment is civil"],
         [("Rude, disrespectful or hostile", "Civil, even when disagreeing")], fields=["comment"]),
    Task("insincere", ("SetFit/insincere-questions", None, "train", "test"), "binary", lambda r: r["text"], lambda r: int(r["label"]) == 1,
         {}, [], ["This question is insincere: it makes a statement rather than seeking an answer.", "Is this a genuine question?"], [],
         fields=["question"]),
    Task("clickbait", ("marksverdhei/clickbait_title_classification", None, "train", None), "binary",
         lambda r: r["title"], lambda r: int(r["clickbait"]) == 1,
         {"clickbait": ["A teasing headline engineered for clicks"], "news": ["A straightforward informative headline"]},
         ["Is this headline clickbait"], ["This headline is clickbait.", "Is this a clickbait title?"], [], fields=["headline", "title"], holdout=True),
    Task("subjectivity", ("SetFit/subj", None, "train", "test"), "binary", lambda r: r["text"], lambda r: r["label_text"] == "subjective",
         {"subjective": ["Expresses an opinion or feeling"], "objective": ["States a fact or describes events"]},
         ["Is this sentence subjective or objective"], ["This sentence expresses an opinion.", "Is this statement subjective?", "The sentence is a plain statement of fact"], [],
         fields=["sentence"]),
    Task("cola", ("nyu-mll/glue", "cola", "train", "validation"), "binary", lambda r: r["sentence"], lambda r: int(r["label"]) == 1,
         {}, [], ["This sentence is grammatically acceptable English.", "Is this sentence grammatical?"], [], fields=["sentence"]),
    Task("counterfactual", ("mteb/amazon_counterfactual", "en", "train", "test"), "binary", lambda r: r["text"], lambda r: int(r["label"]) == 1,
         {}, [], ["The text describes something that did not actually happen (a counterfactual, e.g. 'I wish it had...')."], [], fields=["review"]),
    Task("climate", ("climatebert/climate_detection", None, "train", "test"), "binary", lambda r: r["text"], lambda r: int(r["label"]) == 1,
         {}, [], ["This paragraph is about climate or environmental topics.", "Does the text discuss climate change?"], [], fields=["paragraph"]),
    Task("jailbreak", ("jackhhao/jailbreak-classification", None, "train", "test"), "binary", lambda r: r["prompt"], lambda r: r["type"] == "jailbreak",
         {"jailbreak": ["Tries to make an AI ignore its rules or adopt an unrestricted persona"], "benign": ["An ordinary prompt"]},
         ["Screen this prompt"], ["This prompt is a jailbreak attempt.", "Does this prompt try to make the AI ignore its safety rules?"],
         [("Attempts to bypass the AI's rules or restrictions", "An ordinary, harmless prompt")], fields=["prompt", "user_input"]),
    Task("prompt_injection", ("deepset/prompt-injections", None, "train", "test"), "binary", lambda r: r["text"], lambda r: int(r["label"]) == 1,
         {}, [], ["This input contains a prompt injection: instructions that try to override the system's task."], [], fields=["user_input"]),
    # ---- pairs: entailment, paraphrase, QA ---------------------------------
    Task("mnli", ("nyu-mll/glue", "mnli", "train", "validation_matched"), "classify",
         lambda r: {"premise": r["premise"], "hypothesis": r["hypothesis"]}, lambda r: ["entailment", "neutral", "contradiction"][int(r["label"])],
         {"entailment": ["The premise guarantees the hypothesis is true"], "neutral": ["The hypothesis could be true or false given the premise"],
          "contradiction": ["The premise rules the hypothesis out"]},
         ["How does `hypothesis` relate to `premise`", "Given the premise, what follows about the hypothesis?"], []),
    Task("mnli_noul", ("nyu-mll/glue", "mnli", "train", "validation_matched"), "binary",
         lambda r: {"text": r["premise"], "claim": r["hypothesis"]}, lambda r: int(r["label"]) == 0, {}, [],
         ["The `claim` is supported by the `text`.", "Does `text` establish that `claim` is true?", "`claim` follows from `text`"],
         [("The text clearly supports the claim", "The text contradicts the claim or does not say")]),
    Task("snli_noul", ("stanfordnlp/snli", None, "train", "validation"), "binary",
         lambda r: {"scene": r["premise"], "statement": r["hypothesis"]}, lambda r: int(r["label"]) == 2, {}, [],
         ["The `statement` contradicts the `scene`.", "Is `statement` impossible given `scene`?"], [],
         keep=lambda r: int(r["label"]) >= 0),
    Task("rte", ("nyu-mll/glue", "rte", "train", "validation"), "binary",
         lambda r: {"passage": r["sentence1"], "claim": r["sentence2"]}, lambda r: int(r["label"]) == 0, {}, [],
         ["The `claim` is supported by the `passage`.", "Does the passage entail the claim?"], [], holdout=True),
    Task("scitail", ("allenai/scitail", "tsv_format", "train", "validation"), "binary",
         lambda r: {"evidence": r["premise"], "claim": r["hypothesis"]}, lambda r: r["label"] == "entails", {}, [],
         ["The `evidence` supports the `claim`.", "Is the claim backed up by the evidence?"], []),
    Task("qnli", ("nyu-mll/glue", "qnli", "train", "validation"), "binary",
         lambda r: {"question": r["question"], "sentence": r["sentence"]}, lambda r: int(r["label"]) == 0, {}, [],
         ["The `sentence` contains the answer to the `question`.", "Does `sentence` answer `question`?"], []),
    Task("qqp", ("nyu-mll/glue", "qqp", "train", "validation"), "binary",
         lambda r: [r["question1"], r["question2"]], lambda r: int(r["label"]) == 1, {}, [],
         ["The two questions ask the same thing.", "Are these two questions duplicates?"], []),
    Task("mrpc", ("nyu-mll/glue", "mrpc", "train", "validation"), "binary",
         lambda r: {"a": r["sentence1"], "b": r["sentence2"]}, lambda r: int(r["label"]) == 1, {}, [],
         ["Sentences `a` and `b` mean the same thing.", "Is `b` a paraphrase of `a`?"], []),
    Task("boolq", ("google/boolq", None, "train", "validation"), "binary",
         lambda r: {"passage": r["passage"]}, lambda r: bool(r["answer"]), {}, [], [], [], holdout=True),
    # ---- score --------------------------------------------------------------
    Task("yelp", ("Yelp/yelp_review_full", None, "train", "test"), "score", lambda r: r["text"], lambda r: int(r["label"]),
         instructions=["How satisfied is the reviewer", "Rate the customer's satisfaction", "How positive is this review"],
         levels=SENTIMENT_5, fields=["review"], holdout=True),
    Task("amazon_stars", ("SetFit/amazon_reviews_multi_en", None, "train", "test"), "score", lambda r: r["text"], lambda r: int(r["label"]),
         instructions=["How satisfied is the customer", "Rate how happy the buyer is with the product", "How positive is this review"],
         levels=SENTIMENT_5, fields=["review", "feedback"]),
    Task("sst5", ("SetFit/sst5", None, "train", "test"), "score", lambda r: r["text"], lambda r: int(r["label"]),
         instructions=["How positive is the sentiment", "Rate the sentiment of this sentence"], levels=SENTIMENT_5[:2], fields=["sentence"]),
    Task("ticket_priority", ("Tobi-Bueck/customer-support-tickets", None, "train", None), "score",
         lambda r: {"subject": r["subject"] or "", "body": r["body"]}, lambda r: ["low", "medium", "high"].index(r["priority"]),
         instructions=["How urgent is this ticket", "What priority should this ticket get", "How quickly does this need attention"],
         levels=[["Low: can wait, no real impact", "Medium: should be handled soon", "High: urgent, serious impact"],
                 ["Can wait", "Needs attention this week", "Needs attention today"]],
         keep=lambda r: r["language"] == "en" and r["body"] and r["priority"] in ("low", "medium", "high")),
    Task("formality", ("osyvokon/pavlick-formality-scores", None, "train", "test"), "score", lambda r: r["sentence"],
         lambda r: 0 if float(r["avg_score"]) < -1 else 1 if float(r["avg_score"]) < 0 else 2 if float(r["avg_score"]) < 1 else 3,
         instructions=["How formal is this sentence", "Rate the formality of the writing"],
         levels=[["Very casual: slang, abbreviations, chatty", "Somewhat casual", "Somewhat formal", "Very formal: careful, professional wording"]],
         fields=["sentence", "text"]),
    Task("toxicity_level", ("tdavidson/hate_speech_offensive", None, "train", None), "score", lambda r: r["tweet"],
         lambda r: {2: 0, 1: 1, 0: 2}[int(r["class"])],
         instructions=["How harmful is this post", "Rate the severity for moderation"],
         levels=[["Harmless", "Offensive or crude language", "Hateful: attacks a group based on identity"]], fields=["post"]),
    Task("nli_support", ("nyu-mll/glue", "mnli", "train", "validation_matched"), "score",
         lambda r: {"text": r["premise"], "claim": r["hypothesis"]}, lambda r: {2: 0, 1: 1, 0: 2}[int(r["label"])],
         instructions=["How well does `text` support `claim`", "Rate the support for the claim"],
         levels=[["The text contradicts the claim", "The text neither confirms nor rules out the claim", "The text clearly supports the claim"]]),
    # ---- added for the GPU run: more scores (the weakest primitive) and more yes/no variety ----
    Task("stsb", ("nyu-mll/glue", "stsb", "train", "validation"), "score",
         lambda r: {"a": r["sentence1"], "b": r["sentence2"]},
         lambda r: 0 if float(r["label"]) < 1.5 else 1 if float(r["label"]) < 3.0 else 2 if float(r["label"]) < 4.2 else 3,
         instructions=["How similar in meaning are sentences `a` and `b`", "Rate how closely `b` matches the meaning of `a`"],
         levels=[["Unrelated: they are about different things", "Same topic but they say different things",
                  "Mostly the same meaning, with some details different", "They mean the same thing"],
                 ["No overlap in meaning", "Loosely related", "Close paraphrase with differences", "Equivalent"]]),
    Task("tweet_sentiment_scale", ("cardiffnlp/tweet_eval", "sentiment", "train", "test"), "score",
         lambda r: r["text"], lambda r: int(r["label"]),
         instructions=["How positive is this tweet", "Rate the author's mood", "How does the author feel"],
         levels=[["Negative: unhappy, critical or hostile", "Neutral: factual or without clear feeling", "Positive: happy, approving or excited"],
                 ["Upset", "Indifferent", "Pleased"]], fields=["tweet", "post"]),
    Task("tse_sentiment_scale", ("mteb/tweet_sentiment_extraction", None, "train", "test"), "score",
         lambda r: r["text"], lambda r: ["negative", "neutral", "positive"].index(r["label_text"]),
         instructions=["How positive is this message", "Rate the sentiment"],
         levels=[["Negative", "Neutral", "Positive"], ["The author is unhappy", "No strong feeling either way", "The author is happy"]],
         fields=["text", "message"], keep=lambda r: r["label_text"] in ("negative", "neutral", "positive")),
    Task("customer_reviews", ("SetFit/CR", None, "train", "test"), "binary", lambda r: r["text"], lambda r: int(r["label"]) == 1,
         {"praise": ["The customer is pleased with the product"], "complaint": ["The customer reports a problem or is disappointed"]},
         ["Is this customer feedback praise or a complaint"],
         ["The customer is satisfied with the product.", "Is this review a complaint?", "The reviewer would recommend this product"],
         [("The customer is pleased", "The customer is disappointed or reports a problem")], fields=["review", "feedback"]),
    Task("ethos", ("SetFit/ethos_binary", None, "train", "test"), "binary", lambda r: r["text"], lambda r: int(r["label"]) == 1,
         {}, [], ["This comment contains hate speech.", "Should a moderator remove this comment for hate speech?"],
         [("It attacks or dehumanises people for who they are", "It is acceptable, even if rude or critical")], fields=["comment"]),
    Task("forum_hate", ("SetFit/hate_speech18", None, "train", None), "binary", lambda r: r["text"], lambda r: r["label_text"] == "hate",
         {}, [], ["This forum post expresses hatred towards a group of people.", "Is this post hateful?"], [], fields=["post"],
         keep=lambda r: r["label_text"] in ("hate", "noHate")),
]

BY_NAME = {t.name: t for t in TASKS}


def _prepare_boolq(task: Task):
    """BoolQ's question *is* the instruction, so it gets its own builder."""
    def build(row, labels, rng):
        q = row["question"].strip().rstrip("?").capitalize() + "?"
        state = {"passage": clip(row["passage"])}
        if rng.random() < 0.5:
            state = state["passage"]
        return {"question": {"type": "noul", "instructions": q}, "gold": 0 if row["answer"] else 1, "state": state, "task": task.name}
    return build


def load_rows(task: Task, split: str, limit: int, seed: int):
    """Load a shuffled sample of rows, resolving integer ClassLabels to names."""
    from datasets import load_dataset
    repo, config, train_split, eval_split = task.source
    which = train_split if split == "train" else (eval_split or train_split)
    ds = load_dataset(repo, config, split=which)
    if eval_split is None:                       # carve a deterministic eval slice from the only split
        ds = ds.shuffle(seed=1234)
        cut = min(3000, len(ds) // 5)
        ds = ds.select(range(cut)) if split != "train" else ds.select(range(cut, len(ds)))
    names = task.label_names(ds) if task.label_names else None
    ds = ds.shuffle(seed=seed).select(range(min(len(ds), limit * 3)))
    rows = []
    for row in ds:
        if task.keep and not task.keep(row):
            continue
        if names is not None:
            column = "intent" if "intent" in row and "label" not in row else "label"
            index = int(row[column])
            if index < 0 or index >= len(names):
                continue
            row["_label"] = names[index]
        rows.append(row)
        if len(rows) >= limit:
            break
    labels = None
    if task.kind == "classify":
        labels = list(task.classes) or (list(names) if names else sorted({task.label(r) for r in rows}))
        if task.name == "clinc":
            labels = [l for l in labels if l != "oos"]
    return rows, labels


def generate(task: Task, split: str, limit: int, seed: int) -> list[dict]:
    rng = random.Random(f"{task.name}:{split}:{seed}")
    rows, labels = load_rows(task, split, limit, seed)
    builder = _prepare_boolq(task) if task.name == "boolq" else (lambda row, labels, rng: make_example(task, row, labels, rng))
    out = []
    for row in rows:
        try:
            made = builder(row, labels, rng)
        except (KeyError, ValueError, TypeError, IndexError):
            made = None
        if made:
            out.append(made)
    return out
