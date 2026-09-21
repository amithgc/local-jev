"""A hand-authored task family for comments and community posts (video comments, forum replies, chat).

Written for this project and labelled by construction, like training/email_triage.py.
None of these comments appears in any sample project.

    python training/comment_triage.py --out data/comment_triage.jsonl
"""
import argparse
import json
import random

# (kind, text, sentiment 0-4, needs_reply, reports_problem)
# kinds: praise, question, complaint, spam, suggestion, off_topic, hostile
C = [
 ("praise", "Best explanation of closures I've ever seen. Subscribed.", 4, 0, 0),
 ("praise", "I've watched this three times and finally passed my interview. Thank you so much!", 4, 0, 0),
 ("praise", "The pacing in this one was perfect, not too fast, not too slow.", 4, 0, 0),
 ("praise", "Your channel is criminally underrated.", 4, 0, 0),
 ("praise", "Clear, concise, no fluff. More of this please.", 4, 0, 0),
 ("praise", "Came for the tutorial, stayed for the dry humour.", 3, 0, 0),
 ("praise", "This saved my weekend project. Legend.", 4, 0, 0),
 ("praise", "Really solid video, thanks for putting in the effort.", 3, 0, 0),
 ("praise", "The diagrams make such a difference. Great work as always.", 4, 0, 0),
 ("praise", "Shared this with my whole team.", 3, 0, 0),
 ("question", "Does this approach still work with version 5? The config format changed.", 2, 1, 0),
 ("question", "What font are you using in the terminal?", 2, 1, 0),
 ("question", "Could you explain why you used a mutex there instead of a channel?", 2, 1, 0),
 ("question", "Is the source code available anywhere?", 2, 1, 0),
 ("question", "How would this scale to a million rows? Would you shard it?", 2, 1, 0),
 ("question", "Which mic is that? Sounds great.", 3, 1, 0),
 ("question", "Will there be a part two covering deployment?", 2, 1, 0),
 ("question", "Is this suitable for a complete beginner or should I learn Python first?", 2, 1, 0),
 ("question", "At 8:15 you skip the migration step, what command did you run?", 2, 1, 0),
 ("question", "Do you offer any mentoring or consulting?", 2, 1, 0),
 ("complaint", "The background music is way too loud, I could barely hear you.", 1, 0, 1),
 ("complaint", "Video cuts off at 14 minutes, is the upload broken?", 1, 1, 1),
 ("complaint", "The repo link in the description gives a 404.", 1, 1, 1),
 ("complaint", "Text is unreadable at 1080p, please zoom in next time.", 1, 0, 1),
 ("complaint", "Followed every step and it crashes with 'module not found' on line 12.", 1, 1, 1),
 ("complaint", "Too many ads. Three mid-rolls in a ten minute video is ridiculous.", 1, 0, 0),
 ("complaint", "You promised timestamps last time and there still aren't any.", 1, 0, 0),
 ("complaint", "The captions are out of sync by about two seconds.", 1, 0, 1),
 ("complaint", "This is just the docs read aloud. Expected more depth.", 1, 0, 0),
 ("complaint", "Your course checkout page rejects my card every time.", 1, 1, 1),
 ("spam", "I made $4,800 last week working from home, message me on Telegram @fastcash_kira", 2, 0, 0),
 ("spam", "Nice video! Check out my channel, I post daily crypto tips, sub4sub?", 2, 0, 0),
 ("spam", "🔥🔥 FREE ROBUX generator working 2026 👉 rbx-free-now dot com 🔥🔥", 2, 0, 0),
 ("spam", "Thanks to Mrs Angela Moore my investment grew to $38,000 in a month, contact her on WhatsApp +1 415 ...", 2, 0, 0),
 ("spam", "Want 10k followers overnight? Visit boostmyreach today, cheapest prices guaranteed.", 2, 0, 0),
 ("spam", "I am giving away 2 BTC to celebrate, just send 0.1 to verify your wallet.", 2, 0, 0),
 ("spam", "Hot singles in your area are waiting, link in my profile", 2, 0, 0),
 ("spam", "Buy cheap watch replicas, luxury quality, worldwide shipping, link below.", 2, 0, 0),
 ("suggestion", "Would love a video comparing this with the built-in solution.", 3, 0, 0),
 ("suggestion", "Idea: a series where you review subscribers' code.", 3, 0, 0),
 ("suggestion", "Please add chapters, it makes it much easier to come back to.", 2, 0, 0),
 ("suggestion", "You should cover testing next, nobody explains it well.", 3, 0, 0),
 ("suggestion", "A dark theme for the slides would be easier on the eyes.", 2, 0, 0),
 ("suggestion", "Could you do a shorter version of this for people who just need the summary?", 2, 0, 0),
 ("suggestion", "It would help if you linked the previous episode in the description.", 2, 0, 0),
 ("off_topic", "Anyone else watching this at 3am instead of sleeping?", 2, 0, 0),
 ("off_topic", "Early squad, where you at?", 2, 0, 0),
 ("off_topic", "Your cat walking past at 6:40 is the highlight.", 3, 0, 0),
 ("off_topic", "Who's here after the keynote?", 2, 0, 0),
 ("off_topic", "My dog barked exactly when you said 'fetch'. Coincidence?", 3, 0, 0),
 ("off_topic", "Greetings from Brazil!", 3, 0, 0),
 ("off_topic", "That plant behind you needs water.", 2, 0, 0),
 ("hostile", "You clearly have no idea what you're talking about. Delete this.", 0, 0, 0),
 ("hostile", "Imagine being this bad at your job and still making videos.", 0, 0, 0),
 ("hostile", "Stop talking and get to the point you clown.", 0, 0, 0),
 ("hostile", "Worst channel on the platform. Reported.", 0, 0, 0),
 ("hostile", "Nobody asked for your opinion, idiot.", 0, 0, 0),
 ("hostile", "This is garbage and so are you.", 0, 0, 0),
]

KINDS = {
    "praise": ("Praise", ["Thanks or compliments", "The commenter liked it"]),
    "question": ("Question", ["Asks the creator something", "The commenter wants an answer"]),
    "complaint": ("Complaint", ["Reports a problem or criticises the content", "Something is wrong or disappointing"]),
    "spam": ("Spam", ["Advertising, scams and self-promotion", "Promotes a link, a channel or a scheme"]),
    "suggestion": ("Suggestion", ["Proposes an idea or a future topic", "Asks for something to be made or changed"]),
    "off_topic": ("Off topic", ["Chatter unrelated to the content", "Jokes and greetings that need no action"]),
    "hostile": ("Hostile", ["Insults or personal attacks", "Abuse aimed at the creator"]),
}
NOULS = {
    "spam": (lambda k, f: k == "spam", ["Is this comment spam or self-promotion?", "This comment advertises something.", "Should this comment be removed as spam?"],
             ("It promotes a link, a channel, a giveaway or a money scheme", "A genuine reaction to the content, positive or negative")),
    "question": (lambda k, f: k == "question", ["Is the commenter asking the creator a question?", "This comment asks for information."], None),
    "reply": (lambda k, f: bool(f["reply"]), ["Does this comment deserve a reply from the creator?", "The commenter is waiting for an answer or a fix."],
              ("A sincere question, or a problem the creator can fix", "Praise, jokes, spam and abuse")),
    "problem": (lambda k, f: bool(f["problem"]), ["Does this comment report something broken or wrong with the video or product?", "The commenter reports a technical problem."],
                ("Broken links, errors, bad audio or video, failed payments", "Opinions about the content itself")),
    "hostile": (lambda k, f: k == "hostile", ["Is this comment abusive?", "This comment insults the creator.", "Should a moderator hide this comment for abuse?"], None),
    "positive": (lambda k, f: f["sentiment"] >= 3, ["Is the commenter happy with the content?", "This is a positive comment."], None),
    "idea": (lambda k, f: k == "suggestion", ["Does this comment suggest a future topic or an improvement?", "The commenter proposes an idea."], None),
}
SENTIMENT_5 = [["Hostile or abusive", "Unhappy or critical", "Neutral", "Pleased", "Delighted"],
               ["Attacks the creator", "Disappointed or reporting a problem", "No strong feeling", "Likes it", "Loves it"]]
SENTIMENT_3 = [["Negative", "Neutral", "Positive"], ["Unhappy with the content", "Neither happy nor unhappy", "Happy with the content"]]
FIELDS = ["comment", "text", "post", "message"]


def examples(per_item: int, seed: int):
    rng = random.Random(seed)
    out = []
    for kind, text, sentiment, reply, problem in C:
        flags = {"sentiment": sentiment, "reply": reply, "problem": problem}
        for _ in range(per_item):
            state = text if rng.random() < 0.65 else {rng.choice(FIELDS): text}
            pick = rng.random()
            if pick < 0.5:
                truth, phrasings, criteria = NOULS[rng.choice(list(NOULS))]
                q = {"type": "noul", "instructions": rng.choice(phrasings)}
                if criteria and rng.random() < 0.4:
                    q["criteria"] = {"true": criteria[0], "false": criteria[1]} if rng.random() < 0.6 else {"true": criteria[0]}
                gold = 0 if truth(kind, flags) else 1
            elif pick < 0.8:
                others = [k for k in KINDS if k != kind]
                rng.shuffle(others)
                size = rng.randint(3, len(KINDS))
                drop_gold = size < len(KINDS) and rng.random() < 0.08
                chosen = others[: size - 1] + ([] if drop_gold else [kind])
                with_desc = rng.random() < 0.7
                opts = [(KINDS[k][0], rng.choice(KINDS[k][1]) if with_desc else None) for k in chosen]
                if drop_gold or rng.random() < 0.3:
                    opts.append((rng.choice(["Other", "Something else", "None of these"]), "Anything that does not fit the other options" if with_desc else None))
                gold_key = opts[-1][0] if drop_gold else KINDS[kind][0]
                rng.shuffle(opts)
                q = {"type": "choice", "instructions": rng.choice(["What kind of comment is this?", "Classify this comment", "Sort this comment into a bucket"]),
                     "criteria": dict(opts)}
                gold = [k for k, _ in opts].index(gold_key)
            else:
                if rng.random() < 0.55:
                    levels, gold = rng.choice(SENTIMENT_5), sentiment
                else:
                    levels, gold = rng.choice(SENTIMENT_3), (0 if sentiment <= 1 else 1 if sentiment == 2 else 2)
                q = {"type": "score", "instructions": rng.choice(["How does the commenter feel about the content?", "Rate the sentiment of this comment", "How positive is this comment"]),
                     "criteria": levels}
            out.append({"state": state, "question": q, "gold": gold, "task": "comment_triage"})
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/comment_triage.jsonl")
    ap.add_argument("--per-item", type=int, default=16)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    rows = examples(args.per_item, args.seed)
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(C)} comments -> {len(rows)} examples -> {args.out}")
