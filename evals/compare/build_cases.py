"""The 100 hand-written comparison cases. Every case has a correct answer, so both systems
are scored against the truth and not only against each other.

    python evals/compare/build_cases.py     # writes evals/compare/cases.json

None of this text is used for training, and none of it comes from a public dataset.
"""
import json
import pathlib

TEAM = {"type": "choice", "instructions": "Which team should handle this support ticket?",
        "criteria": {"billing": "Charges, invoices, refunds and subscriptions", "technical": "Bugs, errors, outages and integrations",
                     "sales": "Pricing, plans and buying questions from prospects", "account": "Login, passwords, permissions and profile settings"}}
REFUND = {"type": "noul", "instructions": "The customer is asking for their money back.",
          "criteria": {"true": "Explicitly requests a refund, a reversal of a charge or a credit", "false": "Mentions a charge without asking for money back, or asks something else"}}
FRUSTRATION = {"type": "score", "instructions": "How frustrated is the customer?",
               "criteria": ["Calm and matter-of-fact", "Annoyed but polite", "Angry, using strong language or threats to leave"]}
REPLY = {"type": "noul", "instructions": "Does this email need a personal reply from the recipient?",
         "criteria": {"true": "A real person asks a direct question or is waiting on a decision", "false": "Automated mail, newsletters, receipts and FYI messages"}}
EMAIL_TYPE = {"type": "choice", "instructions": "What kind of email is this?",
              "criteria": {"receipt": "Documents a payment, invoice or charge", "newsletter": "A periodic digest or marketing mailing",
                           "security": "Sign-in alerts, password changes and account warnings", "personal": "From a friend or family member",
                           "work": "From a colleague about ongoing work"}}
URGENCY = {"type": "score", "instructions": "How urgently does this email need attention?",
           "criteria": ["Can wait or needs nothing", "Should be dealt with this week", "Needs attention today"]}
SATISFACTION = {"type": "score", "instructions": "How satisfied is the reviewer with the product?",
                "criteria": ["Very dissatisfied: regrets buying it", "Dissatisfied: more bad than good", "Mixed or neutral", "Satisfied: minor complaints at most", "Delighted: recommends it without reservation"]}
DAMAGED = {"type": "noul", "instructions": "The reviewer says the product arrived damaged, broken or defective."}
REVIEW_TOPIC = {"type": "choice", "instructions": "What is this review mainly about?",
                "criteria": {"price": "Whether it is worth the money", "quality": "How well the product is made or performs",
                             "shipping": "Delivery speed, packaging and courier", "customer_service": "Dealing with the seller's support staff"}}
SPAM = {"type": "noul", "instructions": "Is this comment spam or self-promotion?",
        "criteria": {"true": "Advertises a link, a channel, a giveaway or a money scheme", "false": "A genuine reaction to the content, positive or negative"}}
INSULT = {"type": "noul", "instructions": "This comment contains a personal insult aimed at someone."}
COMMENT_KIND = {"type": "choice", "instructions": "What kind of comment is this?",
                "criteria": {"praise": "Thanks or compliments", "question": "Asks the creator something", "complaint": "Reports a problem or criticises", "spam": "Advertising or scams"}}
NEWS = {"type": "choice", "instructions": "Which section of the newspaper does this headline belong in?",
        "criteria": {"world": "International affairs, politics and conflict", "sports": "Sport and athletes", "business": "Companies, markets and the economy",
                     "science_tech": "Science, technology and medicine", "entertainment": "Film, music, TV and celebrities"}}
CONTACT = {"type": "noul", "instructions": "The text contains a phone number or an email address."}
CARD = {"type": "noul", "instructions": "The text contains what looks like a payment card number."}
INTENT = {"type": "choice", "instructions": "What does the user want the assistant to do?",
          "criteria": {"set_alarm": "Set an alarm, timer or reminder", "play_music": "Play a song, artist, playlist or radio", "weather": "Tell them the weather or forecast",
                       "send_message": "Send a text or message to someone", "navigation": "Give directions or travel time to a place"}}
SUPPORTED = {"type": "noul", "instructions": "The `claim` is supported by the `text`.",
             "criteria": {"true": "The text clearly states or directly implies the claim", "false": "The text contradicts the claim or does not say"}}

C = []
def add(domain, question, rows):
    for state, gold in rows:
        C.append({"domain": domain, "state": state, "question": question, "gold": gold})

add("Support tickets", TEAM, [
 ("I was charged twice for my March subscription. Can you look into it?", "billing"),
 ("The export button throws a 500 error every time I click it since yesterday's update.", "technical"),
 ("We're a team of 40 evaluating your product. Do you offer volume discounts on the annual plan?", "sales"),
 ("I can't log in. The password reset email never arrives, I've checked spam too.", "account"),
 ("Your Zapier integration stopped syncing new contacts two days ago.", "technical"),
 ("Please send me a VAT invoice for order 88213, our accountant needs it.", "billing"),
 ("How do I add a colleague as an admin on our workspace?", "account"),
 ("What's the difference between the Pro and Business plans? Thinking of upgrading from the free trial.", "sales")])
add("Support tickets", REFUND, [
 ("I cancelled within the trial period but was still charged $49. Please refund it.", True),
 ("This is not what I expected at all. I want my money back.", True),
 ("Could you reverse last month's charge? I never used the account.", True),
 ("I see a $12 charge on my statement. What is it for?", False),
 ("Can I switch from monthly to annual billing?", False),
 ("The app keeps crashing when I upload photos.", False)])
add("Support tickets", FRUSTRATION, [
 ("Hi, my invoice shows the wrong company name. Could you update it to Northwind Ltd? Thanks.", 0),
 ("Just checking whether the CSV import supports semicolon delimiters.", 0),
 ("This is the second time I'm writing about this. It's getting a bit tiresome, but I'd appreciate an update.", 1),
 ("I've been waiting four days for a reply, which is not great. Please can someone get back to me.", 1),
 ("This is absolutely ridiculous. Your product has cost us a whole day of work. Fix it NOW or we're cancelling.", 2),
 ("Are you people even reading these tickets?! Third outage this week. Utterly useless. I'm done with you.", 2)])
add("Email", REPLY, [
 ({"from": "Lucy Hart <lucy@hartdesign.co>", "subject": "Logo concepts", "body": "Hi, I've attached three logo directions. Which one should I develop further? I'd like to start on Monday."}, True),
 ({"from": "Omar <omar@fieldnotes.io>", "subject": "Quick question on the contract", "body": "Are you happy with the 30-day payment terms, or do you want me to push for 14?"}, True),
 ({"from": "Trainline <no-reply@trainline.com>", "subject": "Your e-ticket to Manchester", "body": "Your booking is confirmed. Show this barcode at the gate. Coach C, seat 42."}, False),
 ({"from": "Medium Daily Digest <noreply@medium.com>", "subject": "Stories for you", "body": "Top stories in Programming and Design, picked for you today."}, False),
 ({"from": "HR <hr@acme.dev>", "subject": "Office closed on Monday", "body": "A reminder that the office is closed this Monday for the bank holiday. No action needed."}, False)])
add("Email", EMAIL_TYPE, [
 ({"from": "Octopus Energy <hello@octopus.energy>", "subject": "Your bill for April", "body": "Your electricity bill for April is £64.12 and will be taken by Direct Debit on the 9th."}, "receipt"),
 ({"from": "Deliveroo <no-reply@deliveroo.co.uk>", "subject": "Your order receipt", "body": "Thanks for ordering from Dishoom. Total paid: £31.40."}, "receipt"),
 ({"from": "The Browser <editor@thebrowser.com>", "subject": "Five links worth your time", "body": "This week's selection of the best writing on the web."}, "newsletter"),
 ({"from": "Instagram <security@mail.instagram.com>", "subject": "New login from Chrome on Windows", "body": "We noticed a login from a device you don't usually use. If this wasn't you, secure your account."}, "security"),
 ({"from": "Grandad", "subject": "Allotment", "body": "The tomatoes are coming on lovely. Come round Sunday and I'll give you a bag. Bring the little ones."}, "personal"),
 ({"from": "Femi <femi@acme.dev>", "subject": "PR #412 ready for review", "body": "I've addressed your comments on the caching layer. Could you take another look before standup?"}, "work")])
add("Email", URGENCY, [
 ({"subject": "Your weekly screen time report", "body": "Your screen time was down 8% last week."}, 0),
 ({"subject": "New features in your notes app", "body": "We've added tables and a faster search. Update any time."}, 0),
 ({"subject": "Expense report due Friday", "body": "Please submit March expenses by end of day Friday so they make this pay run."}, 1),
 ({"subject": "Production database is down", "body": "The primary is unreachable and customers are seeing errors. We need you on the incident call right now."}, 2),
 ({"subject": "Your flight departs in 3 hours", "body": "Online check-in closes in 60 minutes. Check in now to avoid airport fees."}, 2)])
add("Product reviews", SATISFACTION, [
 ("Complete waste of money. It stopped working after two days and the seller ignored me. Avoid.", 0),
 ("Not great. The battery barely lasts an hour and it feels cheap, though the screen is okay.", 1),
 ("It's fine. Does what it says, nothing special. Some things I like, some I don't.", 2),
 ("Really happy with it overall. Setup took a while but it works well now.", 3),
 ("Absolutely love it. Best purchase I've made this year, I've already recommended it to three friends.", 4),
 ("Exceeded every expectation. Flawless build, gorgeous sound, and it arrived early. Five stars without hesitation.", 4)])
add("Product reviews", DAMAGED, [
 ("The box was crushed and the glass lid was shattered when I opened it.", True),
 ("One of the legs was snapped off on arrival. Very disappointing.", True),
 ("Works perfectly and arrived well packaged. Very pleased.", False),
 ("A bit pricey for what it is, but the quality is good.", False)])
add("Product reviews", REVIEW_TOPIC, [
 ("Took three weeks to arrive and the courier left it in the rain.", "shipping"),
 ("At this price it's a steal. You won't find better value anywhere.", "price"),
 ("I emailed support twice about a missing part and they were rude and unhelpful both times.", "customer_service"),
 ("The stitching is coming apart after a month and the zip feels flimsy.", "quality")])
add("Comments", SPAM, [
 ("Earn $500 a day from home! Message me on Telegram to find out how.", True),
 ("Subscribe to my channel for daily giveaways, link in my profile!!", True),
 ("Get 10,000 followers instantly at followboost dot net, cheap and safe.", True),
 ("I disagree with your point about testing, but it was a well made video.", False),
 ("Could you share the slides from this talk?", False)])
add("Comments", INSULT, [
 ("You're an idiot and everyone here knows it.", True),
 ("Only a complete moron would write code like this.", True),
 ("I think this approach is wrong because it ignores caching.", False),
 ("Thanks for explaining this so patiently.", False)])
add("Comments", COMMENT_KIND, [
 ("This was exactly what I needed, thank you so much!", "praise"),
 ("Which library did you use for the charts?", "question"),
 ("The sound cuts out completely at the five minute mark.", "complaint"),
 ("Make money fast with my forex signals group, join free today.", "spam"),
 ("Brilliant explanation, clearest one I've found.", "praise")])
add("News headlines", NEWS, [
 ("Central bank holds interest rates as inflation cools to 2.1%", "business"),
 ("Striker's hat-trick sends underdogs into cup final", "sports"),
 ("Ceasefire talks resume as foreign ministers meet in Geneva", "world"),
 ("Researchers unveil battery that charges in under five minutes", "science_tech"),
 ("Director's long-awaited sequel tops weekend box office", "entertainment"),
 ("Retail giant to close 40 stores after profits slump", "business"),
 ("Marathon world record falls in Berlin", "sports"),
 ("New telescope captures sharpest image yet of distant galaxy", "science_tech"),
 ("Pop star announces surprise album and world tour", "entertainment"),
 ("Thousands evacuated as floods hit coastal region abroad; aid agencies appeal to governments", "world")])
add("PII screening", CONTACT, [
 ("You can reach me on 07700 900123 after six.", True),
 ("Send the files to maria.lopez@example.org when they're ready.", True),
 ("I'll be in the office on Tuesday, let's talk then.", False),
 ("The meeting is in room 4 on the second floor.", False)])
add("PII screening", CARD, [
 ("Please charge it to 4539 1488 0343 6467, expiry 09/27.", True),
 ("My card is 5500-0000-0000-0004, the name on it is J Smith.", True),
 ("Order number 4539 has shipped.", False),
 ("I paid by card yesterday and got a receipt.", False)])
add("Assistant intents", INTENT, [
 ("wake me up at half six tomorrow", "set_alarm"),
 ("put on some Miles Davis", "play_music"),
 ("will I need an umbrella this afternoon", "weather"),
 ("text Priya that I'm running ten minutes late", "send_message"),
 ("how long will it take to drive to the airport", "navigation"),
 ("remind me to take the bins out at eight", "set_alarm"),
 ("play my workout playlist", "play_music"),
 ("what's the forecast for Saturday in Leeds", "weather"),
 ("tell mum I'll call her tonight", "send_message"),
 ("take me to the nearest petrol station", "navigation")])
add("Claim checking", SUPPORTED, [
 ({"text": "The museum is open Tuesday to Sunday, 10am to 6pm. It is closed on Mondays.", "claim": "The museum is closed on Mondays."}, True),
 ({"text": "The museum is open Tuesday to Sunday, 10am to 6pm. It is closed on Mondays.", "claim": "The museum is open every day of the week."}, False),
 ({"text": "Revenue rose 12% to $4.1 billion, driven by strong demand in Asia.", "claim": "Revenue increased."}, True),
 ({"text": "Revenue rose 12% to $4.1 billion, driven by strong demand in Asia.", "claim": "Revenue fell compared with last year."}, False),
 ({"text": "The trial enrolled 300 adults and found no significant difference between the drug and placebo.", "claim": "The drug worked better than placebo."}, False),
 ({"text": "All flights from the airport were cancelled on Friday because of heavy snow.", "claim": "Bad weather disrupted flights on Friday."}, True),
 ({"text": "The recipe uses butter, flour, sugar and two eggs.", "claim": "The recipe is vegan."}, False),
 ({"text": "Anna moved to Lisbon in 2019 and has worked there as an architect ever since.", "claim": "Anna works as an architect."}, True)])

assert len(C) == 100, len(C)
for i, case in enumerate(C, 1):
    case["id"] = f"c{i:03d}"
    q = case["question"]
    if q["type"] == "choice":
        assert case["gold"] in q["criteria"], case
    elif q["type"] == "score":
        assert 0 <= case["gold"] < len(q["criteria"]), case
    else:
        assert isinstance(case["gold"], bool), case
out = pathlib.Path(__file__).with_name("cases.json")
out.write_text(json.dumps(C, indent=1, ensure_ascii=False))
kinds = {k: sum(c["question"]["type"] == k for c in C) for k in ("choice", "noul", "score")}
print(f"{len(C)} cases {kinds}, {len({c['domain'] for c in C})} domains -> {out}")
