"""A hand-authored email-triage task family.

Public datasets cover spam and phishing but not the everyday questions people ask of an inbox:
is this a bill, does it need a reply, how urgent is it, what kind of mail is it. These emails
were written for this project and are labelled by construction. None of them appears in the
sample project that ships with the server (src/local_jev/seed.py), so the sample stays a fair demo.

    python training/email_triage.py --out data_v2/email_triage.jsonl
"""
import argparse
import json
import random

# (category, sender, subject, body, needs_reply, urgency 0-4, extra flags)
# categories: billing, sponsorship, scam, newsletter, notification, security, personal, team, support, opportunity
E = [
 ("billing", "Spotify <no-reply@spotify.com>", "Your Spotify Premium receipt", "Thanks for your payment of $10.99 for Premium Individual. Your next billing date is 3 March. Payment method: Mastercard ending 1182.", 0, 0),
 ("billing", "Hetzner Online <billing@hetzner.com>", "Invoice 8842013 for February", "Dear customer, your invoice for February is attached. Amount due: EUR 41.20. It will be debited from your account in 5 days.", 0, 1),
 ("billing", "Adobe <mail@adobe.com>", "We couldn't process your payment", "Your Creative Cloud subscription payment of $54.99 failed. Update your card by 12 April to keep access to your apps and files.", 0, 3),
 ("billing", "Uber Receipts <noreply@uber.com>", "Your Tuesday evening trip with Uber", "Total: £14.62. Trip from Paddington to Shoreditch, 6.1 miles, 24 min. Paid with Apple Pay.", 0, 0),
 ("billing", "Finance <ap@brightwave.studio>", "Invoice #2207 overdue - second reminder", "Hi, invoice #2207 for $3,400 was due on the 1st and remains unpaid. Please settle it this week or let us know if something is wrong.", 1, 3),
 ("billing", "Namecheap <support@namecheap.com>", "Domain renewal confirmation: alexbuilds.dev", "Your domain alexbuilds.dev has been renewed for 1 year. Amount charged: $16.98. Order #94412207.", 0, 0),
 ("billing", "PayPal <service@paypal.com>", "You sent a payment of $120.00 USD to Tara Wills", "Transaction ID 7HX20391KD. It may take a few minutes for this transaction to appear in your account.", 0, 0),
 ("billing", "Revolut Business <no-reply@revolut.com>", "Your monthly statement is ready", "Your statement for January is ready to download in the app. Closing balance: £18,204.11.", 0, 0),
 ("billing", "Linode <billing@linode.com>", "Linode.com: Payment Receipt [29918441]", "This is your receipt of payment against your credit card in the amount of $24.00. Thank you.", 0, 0),
 ("billing", "Accounts Payable <ap@nordlight.media>", "Remittance advice: payment of $2,750 sent", "We have paid invoice 0114 by bank transfer today. Funds should reach you within 2 working days.", 0, 0),
 ("billing", "Apple <no_reply@email.apple.com>", "Your receipt from Apple", "iCloud+ with 2 TB storage, monthly: £8.99. Billed to Visa 4417. Order ID MN3KQ0X2B.", 0, 0),
 ("billing", "Zoom <billing@zoom.us>", "Your subscription will renew in 7 days", "Your Zoom Pro annual plan renews on 19 June for $149.90. No action is needed unless you want to change your plan.", 0, 1),
 ("sponsorship", "Dana Whitfield <dana@squarespace-partners.example.com>", "Squarespace x your channel", "Hi! We'd love to sponsor two videos in May. Our standard rate for channels your size is $4,000 per 60-second read. Do you have a media kit you can share?", 1, 2),
 ("sponsorship", "Kenji Arai <kenji@warpdev.example.com>", "Paid integration for Warp terminal", "We are launching Warp 2.0 next month and want developer creators to try it on camera. Budget: $9,000 for one dedicated video. Can we talk this week?", 1, 3),
 ("sponsorship", "Influencer Team <collabs@glowvita.co>", "Collab offer: GlowVita gummies", "Hey babe! We love your vibe. We'll send free gummies plus a 15% affiliate code for your followers. Reply with your address!", 1, 1),
 ("sponsorship", "Marta Silva <marta@brilliant.example.com>", "Brilliant sponsorship, Q3 slots", "We have three sponsorship slots left for Q3 at $5,500 each, with a 30-day free trial link for your viewers. Let me know if you would like to hold one.", 1, 2),
 ("sponsorship", "Rohit Menon <rohit@adstellar.example.com>", "Brand deal: client in the VPN space", "I represent a VPN brand looking for tech YouTubers. Flat fee $1,500 plus CPA. If interested, send your rates and last 30 day analytics.", 1, 1),
 ("sponsorship", "Ella Brandt <ella@notion.example.com>", "Re: Notion partnership - revised terms", "Thanks for the call. We can do $12,000 for three integrations, payment net 30, with usage rights for 6 months. Contract attached, please confirm by Friday.", 1, 3),
 ("sponsorship", "partnerships@casinoroyalebets.io", "Earn $$$ promoting our casino", "Promote our online casino to your audience and earn 45% revenue share for life. No limits. Sign up as an affiliate today.", 0, 0),
 ("sponsorship", "Hugo Lindqvist <hugo@framework.example.com>", "Would you review the new Framework 16?", "We'd like to send you a Framework 16 to keep, and sponsor the video at $7,500 with no script approval. Interested?", 1, 2),
 ("sponsorship", "Priyanka D <priyanka@skillforge.example.com>", "Affiliate programme invitation", "Join our affiliate programme and earn 20% on every course sale from your link. There is no upfront payment.", 0, 0),
 ("scam", "Microsoft Account Team <security@micros0ft-verify.com>", "Unusual sign-in activity - verify now", "We detected a sign-in from Russia. Your account will be locked in 24 hours unless you verify your password here: http://micros0ft-verify.com/login", 0, 1, {"scam": 1}),
 ("scam", "DHL Express <tracking@dhl-parcel-redelivery.info>", "Your parcel could not be delivered", "A customs fee of £1.99 is required to release your parcel. Pay within 48 hours using your card details at the link or it will be returned.", 0, 1, {"scam": 1}),
 ("scam", "Mr. Daniel Okafor <d.okafor.esq@example.com>", "CONFIDENTIAL BUSINESS PROPOSAL", "I am a barrister with a client who died leaving $14.5 million unclaimed. I need your help to transfer the funds; you will receive 40%. Reply with your full name and bank details.", 0, 0, {"scam": 1}),
 ("scam", "Netflix <billing@netfIix-account.co>", "Your membership is on hold", "We couldn't process your last payment. Update your card number, expiry date and CVV now to avoid cancellation: netfIix-account.co/update", 0, 1, {"scam": 1}),
 ("scam", "YouTube Partner Support <yt-partner@creator-verify.support>", "Final warning: channel termination", "Your channel violated policy and will be deleted in 12 hours. Download and run the attached appeal form (appeal.exe) to keep your channel.", 0, 1, {"scam": 1}),
 ("scam", "Elena <elena.k92@example.com>", "hi remember me?", "Hi dear I saw your profile and I feel we have connection. I am stuck abroad and need $300 for a ticket, I will pay back double. Please send by gift cards.", 0, 0, {"scam": 1}),
 ("scam", "IT Support <helpdesk@company-mail-upgrade.net>", "Mailbox upgrade required today", "All staff must re-enter their email password at the portal below before 5pm to avoid losing mail. This is mandatory.", 0, 1, {"scam": 1}),
 ("scam", "Binance Rewards <promo@binance-airdrop.live>", "Claim your 0.5 BTC airdrop", "You were selected. Connect your wallet and enter your 12-word seed phrase to receive 0.5 BTC instantly. Offer ends tonight.", 0, 0, {"scam": 1}),
 ("scam", "HR Department <hr@payroll-update-portal.com>", "Action needed: payroll details", "Due to a system migration, confirm your bank account number and national insurance number at the link so your salary is not delayed.", 0, 1, {"scam": 1}),
 ("newsletter", "Morning Brew <crew@morningbrew.com>", "☕ Chips are down", "Good morning. Today: semiconductor stocks slide, a new airline loyalty scheme, and why everyone is suddenly talking about copper.", 0, 0),
 ("newsletter", "TLDR <dan@tldrnewsletter.example.com>", "TLDR 2026-02-11: new open weights model, Rust in the kernel", "Big tech and startups, science and futuristic technology, programming and design. Sponsored by Retool.", 0, 0),
 ("newsletter", "Patagonia <news@patagonia.com>", "New arrivals for spring", "Lightweight layers built for shoulder season. Shop the new collection, with free repairs for life.", 0, 0),
 ("newsletter", "James Clear <james@jamesclear.example.com>", "3-2-1: on patience, small wins and asking better questions", "3 ideas from me, 2 quotes from others, and 1 question for you to ponder this week.", 0, 0),
 ("newsletter", "Product Hunt <hello@digest.producthunt.com>", "Today's top products", "An AI meeting notetaker, a mechanical keyboard configurator, and a habit tracker for teams made the top five today.", 0, 0),
 ("newsletter", "The Verge <newsletters@theverge.com>", "Installer: the best new stuff this week", "A great new e-reader, a documentary worth your weekend, and the to-do app I can't stop recommending.", 0, 0),
 ("newsletter", "IKEA Family <ikea@email.ikea.com>", "20% off storage this weekend", "Members save on wardrobes, shelving and boxes until Sunday. See what's new in store.", 0, 0),
 ("newsletter", "Lenny's Newsletter <lenny@substack.example.com>", "How the best PMs run roadmap reviews", "This week's post covers a framework for quarterly planning, with templates from three companies.", 0, 0),
 ("notification", "GitHub <notifications@github.com>", "[acme/api] CI failed on main (run #4412)", "The workflow 'test' failed for commit 9f3c2ab. 3 of 212 tests failed. View the run for details.", 0, 2),
 ("notification", "Slack <feedback@slack.com>", "You have 14 unread messages in #general", "Catch up on what you missed in the Acme workspace.", 0, 0),
 ("notification", "Google Calendar <calendar-notification@google.com>", "Reminder: Dentist @ Thu 9:30am", "You have an event starting in 1 day. Dentist, Thursday 9:30 - 10:00.", 0, 1),
 ("notification", "Trello <do-not-reply@trello.com>", "Mina moved 'Thumbnail v2' to Done", "Mina Park moved the card Thumbnail v2 from In review to Done on the board Channel production.", 0, 0),
 ("notification", "YouTube <no-reply@youtube.com>", "Your video has finished processing", "\"Building a CLI in Go\" is now available in HD. Check it's set to public when you are ready.", 0, 0),
 ("notification", "Amazon.co.uk <shipment-tracking@amazon.co.uk>", "Your package has been delivered", "Your parcel was handed to a resident at 14:02. Order #204-1188812-0017.", 0, 0),
 ("notification", "Sentry <noreply@sentry.io>", "New issue: TypeError in checkout.js", "TypeError: Cannot read properties of undefined (reading 'id'). First seen 4 minutes ago, 212 events, 187 users affected.", 0, 3),
 ("notification", "Strava <no-reply@strava.com>", "You got 6 kudos on your morning run", "Nice work! Your friends are cheering you on.", 0, 0),
 ("notification", "Netlify <team@netlify.com>", "Deploy succeeded for alexbuilds", "Production deploy of main@5c1d2e9 is live. Build time: 48s.", 0, 0),
 ("notification", "UptimeRobot <alert@uptimerobot.com>", "Monitor is DOWN: api.alexbuilds.dev", "Your monitor api.alexbuilds.dev is currently DOWN (Connection timeout). It has been down for 6 minutes.", 0, 4),
 ("security", "GitHub <noreply@github.com>", "A new SSH key was added to your account", "The following SSH key was added to your account: 'work laptop'. If you believe this was done in error, remove the key and change your password.", 0, 2),
 ("security", "Dropbox <no-reply@dropbox.com>", "New sign-in to your Dropbox", "Someone signed in from Chrome on Windows near Berlin, Germany. If this wasn't you, secure your account.", 0, 2),
 ("security", "Twitter <verify@x.com>", "Your password was changed", "Your password was changed on 11 Feb at 08:14. If you did not make this change, reset your password immediately.", 0, 3),
 ("security", "Coinbase <no-reply@coinbase.com>", "Your verification code is 449 201", "Enter this code to finish signing in. It expires in 10 minutes. Never share it with anyone.", 0, 1),
 ("security", "1Password <watchtower@1password.com>", "2 of your passwords appeared in a data breach", "Watchtower found that logins for two websites were exposed. Change those passwords as soon as you can.", 0, 3),
 ("security", "AWS Notifications <no-reply-aws@amazon.com>", "Root account sign-in detected", "A sign-in using root credentials was detected for account 8812-xxxx-0031 from a new IP address.", 0, 3),
 ("personal", "Dad", "lawnmower", "Do you still have my lawnmower? I need it back before the weekend if you can drop it round. No rush before then.", 1, 2),
 ("personal", "Jess", "Saturday??", "Are we still on for climbing on Saturday? I can do 10 or 2. Let me know which and I'll book.", 1, 2),
 ("personal", "Aunt Carol <carol.h@example.com>", "Photos from the wedding", "Hello darling, I finally got the photos printed. I've attached a few. Your speech was lovely. Give my love to Sam.", 0, 0),
 ("personal", "Tom R", "you won't believe this", "Remember that flat we nearly rented in Hackney? It's on the news, they found a Roman wall under the kitchen. Mad.", 0, 0),
 ("personal", "Sam", "boiler", "The boiler is making that noise again and there's no hot water. Can you call the landlord today? I'm in meetings until 6.", 1, 4),
 ("personal", "Nina <nina.petrova@example.com>", "Happy birthday!", "Happy birthday old friend. Hope this year is a good one. We should catch up properly soon.", 0, 0),
 ("personal", "Mum", "Train times", "What time does your train get in on Friday? Dad will pick you up from the station.", 1, 2),
 ("team", "Marcus (producer) <marcus@studiofold.example.com>", "Shoot schedule for next week - please confirm", "Tuesday: desk setup b-roll. Thursday: interview with Lena at 2pm. Can you confirm both by end of day so I can book the studio?", 1, 3),
 ("team", "Ife <ife@studiofold.example.com>", "Captions done for ep 88", "English captions are uploaded and synced. Spanish will follow tomorrow. Nothing needed from you.", 0, 0),
 ("team", "Marcus (producer) <marcus@studiofold.example.com>", "Sponsor read is 12 seconds over", "The Warp read runs 72 seconds and the contract says 60. Do you want me to trim it or will you re-record? Need to know before tonight's export.", 1, 4),
 ("team", "Bea <bea@studiofold.example.com>", "Q1 analytics summary", "Views up 14%, average view duration flat, newsletter signups up 31%. Full deck attached for Monday's meeting.", 0, 1),
 ("team", "Ife <ife@studiofold.example.com>", "which intro music?", "I have two options for the new intro sting, both attached. Which do you prefer? Not urgent, any time this week.", 1, 1),
 ("team", "Dev team <eng@acme.example.com>", "Code freeze starts Wednesday", "Reminder that the release branch is cut on Wednesday at noon. Merge anything you need before then.", 0, 2),
 ("support", "Carlos M <carlosm88@example.com>", "Course login not working", "I bought your Go course yesterday but the login link says my account doesn't exist. Order number 5521. Can you help?", 1, 3),
 ("support", "Aisha K <aisha.k@example.com>", "Small typo in lesson 4", "In lesson 4 the code sample uses 'lenght' instead of 'length'. Thought you'd want to know. Love the course!", 0, 1),
 ("support", "Pieter <pieter.vd@example.com>", "Refund request", "The course is more basic than I expected. I'd like a refund please, I bought it 3 days ago.", 1, 2),
 ("support", "Yuki <yuki.tan@example.com>", "Which video covers goroutines?", "You mentioned a goroutines deep dive in your latest video but I can't find it on the channel. Could you point me to it?", 1, 1),
 ("support", "Owen <owen.b@example.com>", "Discord invite expired", "The Discord link in your video description has expired. Is there a new one?", 1, 1),
 ("opportunity", "Laura Chen <laura@gophercon.example.com>", "Speak at GopherCon EU?", "We'd love you to give a 30-minute talk in Berlin this June. Travel and hotel are covered. The CFP closes in two weeks, could you let me know?", 1, 2),
 ("opportunity", "Recruiting <talent@stripe.com>", "Developer Advocate, Stripe", "Your content keeps coming up in our team. Would you be open to a chat about a Developer Advocate role? Fully remote, competitive package.", 1, 1),
 ("opportunity", "Sofia Marin <sofia@oreilly.example.com>", "Write a book with O'Reilly?", "I'm an acquisitions editor. Your Go series would make an excellent book. Would you like to discuss a proposal?", 1, 1),
 ("opportunity", "Ravi <ravi@syntaxpod.example.com>", "Guest spot on Syntax Pod", "We record Thursdays. Would you come on to talk about building an audience as a developer? 40 minutes, remote.", 1, 1),
 ("opportunity", "Hannah Boyd <hannah@bbc-co.example.com>", "Interview request for a radio segment", "I'm producing a segment on people who teach coding online, airing next week. Could you spare 15 minutes on Monday or Tuesday?", 1, 3),
 ("opportunity", "Theo <theo@indiehackers.example.com>", "Feature you in our creator series", "We'd like to write a profile about how you grew the channel. It's a written Q&A, takes about an hour.", 1, 1),
]

CATEGORIES = {
    "billing": ("Billing", ["Receipts, invoices, payouts and payment problems", "Anything about money charged, owed or paid"]),
    "sponsorship": ("Sponsorship", ["A brand or agency offering a paid promotion or partnership", "Offers of money or product in exchange for promotion"]),
    "scam": ("Scam", ["Phishing, fraud and other attempts to trick the reader", "Tries to steal credentials, money or personal data"]),
    "newsletter": ("Newsletter", ["A periodic digest or marketing mailing", "Bulk mail from a publication or a shop"]),
    "notification": ("Notification", ["An automated alert from an app or service", "Machine-generated status updates"]),
    "security": ("Security", ["Sign-in alerts, password changes and account warnings from a real service"]),
    "personal": ("Personal", ["Friends and family", "A private message from someone the reader knows"]),
    "team": ("Team", ["Messages from colleagues about ongoing work", "Internal work mail from collaborators"]),
    "support": ("Support", ["A customer or viewer asking for help", "Questions, problems and refund requests from users"]),
    "opportunity": ("Opportunity", ["Invitations, job offers and press requests", "Someone offering the reader a role, talk, interview or feature"]),
}
NOULS = {
    "billing": (lambda c, f: c == "billing", ["Is this email a receipt, invoice or payment notice?", "This email is about a charge, a payment or a bill.", "Does this email document money being charged, paid or owed?"],
                ("It documents a charge, payment, renewal, invoice or failed payment", "Offers, newsletters and personal mail, even if they mention prices")),
    "sponsorship": (lambda c, f: c == "sponsorship", ["Is this a company proposing a paid sponsorship or brand deal?", "This email offers the reader money or product in exchange for promotion.", "Is this a sponsorship or affiliate offer?"],
                    ("A brand, agency or marketer proposes a paid or in-kind promotion", "Receipts, notifications, job offers and personal mail")),
    "scam": (lambda c, f: bool(f.get("scam")), ["Is this email a scam or phishing attempt?", "This email tries to trick the reader into giving up money, credentials or personal data.", "Is this message fraudulent?"],
             ("Fake urgency, lookalike domains, requests for passwords, card numbers or gift cards", "Legitimate mail, including real security alerts and unsolicited marketing")),
    "reply": (lambda c, f: bool(f.get("reply")), ["Does this email need a personal reply from the recipient?", "A real person is waiting for an answer from the reader.", "Should the reader write back?"],
              ("A person asked a direct question or is waiting on a decision", "Automated mail, newsletters, receipts and FYI messages")),
    "automated": (lambda c, f: c in ("billing", "newsletter", "notification", "security") and not f.get("reply"), ["Was this email sent automatically by a system rather than written by a person?", "This is machine-generated mail."], None),
    "personal": (lambda c, f: c == "personal", ["Is this a personal message from a friend or family member?", "This email is private correspondence, not work or commerce."], None),
    "urgent": (lambda c, f: f["urgency"] >= 3, ["Does this email need attention within a day?", "This email is urgent.", "Is this time-sensitive?"],
               ("Something breaks, expires or blocks someone within about a day", "It can wait several days or needs no action")),
}
URGENCY_5 = [["No action needed at all", "Could be handled whenever there is spare time", "Should be handled this week", "Needs attention within a day or two", "Needs attention right now"],
             ["Ignore or archive", "Low priority", "Normal priority", "High priority", "Drop everything"]]
URGENCY_3 = [["Can wait or needs nothing", "Should be dealt with this week", "Needs attention today"], ["Low", "Medium", "High"]]


def state_of(sender, subject, body, rng):
    pick = rng.random()
    if pick < 0.7:
        return {"from": sender, "subject": subject, "body": body}
    if pick < 0.85:
        return {"subject": subject, "body": body}
    return f"From: {sender}\nSubject: {subject}\n\n{body}"


def examples(per_email: int, seed: int):
    rng = random.Random(seed)
    out = []
    for row in E:
        cat, sender, subject, body, reply, urgency = row[:6]
        flags = dict(row[6]) if len(row) > 6 else {}
        flags.update(reply=reply, urgency=urgency)
        for _ in range(per_email):
            state, kind = state_of(sender, subject, body, rng), rng.random()
            if kind < 0.5:
                name = rng.choice(list(NOULS))
                truth, phrasings, criteria = NOULS[name]
                q = {"type": "noul", "instructions": rng.choice(phrasings)}
                if criteria and rng.random() < 0.4:
                    q["criteria"] = {"true": criteria[0], "false": criteria[1]} if rng.random() < 0.6 else {"true": criteria[0]}
                gold = 0 if truth(cat, flags) else 1
            elif kind < 0.8:
                others = [c for c in CATEGORIES if c != cat]
                rng.shuffle(others)
                size = rng.randint(3, len(CATEGORIES))
                drop_gold = size < len(CATEGORIES) and rng.random() < 0.08
                chosen = others[: size - 1] + ([] if drop_gold else [cat])
                with_desc = rng.random() < 0.7
                opts = [(CATEGORIES[c][0], rng.choice(CATEGORIES[c][1]) if with_desc else None) for c in chosen]
                if drop_gold or rng.random() < 0.3:
                    opts.append((rng.choice(["Other", "Something else", "None of these"]), "Anything that does not fit the other options" if with_desc else None))
                gold_key = opts[-1][0] if drop_gold else CATEGORIES[cat][0]
                rng.shuffle(opts)
                q = {"type": "choice", "instructions": rng.choice(["What kind of email is this?", "Classify this email", "Which folder does this email belong in?", "What type of message is this"]),
                     "criteria": dict(opts)}
                gold = [k for k, _ in opts].index(gold_key)
            else:
                if rng.random() < 0.6:
                    levels, gold = rng.choice(URGENCY_5), urgency
                else:
                    levels, gold = rng.choice(URGENCY_3), (0 if urgency <= 1 else 1 if urgency == 2 else 2)
                q = {"type": "score", "instructions": rng.choice(["How urgently does this email need attention?", "How urgent is this email", "Rate the priority of this message"]), "criteria": levels}
            out.append({"state": state, "question": q, "gold": gold, "task": "email_triage"})
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_v2/email_triage.jsonl")
    ap.add_argument("--per-email", type=int, default=16)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    rows = examples(args.per_email, args.seed)
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    kinds = {k: sum(r["question"]["type"] == k for r in rows) for k in ("noul", "choice", "score")}
    print(f"{len(E)} emails -> {len(rows)} examples {kinds} -> {args.out}")
