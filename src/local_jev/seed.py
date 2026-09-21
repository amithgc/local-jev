"""The sample project created on first start: a creator's inbox and seven questions about it."""
from __future__ import annotations

from .importer import title_of

QUESTIONS = [
    {"name": "Invoice or receipt", "type": "noul",
     "instructions": "Is this email a receipt, invoice, payment confirmation, or billing notice?",
     "criteria": {"true": "It documents a charge, payment, payout, renewal, or failed payment.", "false": ""}, "threshold": 0.5},
    {"name": "Brand deal", "type": "noul",
     "instructions": "Is this email a company or agency proposing a paid sponsorship, partnership or brand deal?",
     "criteria": {"true": "A brand, agency or marketer offers money or product in exchange for promotion.",
                  "false": "Newsletters, receipts, notifications, scams and personal mail."}, "threshold": 0.5},
    {"name": "Scam or phishing", "type": "noul",
     "instructions": "Is this email a scam or phishing attempt?",
     "criteria": {"true": "It tries to trick the reader into giving up credentials, money or personal data, often with fake urgency.",
                  "false": "A legitimate email, even if it is unsolicited marketing."}, "threshold": 0.5},
    {"name": "Email type", "type": "choice", "instructions": "What kind of email is this?",
     "criteria": {"Notification": "An automated alert from an app or service", "Newsletter": "A periodic digest or marketing mailing",
                  "Billing": "Receipts, invoices, payouts and payment problems", "Opportunity": "Sponsorships, collaborations, invitations and job offers",
                  "Team": "Messages from colleagues, editors or collaborators about ongoing work", "Personal": "Friends and family",
                  "Support": "A viewer or customer asking for help", "Security": "Sign-in alerts, password resets and account warnings",
                  "Other": "Anything that does not fit the other options"}},
    {"name": "Needs a reply from me", "type": "noul",
     "instructions": "Does this email need a personal reply from the recipient?",
     "criteria": {"true": "A real person asked a direct question or is waiting on a decision.",
                  "false": "Automated mail, newsletters, receipts and FYI messages."}, "threshold": 0.5},
    {"name": "Urgency", "type": "score", "instructions": "How urgently does this email need attention?",
     "criteria": ["No action needed at all", "Could be handled whenever there is spare time", "Should be handled this week",
                  "Needs attention within a day or two", "Needs attention right now"]},
    {"name": "Sponsor fit", "type": "score",
     "instructions": "How good a sponsorship fit is this for a channel about software, productivity and developer tools?",
     "criteria": ["Not a sponsorship offer at all", "Poor fit. A sponsor whose product is unrelated or disreputable",
                  "Weak fit. A sponsor only loosely related to the audience", "Good fit. A sponsor the audience would plausibly use",
                  "Strong fit. A sponsor squarely aimed at developers and productivity"]},
]

E = lambda sender, subject, body: {"from": sender, "subject": subject, "body": body}  # noqa: E731

EMAILS = [
    E("Stripe <receipts@stripe.com>", "Your receipt from Descript [#2291-4410]", "Receipt from Descript. Amount paid: $24.00. Date paid: Sep 14, 2026. Payment method: Visa ending 4242. Descript Creator plan, Sep 14 - Oct 14. If you have questions, contact support@descript.com."),
    E("Maya Chen <maya@northpeak.example.com>", "Paid partnership: Linear x your channel (Q4)", "Hi Alex, I run creator partnerships at NorthPeak. Our client Linear is booking Q4 integrations with developer-focused channels and yours is top of our list. We have budget for a 60-90 second integration, $6,500 per video, two videos. Could you share your media kit and availability for a call this week? Best, Maya"),
    E("YouTube <no-reply@youtube.com>", "New comment on \"I rebuilt my whole workflow in Raycast\"", "devin_codes commented: \"The window management bit at 7:42 saved me so much time, thank you!\" Reply or manage comments in YouTube Studio."),
    E("PayPaI Security <service@paypa1-secure-review.com>", "Your account has been limited - action required", "Dear customer, we noticed unusual activity and your account has been temporarily limited. You must confirm your identity within 24 hours or your funds will be permanently frozen. Click here to verify: http://paypa1-secure-review.com/verify?id=88213. Failure to comply will result in account closure."),
    E("Priya (editor) <priya@cutroom.example.com>", "Ep 142 rough cut is up - need your notes by Thursday", "Hey Alex, rough cut of 142 is in the shared folder. Two questions: do you want to keep the cold open as is, and are we cutting the Notion segment? I need your notes by Thursday noon to hit the Friday upload. Thanks! P"),
    E("The Pragmatic Engineer <newsletter@pragmaticengineer.com>", "The Pulse #112: What the latest layoffs tell us", "This week: a look at hiring data across Big Tech, how staff engineers are spending their time in 2026, and an inside look at a payments migration. Read the full issue online. You are receiving this because you subscribed."),
    E("GitHub <noreply@github.com>", "[alexdev/dotfiles] Dependabot: bump actions/checkout from 4 to 5", "Bumps actions/checkout from 4 to 5. Release notes and changelog are available on the PR. Dependabot will resolve any conflicts with this PR as long as you don't alter it yourself."),
    E("Mum", "Sunday lunch?", "Hi love, are you and Sam coming on Sunday? Dad is doing the lamb. Let me know by Friday so I know how much to buy. Also your cousin asked if you could look at her laptop. Love, Mum x"),
    E("Google <no-reply@accounts.google.com>", "Security alert: new sign-in on Mac", "We noticed a new sign-in to your Google Account on a Mac device. If this was you, you don't need to do anything. If not, we'll help you secure your account. Check activity."),
    E("Tomás Rivera <tomas@nordvpn-partners.com>", "Sponsorship opportunity - NordVPN", "Hello! We love your content and think NordVPN would be a great fit for your audience. We offer $1,200 flat per integration plus 30% affiliate commission. We can send a brief and talking points today. Are you open to it?"),
    E("AWS Billing <aws-billing@amazon.com>", "Amazon Web Services Billing Statement Available", "Your AWS billing statement for the period Aug 1 - Aug 31, 2026 is now available. Total: $87.13. Your payment method on file will be charged. View your bill in the Billing console."),
    E("Jordan Blake <jordan@devrelconf.example.com>", "Invitation to speak at DevRelCon London", "Hi Alex, I'm on the programme committee for DevRelCon London (Nov 18-19). We'd love you to give a 25 minute talk on building an audience as an engineer. We cover travel and two nights' hotel. Could you let me know by the end of next week whether you're interested?"),
    E("Notion <team@mail.notion.so>", "What's new: database automations and a faster sidebar", "Automations can now trigger on page edits. Plus a redesigned sidebar, offline improvements and 14 bug fixes. See everything that shipped in September."),
    E("support@apple-id-verify.help", "Your Apple ID will be disabled today", "Your Apple ID was used to sign in from Lagos, Nigeria. If you do not confirm your password and card details in the next 2 hours your Apple ID will be DISABLED. Confirm now at apple-id-verify.help/login"),
    E("Wise <noreply@wise.com>", "You received 4,250.00 USD from Google AdSense", "Good news - 4,250.00 USD from GOOGLE ADSENSE has arrived in your USD balance. Reference: AdSense payout Aug 2026. View transaction."),
    E("Sam", "did you book the vet", "Hey, did you manage to book Biscuit's vet appointment? They close at 5 today and she's still limping. Call me if you can't and I'll do it on my lunch."),
    E("Lena Fischer <lena@raycast.example.com>", "Re: Raycast sponsorship renewal - contract attached", "Hi Alex, thanks for the great results last quarter. Attached is the renewal: 4 integrations at $7,000 each, net 30. Legal needs a signature by Monday to lock the October slot. Any changes you want to the talking points?"),
    E("Substack <no-reply@substack.com>", "You have 38 new subscribers", "Your publication gained 38 new free subscribers and 2 paid subscribers this week. Your open rate was 54%. See your stats."),
    E("crypto-gains@fastprofit-mail.biz", "Alex, turn $250 into $19,000 in 7 days (creators only)", "Exclusive for YouTubers: our AI trading bot guarantees 80x returns. Promote us and keep 50% of every deposit your viewers make. Spots closing TONIGHT. Reply with your WhatsApp number."),
    E("Figma <billing@figma.com>", "Payment failed for your Figma Professional plan", "We were unable to charge your card ending 4242 for $15.00. Please update your payment method within 7 days to avoid losing editor access."),
    E("Dev Patel <dev.patel@example.com>", "Question about your Neovim config video", "Hi Alex, long time viewer. I followed your Neovim setup but the LSP for TypeScript never attaches on my machine - I get 'client quit with exit code 1'. Is there somewhere I can see your full config? Thanks for everything you make."),
    E("Calendly <notifications@calendly.com>", "New event: Intro call with Maya Chen", "A new event has been scheduled. Invitee: Maya Chen. Event: 30 minute intro call. Date: Thursday, Sep 24 at 3:00pm. Location: Google Meet."),
    E("Hacker Newsletter <kale@hackernewsletter.example.com>", "Hacker Newsletter #712", "Favourite links this week: a deep dive on SQLite internals, why your terminal is slow, building a tiny language model from scratch, and the story of the Friendly Floppy."),
    E("Ops at Cutroom <ops@cutroom.co>", "Invoice INV-0381 from Cutroom Ltd - due Oct 1", "Hi Alex, please find attached invoice INV-0381 for editing services in September: 4 episodes x $450 = $1,800. Payment due Oct 1 by bank transfer. Thank you!"),
    E("Hiring Team <talent@brightloop.ai>", "Staff Developer Advocate role at Brightloop", "Hi Alex, your videos came up repeatedly when we asked engineers who they learn from. We're hiring a Staff Developer Advocate (remote, $210-250k + equity). Would you be open to a conversation? No pressure if the timing is wrong."),
    E("Vercel <notifications@vercel.com>", "Deployment failed: alexdev-site (main)", "The deployment for alexdev-site on branch main failed. Error: Build exceeded maximum duration. View the build logs for details."),
    E("Ben Carter <ben@thedevtoolspod.example.com>", "Would you come on the podcast?", "Hey Alex - Ben from The Devtools Pod. We'd love to have you on to talk about your editor workflow series. Episodes are 45 minutes, remote, and we have about 30k downloads each. Does any time in the first two weeks of October work?"),
    E("1Password <hello@1password.com>", "Your 1Password Families renewal receipt", "Thanks for renewing. We charged $59.88 to your card ending 4242 for one year of 1Password Families. Your next renewal date is Sep 12, 2027."),
    E("it-helpdesk@cutroom-co.support", "URGENT: mailbox storage full, re-validate now", "Your mailbox has exceeded its quota. Incoming messages are being rejected. Re-validate your account by entering your email password at the link below within 12 hours or your mailbox will be deleted."),
    E("Priya (editor) <priya@cutroom.example.com>", "thumbnail options for 142", "Three thumbnail options attached. I like B best but A has your face bigger. Which do you want? No rush, tomorrow is fine."),
    E("Skillshare Partnerships <partners@skillshare.com>", "Partner with Skillshare this autumn", "Hi there, we're running our autumn creator programme. We pay $10 per free-trial signup from your link, with no fixed fee. Creators in the tech space typically see 100-300 signups per video. Interested in a custom link?"),
    E("HMRC <noreply@tax.service.gov.uk>", "Your Self Assessment payment is due on 31 January", "This is a reminder that your Self Assessment tax return and payment are due by 31 January. You can pay through your Government Gateway account. Do not reply to this email."),
    E("Olu Adeyemi <olu@adeyemi.example.com>", "Collab idea: terminal tools tier list", "Alex! Loved the Raycast video. Want to do a joint 'terminal tools tier list'? I can come to London the week of the 12th or we do it remote. Thinking we each bring ten tools and argue about it. Let me know what you think."),
    E("LinkedIn <messages-noreply@linkedin.com>", "You appeared in 214 searches this week", "See who's looking for people like you. Your profile views are up 18% from last week."),
    E("BetterHelp Partnerships <creators@betterhelp-partners.com>", "Sponsored segment - BetterHelp", "Hi Alex, we'd like to sponsor a 60 second segment on your next three videos at $2,000 each. We provide a script and a unique link. Please confirm your interest and your monthly view counts."),
    E("Dropbox <no-reply@dropbox.com>", "Your Dropbox is almost full", "You've used 96% of your 2 TB. Files may stop syncing when you run out of space. Upgrade your plan or free up space."),
    E("Grace Liu <grace@jetbrains.example.com>", "JetBrains x Alex - Fleet launch integration", "Hi Alex, I lead creator marketing at JetBrains. We're launching a major Fleet update on Oct 20 and would like a dedicated video plus one integration. Budget is $18,000 for the pair. We'd need a draft by Oct 13 - is that timeline workable for you?"),
    E("Accounts <accounts@cutroom.co>", "REMINDER: invoice INV-0352 is 14 days overdue", "Hi Alex, our records show INV-0352 ($1,350) is now 14 days overdue. Could you arrange payment today or let us know if there is a problem? Thanks."),
    E("Patreon <no-reply@patreon.com>", "Your September payout is on its way", "We've sent $2,914.22 to your bank account ending 8831. It should arrive in 2-5 business days. View your earnings report."),
    E("noreply@lottery-intl-claims.org", "CONGRATULATIONS!! You won 850,000 GBP", "Your email address was selected in the International Email Lottery. To claim your prize of 850,000 GBP send your full name, address, bank details and a processing fee of 450 GBP to our claims agent."),
    E("Hannah <hannah.r@example.com>", "your talk last week", "Hi Alex - we met briefly after your meetup talk. I'm a second-year CS student and wanted to ask: would you recommend doing an internship at a big company or a startup first? Totally understand if you're too busy to answer."),
    E("Riverside <team@riverside.fm>", "Your recording is ready", "Your recording \"Devtools Pod guest prep\" has finished processing. Download separate tracks or the combined video from your dashboard."),
    E("Manscaped Influencer Team <influencers@manscaped-collab.com>", "Collab? Free product + $400", "Hey! We'd love to send you The Lawn Mower 5.0 and pay $400 for a 30 second mention. Lots of tech YouTubers work with us. Let us know your address!"),
    E("Tax Refund Service <refunds@hmrc-gov-refund.co>", "You are eligible for a tax refund of 1,284.50 GBP", "After the last annual calculation we determined you are eligible for a refund. Submit the refund form with your card number, expiry and CVV within 3 days: hmrc-gov-refund.co/claim"),
    E("Sam", "flights", "Found flights to Lisbon for the 23rd, 94 quid each if we book tonight. They go up tomorrow. Shall I just book? x"),
    E("Zed Industries <hello@zed.dev>", "Sponsoring your editor series", "Hi Alex, big fans of the editor workflow series. We'd like to sponsor the next four episodes at $5,000 each - no script, just use Zed honestly on camera and say what you think. Happy to jump on a call whenever suits."),
    E("Google Workspace <workspace-noreply@google.com>", "Your invoice is available for alexdev.io", "Your Google Workspace invoice for September is available. Amount: $14.40. It will be charged automatically to your payment method."),
    E("YouTube <no-reply@youtube.com>", "Copyright claim on your video \"My 2026 desk setup\"", "A copyright owner has claimed some material in your video. This is not a copyright strike. The claimed content is a 12 second music clip. The video's monetisation may be affected. You can dispute the claim or trim the segment in YouTube Studio."),
]


def seed(store) -> int:
    pid = store.create_project(
        "Processing email", "A sample inbox. Tick the questions you care about, press Run, then click any result to filter.",
        "emails")
    for q in QUESTIONS:
        store.save_question(pid, q)
    store.add_items(pid, [(title_of(e), e) for e in EMAILS])
    return pid
