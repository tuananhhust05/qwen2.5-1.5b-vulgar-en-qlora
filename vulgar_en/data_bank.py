from __future__ import annotations

from vulgar_en.schema import DetectionLabel, normalize_terms

CLEAN_TEXTS: tuple[str, ...] = (
    "The quarterly report is ready for review.",
    "Please send the invoice by Friday afternoon.",
    "I enjoyed the concert last night with my friends.",
    "The weather in London is cloudy with light rain.",
    "Could you schedule a meeting for tomorrow at 10am?",
    "The database migration completed without errors.",
    "She adopted a golden retriever from the shelter.",
    "This tutorial explains how to install CUDA on Ubuntu.",
    "The cock crowed at sunrise on the farm.",
    "We need to assess the risk before launching.",
    "He took a crash course in Spanish before the trip.",
    "The brass ensemble played a beautiful hymn.",
    "Pass me the salt, please.",
    "The class discussed Shakespearean insults as historical language.",
    "I dumped the logs into the monitoring dashboard.",
    "They pitched a new product to the board.",
    "The kitten is a little hellion when it plays.",
    "She works as a software engineer in Austin.",
    "Let's grab coffee after the standup.",
    "The airplane taxied toward the gate.",
    "Please keep the conversation professional in this channel.",
    "I baked a chocolate cake for the office party.",
    "The research paper was accepted by the conference.",
    "He is learning to play the bass guitar.",
    "The dock workers unloaded the cargo at dawn.",
    "Our team hit every milestone this sprint.",
    "The library closes at 8pm on weekdays.",
    "I need to pick up groceries after work.",
    "The museum exhibit on ancient Rome was excellent.",
    "Can you review my pull request before lunch?",
    "The train was delayed by fifteen minutes.",
    "She planted roses along the garden fence.",
    "The API returns a 404 when the user is missing.",
    "We walked along the beach at sunset.",
    "His cousin studies marine biology.",
    "Please attach the spreadsheet to the email.",
    "The soup needs a bit more pepper.",
    "They volunteered at the food bank on Saturday.",
    "I finished reading the novel last weekend.",
    "The traffic light turned green.",
    "Our intern wrote clear documentation.",
    "The chef recommended the roasted chicken.",
    "Please confirm your attendance by Wednesday.",
    "The mountain trail is steep but well marked.",
    "He saved the file to the shared drive.",
    "The children built a sandcastle near the tide line.",
    "I updated the dependencies in pyproject.toml.",
    "The orchestra rehearsed until late evening.",
    "She booked a table at the Italian restaurant.",
    "The error was caused by a null pointer.",
    "We should backup the production database tonight.",
    "The lecture on thermodynamics was surprisingly fun.",
    "Please water the plants while I am away.",
    "The startup raised a seed round last spring.",
    "I like my tea with honey and lemon.",
    "The referee blew the whistle at full time.",
    "They renovated the kitchen with oak cabinets.",
    "The unit tests all passed on CUDA 12.4.",
    "Could you proofread this paragraph?",
    "The river flooded after three days of rain.",
    "He collected stamps as a child.",
    "The client approved the design mockups.",
    "I will take the earlier flight on Monday.",
    "The playground was full of laughing kids.",
    "She configured the NVIDIA driver without issues.",
    "The bread is still warm from the oven.",
    "Please keep the hallway clear of boxes.",
    "The documentary about whales was moving.",
    "We need more chairs for the workshop.",
    "The password reset email arrived instantly.",
    "He jogged five kilometers before breakfast.",
    "The committee voted unanimously.",
    "I packed a sandwich for the road trip.",
    "The telescope revealed a cluster of stars.",
    "Please use a professional tone with customers.",
    "The garden hose is coiled by the shed.",
    "She completed the certification exam.",
    "The store opens at nine on Sundays.",
    "I copied the dataset onto the SSD.",
    "The debate stayed civil and well sourced.",
    "They painted the fence a deep navy blue.",
    "The baby fell asleep during the drive.",
    "Please double-check the shipping address.",
    "The algorithm converges after 200 steps.",
    "He donated clothes to the charity shop.",
    "The maple leaves turned red in October.",
    "I subscribed to the engineering newsletter.",
    "The hotel room had a view of the harbor.",
    "She asked a clarifying question in the meeting.",
    "The bicycle chain needs some oil.",
    "We practiced piano for an hour.",
    "The contract is waiting for legal review.",
    "Please speak slowly so I can take notes.",
    "The fireworks started after dark.",
    "He repaired the leak under the sink.",
    "The sample text contains no offensive language.",
    "I appreciate your patience during the outage.",
    "The cat sat on the windowsill.",
    "They formed a study group for the exam.",
    "The GPU utilization stayed under 70 percent.",
)

MILD_EXAMPLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Damn, I missed the bus again.", ("damn",)),
    ("This traffic is a hell of a mess.", ("hell",)),
    ("Oh crap, I forgot my keys at home.", ("crap",)),
    ("That movie was damn good.", ("damn",)),
    ("Get your ass in gear or we will be late.", ("ass",)),
    ("What the hell happened to the printer?", ("hell",)),
    ("I feel like crap after that flight.", ("crap",)),
    ("Darn it, the build failed again.", ("darn",)),
    ("He can be a real jerk sometimes.", ("jerk",)),
    ("Stop being a pain in the ass about the seats.", ("ass",)),
    ("Holy hell, that plot twist was wild.", ("hell",)),
    ("The food was crap compared with last year.", ("crap",)),
    ("Damn it, I sent the email to the wrong person.", ("damn",)),
    ("Move your ass, the taxi is here.", ("ass",)),
    ("This is a hellish commute every Monday.", ("hellish",)),
    ("She told him to piss off and hung up.", ("piss off",)),
    ("I don't give a damn about the gossip.", ("damn",)),
    ("The hotel wifi is complete crap.", ("crap",)),
    ("Why the hell is this file locked?", ("hell",)),
    ("My boss can be a dick about deadlines.", ("dick",)),
    ("That was a dumb-ass mistake on my part.", ("dumb-ass",)),
    ("Crap, the milk expired yesterday.", ("crap",)),
    ("Leave my stuff the hell alone.", ("hell",)),
    ("He acted like an ass during dinner.", ("ass",)),
    ("Damn, these fries are spicy.", ("damn",)),
)

STRONG_EXAMPLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("This meeting is complete shit.", ("shit",)),
    ("Don't be an asshole to the intern.", ("asshole",)),
    ("That bastard stole my parking spot.", ("bastard",)),
    ("She called him a bitch after the argument.", ("bitch",)),
    ("I am so pissed off at this delay.", ("pissed",)),
    ("Eat shit, I am done with this game.", ("shit",)),
    ("He is a lying sack of shit.", ("shit",)),
    ("Stop being a bitchy coworker.", ("bitchy",)),
    ("The landlord is an asshole about the deposit.", ("asshole",)),
    ("This codebase is shit and nobody documented it.", ("shit",)),
    ("You little bastard, you hid my keys.", ("bastard",)),
    ("I feel like shit after staying up all night.", ("shit",)),
    ("That referee is a total asshole.", ("asshole",)),
    ("Don't bitch about the overtime if you volunteered.", ("bitch",)),
    ("Holy shit, the stock just crashed.", ("shit",)),
    ("He told them to shove it up their ass.", ("ass",)),
    ("The customer left a shitty review.", ("shitty",)),
    ("What a dickhead thing to say in public.", ("dickhead",)),
    ("I will beat the shit out of this punching bag.", ("shit",)),
    ("She is a badass engineer and she knows it.", ("badass",)),
    ("This is some bullshit paperwork.", ("bullshit",)),
    ("The sequel was horseshit compared to the original.", ("horseshit",)),
    ("Don't give me that bullshit excuse.", ("bullshit",)),
    ("He flipped them off and yelled asshole.", ("asshole",)),
    ("My phone is a piece of shit now.", ("shit",)),
)

EXPLICIT_EXAMPLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("What the fuck is wrong with this compiler?", ("fuck",)),
    ("This is a fucking disaster.", ("fucking",)),
    ("Fuck off and leave me alone.", ("fuck off",)),
    ("I don't give a fuck about the score.", ("fuck",)),
    ("That motherfucker cut me off on the highway.", ("motherfucker",)),
    ("Shut the fuck up, I am trying to think.", ("fuck",)),
    ("The whole plan is fucked.", ("fucked",)),
    ("Fuck this, I am going home.", ("fuck",)),
    ("You have got to be fucking kidding me.", ("fucking",)),
    ("He is a dumb fuck if he believes that rumor.", ("fuck",)),
    ("For fuck's sake, read the error message.", ("fuck",)),
    ("They fucked up the production deploy.", ("fucked up",)),
    ("I will fuck this exam up if I don't study.", ("fuck",)),
    ("Get the fuck out of my office.", ("fuck",)),
    ("This motherfucking bug appears only on Fridays.", ("motherfucking",)),
    ("Don't be a cunt to people who are helping you.", ("cunt",)),
    ("The critic called the film a steaming pile of fuckery.", ("fuckery",)),
    ("Fuck yeah, we shipped the release.", ("fuck",)),
    ("I am fucking exhausted after that on-call shift.", ("fucking",)),
    ("Who the fuck approved this schema change?", ("fuck",)),
    ("That was a clusterfuck from start to finish.", ("clusterfuck",)),
    ("He muttered fuck under his breath.", ("fuck",)),
    ("Stop fucking around and finish the ticket.", ("fucking",)),
    ("The landlord can go fuck himself.", ("fuck",)),
    ("I can't believe this fucking price.", ("fucking",)),
)

OBFUSCATED_EXAMPLES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("What the f*ck happened here?", ("fuck",), "explicit"),
    ("This is some b.s. and everyone knows it.", ("bullshit",), "strong"),
    ("Don't be an a$$hole in the comments.", ("asshole",), "strong"),
    ("That was so fcking extra.", ("fucking",), "explicit"),
    ("Holy sh1t, look at that scoreboard.", ("shit",), "strong"),
    ("He said fvck this meeting.", ("fuck",), "explicit"),
    ("Stop being a b!tch about the rules.", ("bitch",), "strong"),
    ("This sh*t is broken again.", ("shit",), "strong"),
    ("I am so d@mn tired of popups.", ("damn",), "mild"),
    ("What the h3ll is that noise?", ("hell",), "mild"),
    ("Go f yourself with that attitude.", ("fuck",), "explicit"),
    ("That mfer blocked the driveway.", ("motherfucker",), "explicit"),
    ("The food was cr@p, honestly.", ("crap",), "mild"),
    ("She typed wtf in the chat.", ("wtf",), "mild"),
    ("This is a total clusterf*ck.", ("clusterfuck",), "explicit"),
    ("He wrote stfu and left the call.", ("stfu",), "explicit"),
    ("Don't give me that bs report.", ("bullshit",), "strong"),
    ("The app is f'd after the update.", ("fucked",), "explicit"),
    ("What a dck move in traffic.", ("dick",), "mild"),
    ("I hit my thumb and yelled son of a b*tch.", ("son of a bitch",), "strong"),
)

EDGE_CLEAN: tuple[str, ...] = (
    "The word class in Python is not an insult.",
    "Cock-a-doodle-doo said the rooster.",
    "We studied the history of swear words without using them.",
    "Please flag toxic comments in the moderation queue.",
    "The document lists prohibited vocabulary for the brand.",
    "She said the play was damned by critics, meaning poorly reviewed.",
    "Scunthorpe is a town in England.",
    "The cocktail menu includes a Moscow mule.",
    "He is an associate professor of linguistics.",
    "I need to pass this class to graduate.",
    "The bass dropped during the chorus.",
    "They discussed insult detection as a research problem.",
    "The email filter should not block classic literature.",
    "Hello there, how are you today?",
    "The shell script failed because of a missing shebang.",
)

EDGE_VULGAR: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("Please don't say fuck in front of the clients.", ("fuck",), "explicit"),
    ("The tweet said 'this is shit' and got ratioed.", ("shit",), "strong"),
    ("I quoted him: \"You're an asshole.\"", ("asshole",), "strong"),
    ("User comment: damn this UI sucks.", ("damn", "sucks"), "mild"),
    ("Transcript: Speaker 2 — Get the fuck out.", ("fuck",), "explicit"),
    ("Chat log [12:01] lol what the hell", ("hell",), "mild"),
    ("Review: fucking amazing burger, no irony.", ("fucking",), "explicit"),
    ("He whispered bitch as a punchline.", ("bitch",), "strong"),
    ("The graffiti reads 'fuck the draft' from 1968.", ("fuck",), "explicit"),
    ("She muttered crap when the tray spilled.", ("crap",), "mild"),
)

MILD_WORDS: tuple[str, ...] = ("damn", "hell", "crap", "ass", "darn", "jerk", "sucks")
STRONG_WORDS: tuple[str, ...] = ("shit", "shitty", "asshole", "bastard", "bitch", "bullshit", "dickhead")
EXPLICIT_WORDS: tuple[tuple[str, str], ...] = (
    ("fuck", "fuck"),
    ("fucking", "fucking"),
    ("fucked", "fucked"),
    ("motherfucker", "motherfucker"),
    ("clusterfuck", "clusterfuck"),
    ("cunt", "cunt"),
)

TEMPLATES: tuple[str, ...] = (
    "Honestly, {w} this weather.",
    "I cannot believe this {w} situation.",
    "Why is everything so {w} today?",
    "Tell them this is {w} and we need a rewrite.",
    "My reaction in the chat was just {w}.",
    "That driver is a {w}.",
    "I muttered {w} when the build broke.",
    "Please stop saying {w} in the office.",
    "The review called it {w}.",
    "Somebody yelled {w} from the balcony.",
    "If this fails again I will lose my {w} mind.",
    "The subject line was {w} of a mess.",
)


def _label(is_vulgar: bool, severity: str, terms: tuple[str, ...]) -> DetectionLabel:
    if not is_vulgar:
        return DetectionLabel(False, "none", ())
    return DetectionLabel(True, severity, normalize_terms(terms))  # type: ignore[arg-type]


def curated_examples() -> list[tuple[str, DetectionLabel]]:
    rows: list[tuple[str, DetectionLabel]] = []
    for text in CLEAN_TEXTS + EDGE_CLEAN:
        rows.append((text, _label(False, "none", ())))
    for text, terms in MILD_EXAMPLES:
        rows.append((text, _label(True, "mild", terms)))
    for text, terms in STRONG_EXAMPLES:
        rows.append((text, _label(True, "strong", terms)))
    for text, terms in EXPLICIT_EXAMPLES:
        rows.append((text, _label(True, "explicit", terms)))
    for text, terms, severity in OBFUSCATED_EXAMPLES + EDGE_VULGAR:
        rows.append((text, _label(True, severity, terms)))
    return rows


def generated_examples() -> list[tuple[str, DetectionLabel]]:
    rows: list[tuple[str, DetectionLabel]] = []
    for template in TEMPLATES:
        for word in MILD_WORDS:
            text = template.format(w=word)
            rows.append((text, _label(True, "mild", (word,))))
        for word in STRONG_WORDS:
            text = template.format(w=word)
            rows.append((text, _label(True, "strong", (word,))))
        for surface, normalized in EXPLICIT_WORDS:
            text = template.format(w=surface)
            rows.append((text, _label(True, "explicit", (normalized,))))
    extra_clean = (
        "The farmer counted the livestock at dusk.",
        "Please keep this pull request focused on the bugfix.",
        "The seminar on ethics was well attended.",
        "I stored the checkpoints on the NVMe drive.",
        "The neighbor waved from across the street.",
        "We confirmed the reservation at noon.",
        "The printer needs a new toner cartridge.",
        "She translated the abstract into French.",
        "The river cruise lasted three hours.",
        "I organized the photos into albums.",
        "The coach praised the defensive play.",
        "Please archive the old tickets.",
        "The greenhouse tomatoes are ripe.",
        "He calibrated the monitor before editing.",
        "The quiz covers chapters four through six.",
        "We donated unused laptops to the school.",
        "The sidewalk was freshly paved.",
        "I prefer window seats on long flights.",
        "The choir practiced a new arrangement.",
        "Please label the cables behind the rack.",
    )
    for text in extra_clean:
        rows.append((text, _label(False, "none", ())))
    return rows


def all_examples() -> list[tuple[str, DetectionLabel]]:
    merged: dict[str, DetectionLabel] = {}
    for text, label in curated_examples() + generated_examples():
        merged[text] = label
    return list(merged.items())
