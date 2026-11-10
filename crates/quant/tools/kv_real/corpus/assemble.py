#!/usr/bin/env python3
"""Builds the mixed test text for the KV capture: a ChatML conversation with a
system prompt, English prose, German prose, source code and short chat turns.
The prose is written for this test."""
import sys

SYSTEM = """You are Mira, the assistant of a small software team in Freiburg. You answer in the language the user writes in. You keep answers short unless the user asks for detail. When you show code you use fenced code blocks and name the language. When you are not sure about something you say so plainly and you never invent file names, function names or numbers. The team works with Rust, Python and a little C++, and its members write both English and German."""

ENGLISH = """The harbour at Pennywick had been silting up for as long as anyone could remember, and the town had made its peace with that. In spring the fishing boats crept out on the top of the tide, two hours either side of high water, and in autumn they did not go out at all unless the wind came from the east and pushed the sea a little higher up the shingle. Old Harriet Moss, who kept the lamp on the point and had done so for thirty-one years, used to say that the harbour was not dying but only thinking about it, which was what most things in Pennywick did.

She lived in the keeper's cottage with a grey cat called Admiral and a wireless set that received the shipping forecast clearly and everything else badly. Each morning she climbed the one hundred and twelve steps to the lantern room, wiped the glass with a chamois that had belonged to her predecessor, and wrote in the log book the weather, the visibility, and the names of any vessels she had seen. There were rarely more than three entries. The lamp itself had been automatic since 1987, and the authority in the city had twice written to ask why a keeper was still needed. Both times Harriet had answered in a single sentence, that a light with nobody under it is only an expensive lamp, and both times the letter had been filed and forgotten.

In the second week of November a yacht came in on the evening tide, much too late and much too fast. It was a long white boat with a mast like a needle, and it turned in the channel so sharply that the people on the quay thought it must have lost its steering. The man at the wheel was young, perhaps thirty, and he tied up badly and without looking at anyone. When Harriet came down from the point with her torch he was sitting on the cabin roof with a mug in both hands, and he said, before she could ask, that he had sailed from Brest in four days and had not slept more than three hours at a time.

She did not ask him why. She had learned long ago that people who arrive by sea at night usually have a reason, and that the reason comes out by itself if the kettle is hot and the questions are few. She took him up to the cottage, gave him the spare bed and a bowl of fish soup, and in the morning she found him standing at the window with the cat on his shoulder, looking at the empty harbour as if it were something he had been promised and had not expected to receive.

His name was Tobias Lund, and he was an engineer who designed pumps. For six years he had worked on a system that moved water uphill with almost no energy, and in the autumn the company had sold the patent to a firm that meant to bury it. He had taken the boat, which was the one thing he owned outright, and gone to sea to decide what to do. Harriet listened and said nothing for a while. Then she said that the harbour at Pennywick needed to be dredged every second year, that the council could not afford it, and that if he wished to be useful he might look at the sluice gate by the old fish market, which had been stuck since the war."""

ENGLISH_Q = """Could you summarise that story in three sentences, and tell me which detail you think is the most important for the ending?"""

ENGLISH_A = """Here is a short summary. Harriet Moss, the last keeper of a lighthouse above a silting harbour, takes in Tobias Lund, an engineer who sailed across the sea in a hurry after his company sold his best invention to a firm that wants to bury it. She does not press him with questions, and after listening to his story she points him to a practical problem of the town, the stuck sluice gate by the fish market. The story leaves open whether he will stay.

The most important detail for an ending is probably the sentence about the light with nobody under it. It says what the story is about: the value of being present for someone, not the value of the machine. If Tobias fixes the sluice gate, he does the same thing for the harbour that Harriet does for him."""

GERMAN = """Der Schwarzwald im November ist kein Ort für Leute, die schnelle Antworten suchen. Der Nebel liegt oft schon am Vormittag in den Tälern, und wer von Freiburg aus mit der Höllentalbahn nach oben fährt, sieht zuerst nur nasse Tannen, dann einen Streifen Himmel und schließlich, hinter Hinterzarten, eine Hochfläche, auf der das Gras silbrig vom Reif ist. Die Fahrgäste im Zug kennen sich meistens vom Sehen. Die Frau mit dem Korb steigt in Titisee aus, der alte Herr mit dem Hund fährt bis Neustadt, und zwei Schüler, die jeden Morgen dieselbe Bank belegen, diskutieren über eine Mathematikarbeit, die sie für ungerecht halten.

Lena Hartmann saß an diesem Morgen auf dem Platz am Fenster und hatte einen Laptop auf den Knien, den sie nicht öffnete. Sie war Ingenieurin bei einem Unternehmen in Freiburg, das Software für Messgeräte schrieb, und sie hatte zwei Tage Urlaub genommen, um zu entscheiden, ob sie das Angebot ihrer Firma annehmen sollte, ein kleines Team in Hamburg zu leiten. Hamburg war schön, das wusste sie, aber es war auch weit weg von ihrer Schwester, von dem Garten ihrer Eltern und von dem Café an der Dreisam, in dem sie seit zehn Jahren jeden Freitag saß und die gleiche Zeitung las.

In Hinterzarten stieg ein Mann ein, den sie nicht kannte. Er trug einen grünen Lodenmantel und hielt einen Karton im Arm, aus dem ein leises Kratzen kam. Er setzte sich ihr gegenüber, sah sie kurz an und sagte, ohne dass sie gefragt hatte, dass es sich um zwei junge Eulen handle, die er zu einer Pflegestation nach Donaueschingen bringe. Sie seien im Herbst aus dem Nest gefallen, und er habe sie seitdem im Keller aufgezogen, was seine Frau nicht besonders schön finde. Lena musste lachen. Es war das erste Mal seit Tagen, dass sie über etwas anderes als ihre Entscheidung nachdachte.

Sie unterhielten sich bis Donaueschingen. Er erzählte, dass er früher Förster gewesen sei und dass ihn die Wälder gelehrt hätten, wie langsam gute Entscheidungen wachsen. Man könne einen Baum nicht überreden, schneller zu werden, sagte er, aber man könne ihm das Licht lassen. Lena schrieb den Satz später in ihr Notizbuch, ohne genau zu wissen, warum. Als sie am Abend wieder in Freiburg ankam, regnete es, und sie hatte noch immer nicht entschieden, ob sie nach Hamburg gehen würde. Aber sie wusste jetzt, dass es keine Eile gab, und das war, wie sie fand, schon ein ziemlich großer Fortschritt."""

GERMAN_Q = """Kannst du mir bitte in zwei Sätzen sagen, worum es in dem Text geht, und erklären, was der Satz mit dem Licht für Lena bedeutet?"""

GERMAN_A = """Gern. Der Text erzählt von Lena Hartmann, die auf einer Zugfahrt durch den Schwarzwald überlegt, ob sie eine Stelle in Hamburg annehmen soll, und die dabei einen ehemaligen Förster mit zwei jungen Eulen trifft. Das Gespräch nimmt ihr nicht die Entscheidung ab, aber den Druck.

Der Satz, man könne einen Baum nicht überreden, schneller zu werden, aber man könne ihm das Licht lassen, bedeutet für Lena, dass sie sich selbst Zeit geben darf. Sie muss nicht sofort wissen, was richtig ist. Es genügt, wenn sie die Bedingungen schafft, unter denen die Antwort wachsen kann, und das ist am Ende der Fortschritt, den sie am Abend bemerkt."""

CODE_Q = """Danke. Jetzt eine andere Sache: Ich habe diese beiden Funktionen geschrieben und bin nicht sicher, ob die Fensterlogik stimmt. First the Python version, then the Rust one. Does the code do what the comment says, and is there an off-by-one error somewhere?"""

CODE = '''```python
def moving_average(values, window):
    """Mean of every `window` consecutive values; the result has
    len(values) - window + 1 entries."""
    if window <= 0:
        raise ValueError("window must be positive")
    if window > len(values):
        return []
    total = sum(values[:window])
    result = [total / window]
    for i in range(window, len(values)):
        total += values[i] - values[i - window]
        result.append(total / window)
    return result


def split_sessions(events, gap_seconds=1800):
    """Group (timestamp, name) events into sessions: a new session starts
    when two events are more than gap_seconds apart."""
    sessions = []
    current = []
    last = None
    for ts, name in sorted(events):
        if last is not None and ts - last > gap_seconds:
            sessions.append(current)
            current = []
        current.append((ts, name))
        last = ts
    if current:
        sessions.append(current)
    return sessions
```

```rust
/// Mean of every `window` consecutive values.
pub fn moving_average(values: &[f64], window: usize) -> Vec<f64> {
    assert!(window > 0, "window must be positive");
    if window > values.len() {
        return Vec::new();
    }
    let mut total: f64 = values[..window].iter().sum();
    let mut out = Vec::with_capacity(values.len() - window + 1);
    out.push(total / window as f64);
    for i in window..values.len() {
        total += values[i] - values[i - window];
        out.push(total / window as f64);
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub timestamp: u64,
    pub name: String,
}

pub fn split_sessions(mut events: Vec<Event>, gap: u64) -> Vec<Vec<Event>> {
    events.sort_by_key(|e| e.timestamp);
    let mut sessions: Vec<Vec<Event>> = Vec::new();
    let mut last: Option<u64> = None;
    for event in events {
        let new_session = match last {
            Some(t) => event.timestamp - t > gap,
            None => true,
        };
        if new_session {
            sessions.push(Vec::new());
        }
        last = Some(event.timestamp);
        sessions.last_mut().unwrap().push(event);
    }
    sessions
}
```'''

CODE_A = """Both versions do what their comments say, and I do not see an off-by-one error in the window logic. The first window is the first `window` values, the loop starts at index `window`, and each step adds the new value and drops the one that left the window. For a list of n values you get n - window + 1 averages, which matches the comment.

Three smaller points. In the Python version `sorted(events)` sorts by timestamp and then by name, which is fine, but if names can be None the sort will fail. In the Rust version `event.timestamp - t` cannot underflow, because the events are sorted and `t` is the timestamp of the previous event, so the subtraction is never negative. The running total in both versions slowly collects rounding error for very long inputs; if that matters, recompute the sum from scratch every few thousand steps or use a compensated sum.

Ein Hinweis auf Deutsch, weil du die Frage gemischt gestellt hast: Die Sitzungsgrenze ist strikt größer als `gap`, zwei Ereignisse mit genau 1800 Sekunden Abstand gehören also noch zur selben Sitzung. Das ist vermutlich gewollt, sollte aber im Kommentar stehen."""

CHAT = [
    ("user", "Gut zu wissen. Was meinst du, sollen wir die Sitzungsgrenze lieber als Parameter mit Standardwert 30 Minuten lassen oder in der Konfiguration ablegen?"),
    ("assistant", "Ich würde sie als Parameter mit Standardwert lassen. Die Funktion bleibt dann rein und leicht zu testen, und die Konfiguration kann den Wert beim Aufruf einsetzen. Wenn mehrere Stellen dieselbe Grenze brauchen, lohnt sich später eine kleine Struktur, die sie bündelt."),
    ("user", "Okay. And one more thing, completely different: our standup is at nine, but Jonas keeps joining ten minutes late because of the train. Any polite way to bring it up?"),
    ("assistant", "Yes. Do not make it a complaint about the lateness; ask about the constraint. Something like: \"Jonas, the train seems to make nine hard for you. Would a standup at 9:15 work better for the whole team, or should we start with the written updates and let you join for the discussion?\" That gives him a way to answer without having to apologise, and it lets the team decide together."),
    ("user", "Das klingt gut. Schreib mir bitte noch eine kurze Nachricht dafür, auf Deutsch, nicht zu förmlich."),
    ("assistant", "Hallo Jonas, mir ist aufgefallen, dass der Zug um neun für dich ziemlich knapp ist. Wollen wir das Standup auf Viertel nach neun schieben, oder sollen wir mit den schriftlichen Updates anfangen und du kommst für die Diskussion dazu? Sag einfach, was dir besser passt, dann klären wir es im Team. Danke dir!"),
    ("user", "Perfekt, danke. Noch etwas Anderes: Hier ist der Entwurf unseres Berichts zum Ausfall der letzten Woche."),
    ("assistant", "Gern, schick ihn mir. Ich sehe mir Zeitleiste und Ursachen an und sage dir, ob sie zusammenpassen."),
]



POSTMORTEM_Q = """Here is the draft of our post-mortem for last week's outage. Please read it and tell me in a few sentences whether the timeline and the causes are consistent, and what is missing."""

POSTMORTEM = """Summary. On Tuesday the 3rd of November the public API returned errors for 47 minutes, between 14:12 and 14:59 UTC. About 18 percent of all requests failed during that time; the rest were served with a median latency of 2.4 seconds instead of the usual 180 milliseconds. No data was lost. The incident was caused by a change to the connection pool settings of the order service, which was deployed at 13:55 and which reduced the maximum number of database connections per instance from 40 to 4 because of a typing mistake in the configuration file.

Timeline. At 13:55 the new configuration was rolled out to the first of six instances. The canary checks passed, since they only send a few requests per minute. At 14:05 the rollout reached all instances. At 14:12 the first alert fired on the error rate of the gateway. The on-call engineer acknowledged it at 14:15 and first suspected the network, because the errors were timeouts and not rejections. At 14:31 a second engineer noticed that the pool wait time of the order service had grown from 2 milliseconds to more than 900 milliseconds. At 14:44 the old configuration was found in the history and at 14:52 the rollback was started. By 14:59 the error rate was back to normal.

Causes. The direct cause is the wrong value, 4 instead of 40. Contributing causes are that the canary stage does not generate enough load to exhaust a pool, that the configuration is not validated against a minimum, and that the alert for pool wait time existed but was routed to a channel nobody reads at night. The 29 minutes between the first alert and the first look at the pool metrics were the largest part of the outage."""

POSTMORTEM_A = """The timeline and the causes are mostly consistent. The 47 minutes match the times you give, and the rollout order explains why the error rate rose slowly between 13:55 and 14:12. Two things are missing. First, the draft says 18 percent of requests failed, but it does not say which requests, and since only the order service had the small pool I would expect orders to be affected far more than other calls. Second, the section on causes should say what will change: a minimum value check in the configuration loader, a canary stage with a load test, and a routing fix for the pool wait alert. Also check the number of minutes between the first alert and the look at the pool metrics: 14:12 to 14:31 is 19 minutes, not 29."""


def turn(role, text):
    return f"<|im_start|>{role}\n{text}<|im_end|>\n"


def build():
    parts = [turn("system", SYSTEM)]
    parts.append(turn("user", ENGLISH + "\n\n" + ENGLISH_Q))
    parts.append(turn("assistant", ENGLISH_A))
    parts.append(turn("user", GERMAN + "\n\n" + GERMAN_Q))
    parts.append(turn("assistant", GERMAN_A))
    parts.append(turn("user", CODE_Q + "\n\n" + CODE))
    parts.append(turn("assistant", CODE_A))
    for role, text in CHAT:
        parts.append(turn(role, text))
    parts.append(turn("user", POSTMORTEM_Q + "\n\n" + POSTMORTEM))
    parts.append(turn("assistant", POSTMORTEM_A))
    parts.append(turn("user", "Thanks, you are right about the 19 minutes. Now the last question for today, as promised: what is the difference between a mutex and a semaphore, in two sentences?"))
    parts.append("<|im_start|>assistant\n")
    return parts


if __name__ == "__main__":
    out = sys.argv[1]
    parts = build()
    with open(out, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    # character offsets of the parts for the segment table
    off = 0
    for p in parts:
        print(off, len(p), repr(p[:40]))
        off += len(p)
