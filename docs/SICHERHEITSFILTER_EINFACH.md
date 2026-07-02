# Der Sicherheitsfilter, einfach erklärt

*Eine Einführung für Neulinge — ohne Vorwissen über Spieltheorie oder
Suchalgorithmen. Die formale Fassung mit Beweis steht in
[SAFETY.md](SAFETY.md), die Animationen dazu in
[media/showcase/](media/showcase/).*

## Das Problem in einem Satz

Mehrere Roboter planen gleichzeitig und unabhängig voneinander ihre Wege —
wie stellt man sicher, dass sie **niemals** zusammenstoßen, obwohl keiner
weiß, was die anderen gleich tun werden?

## Warum das schwer ist

Jeder Roboter in diesem Projekt plant mit einer Baumsuche (SM-MCTS): Er
simuliert tausende mögliche Zukünfte, inklusive dessen, was die *anderen*
vermutlich tun, und wählt dann seinen besten nächsten Schritt. Das
funktioniert erstaunlich gut — aber es ist eine *Wahrscheinlichkeits*-
Aussage, keine Garantie. Das Paradebeispiel ist der Frontal-Korridor
(Clip `04_head_on_dec_conflict.gif`): Zwei Roboter fahren aufeinander zu.
Beide denken: „Ich weiche aus." Beide weichen zur **selben** Seite aus —
so wie zwei Fußgänger, die auf dem Gehweg dreimal gleichzeitig zur selben
Seite treten. In einem von 60 Testläufen passierte genau das.

Man kann die Suche verbessern, länger rechnen, bessere Belohnungen
definieren — aber man bekommt so nie eine *Garantie*. Denn das Problem
liegt nicht an zu wenig Rechenzeit, sondern daran, dass es **mehrere
gleich gute Lösungen** gibt (ich links / du rechts — oder umgekehrt) und
unabhängige Planer sich nicht absprechen können, welche davon gilt.

## Die Idee des Filters: zwei einfache Regeln

Statt zu hoffen, dass die Suche das Richtige tut, schalten wir einen
Filter *vor* die Suche. Der Filter streicht aus der Aktionsliste jedes
Roboters alle Züge, die gefährlich werden *könnten* — die Suche darf nur
noch aus dem Rest wählen. Der Filter besteht aus zwei Regeln, die jeder
Roboter allein ausrechnen kann (er muss nur sehen, wo die anderen gerade
stehen — keine Funkverbindung, keine Absprache):

**Regel 1 — „Jeder Stehplatz ist reserviert."**
Mein Zug ist nur erlaubt, wenn mein Weg in diesem Schritt einen
Sicherheitsabstand zu den *aktuellen* Positionen aller anderen hält. Ich
darf also nie dorthin fahren, wo gerade jemand steht — selbst wenn ich
glaube, dass er gleich wegfährt.

**Regel 2 — „Bei Unklarheit entscheidet die Nummer."**
Regel 1 allein reicht nicht: Zwei Roboter könnten sich *unterwegs*
kreuzen, ohne je auf dem Startplatz des anderen zu landen. Also prüft
Roboter 2 zusätzlich: Ist mein Zug sicher gegen **jeden** Zug, der für
Roboter 0 und 1 nach deren Filterung noch erlaubt ist? (Roboter 0 prüft
nur Regel 1, Roboter 1 prüft gegen Roboter 0, und so weiter.) Die Nummern
sind öffentlich bekannt — jeder kann die Filter aller anderen exakt
nachrechnen.

## Warum das beweisbar funktioniert

Zwei kurze Argumente — das ist wirklich schon der ganze Beweis:

**„Keiner sitzt jemals fest"** (Lemma): *Stehenbleiben* ist immer erlaubt.
Warum? Regel 1 zwingt alle anderen, meinem Stehplatz fernzubleiben. Wenn
ich also stehen bleibe, kann mir nichts passieren — egal, was die anderen
(Erlaubtes) tun. Der Filter kann eine Aktionsliste also nie ganz
leerräumen. Das ist wichtiger, als es klingt: Ein Filter, der einem
Roboter *alle* Züge verbietet, wäre selbst der Unfall.

**„Es kracht nie"** (Theorem): Nimm zwei beliebige Roboter, der eine hat
die höhere Nummer. Sein Zug wurde per Regel 2 gegen *alle* erlaubten Züge
des anderen geprüft — also auch gegen den, den der andere tatsächlich
wählt. Egal, wie beide sich entscheiden (verschiedene Zufallszahlen,
verschiedene Suchen, keinerlei Kommunikation): Zusammenstoßen ist für
dieses Paar unmöglich. Das gilt für jedes Paar und für jeden Zeitschritt —
und Schritt für Schritt bleibt so der Sicherheitsabstand für immer
erhalten.

## Die Annahmen (was man dafür voraussetzen muss)

1. **Am Anfang ist Abstand.** Die Roboter starten weiter als den
   Kollisionsradius voneinander entfernt. (Wird beim Start geprüft.)
2. **Stehenbleiben geht immer.** Jeder Roboter hat eine „Anhalten"-Aktion.
3. **Alle sehen dasselbe.** Positionen, Blickrichtungen und die
   Nummernreihenfolge sind allen bekannt. Nur so rechnen alle denselben
   Filter aus.
4. **Bewegung ist berechenbar.** Innerhalb eines Zeitschritts bewegen sich
   alle auf geraden Linien — dieselbe Annahme, mit der auch der Simulator
   Kollisionen misst.
5. **Der Boden verschwindet nicht unter den Füßen.** In Karten mit
   beweglichen Wänden (unsere Tor-Szenarien) darf sich eine Wand nicht
   genau dort schließen, wo ein Roboter steht.

Und genauso wichtig, was der Filter **nicht** verspricht: dass jeder
*ankommt*. Er verhindert Unfälle, keine Staus. Dass die Roboter trotzdem
zügig ans Ziel finden, erledigt weiterhin die Baumsuche — sie optimiert
den Fortschritt innerhalb der erlaubten Züge, und ihr eingebauter Zufall
bricht Pattsituationen auf (empirisch nach 1–3 Schritten, siehe
[THEORY.md](THEORY.md)).

## Die Ergebnisse

Gemessen mit unabhängig planenden Robotern (der schwierige Fall), je 20
Zufallsläufe (Tor-Szenario: 8):

| Szenario | ohne Filter | mit Filter |
|---|---|---|
| Frontal-Korridor, 2 Roboter | 3 Kollisionsschritte in 20 Läufen | **0** |
| Engstelle, 2 Roboter | 0 | **0** |
| Wechseltore, 4 Roboter, bewegliche Wand | 0 | **0** |

Drei Beobachtungen, die man Neulingen mitgeben sollte:

1. **Die Null ist keine Statistik.** Sie ist das Theorem. Mehr Läufe
   könnten sie nicht verschlechtern — das ist der ganze Unterschied
   zwischen „hat nie gekracht" und „kann nicht krachen".
2. **Der befürchtete Preis blieb aus.** Wir hatten erwartet, dass der
   Filter die Roboter langsamer macht (er verbietet ja knappes
   Hintereinanderherfahren). Tatsächlich waren die gefilterten Läufe sogar
   etwas *kürzer* (z. B. 9.4 statt 10.1 Schritte) und die Roboter sagten
   die Züge der anderen *besser* voraus (Konsistenz 0.74–0.77 statt
   0.67–0.73). Erklärung: Der Filter streicht genau die mehrdeutigen
   Konflikt-Züge — und nimmt den unabhängigen Planern damit einen Teil der
   Abstimmungsarbeit ab.
3. **Der echte Preis ist Rechenzeit:** etwa Faktor 2 pro Planungsschritt
   (z. B. 2 Roboter dezentral: ~0.2 s → ~0.6 s pro Schritt auf CPU). Die
   Showcase-GIFs sind in Echtzeit gerendert — die Bildwechselrate
   entspricht dem *langsamsten* gemessenen Planungsschritt des jeweiligen
   Laufs, man sieht also ehrlich, wie schnell sich das auf einer CPU
   anfühlt.

## Wo der Filter im Code steckt

Kern ist eine einzige Funktion, `safe_action_mask` in
[`sm_mcts_jax/safety.py`](../sm_mcts_jax/safety.py) (~40 Zeilen Logik):

1. **Wege aufspannen:** Für jeden Roboter und jede seiner ~6 Aktionen wird
   der Weg dieses Schritts als Kette von Stichprobenpunkten berechnet —
   für alle gleichzeitig, als ein einziges Array `[Roboter, Aktion,
   Stützpunkt, xy]`.
2. **Regel 1 vektorisiert:** Abstand jedes Stützpunkts zu jeder fremden
   Position; ein Zug fliegt raus, wenn irgendein Abstand zu klein ist.
3. **Regel 2 als kleine Schleife:** Für Roboter 0, 1, 2, … der Reihe nach:
   streiche jeden Zug, der mit irgendeinem *übrig gebliebenen* Zug eines
   Vorgängers kollidieren würde (paarweise Weg-zu-Weg-Abstände, wieder ein
   Array-Vergleich).

Eingeschaltet wird das mit einem einzigen Flag —
`MCTSParams(safety_filter=True)` — und ersetzt dann überall in der Suche
(Baum, Rollouts, Wurzel) die normale „Wand im Weg?"-Prüfung. Dass der
Beweis stimmt, prüft zusätzlich ein Test maschinell nach: Er probiert in
einer Frontal-Konfiguration **alle** Kombinationen erlaubter Züge aus und
bestätigt, dass der Kollisionsmelder des Simulators nie anschlägt
(`tests/test_safety.py`).

## Die Clips im Showcase (docs/media/showcase/)

| Clip | Was man sieht |
|---|---|
| 01 Kreuzung, zentral | Basisfall: ein Planer für beide, einer wartet an der Kreuzung |
| 02 Frontal, zentral | Beide weichen komplementär aus — vom Koordinator verordnet |
| 03 Frontal, dezentral | Dasselbe Verhalten *entsteht* ohne Koordinator |
| 04 Frontal, dezentral, **Konflikt** | Beide weichen zur selben Seite aus (Kollision!) — der seltene Fehlerfall |
| 05 wie 04, **mit Filter** | Gleicher Zufalls-Seed: Konflikt unmöglich, Lauf sogar optimal |
| 06 Engstelle, zentral | Verhandlung um die einzige Lücke |
| 07 Engstelle, dezentral + Filter | Dasselbe, garantiert kollisionsfrei |
| 08 Vier Roboter, zentral | Positionstausch um ein Hindernis |
| 09 Wechseltore, zentral | Bewegliche Wand: Timing auf die Torfenster |
| 10 Wechseltore, dezentral | Vier unabhängige Planer, bewegliche Wand |
| 11 Wechseltore, dezentral + Filter | Der volle Härtefall mit Garantie (~3 s/Schritt auf CPU) |
