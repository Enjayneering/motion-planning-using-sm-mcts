# Forschungsvision: Emergente sichere Koordination durch interaktive Suche

*Das Rahmendokument der Forschung. Die technischen Einzelergebnisse stehen
in SAFETY.md, THEORY.md und ASYNC.md; dieses Dokument ordnet sie.*

## Die Forschungsfrage

> Können wir Roboterflotten **schnell, sicher und intelligent dezentral**
> in die Welt lassen, indem wir einen interaktiven Suchalgorithmus
> (SM-MCTS) mit **bewusst vereinfachten Simulationen** kombinieren —
> diskrete Aktionen, ausreichend feine Grids und Zeitschritte — und diese
> Kombination über **strategische (lange) und taktische (kurze)
> Zeithorizonte** schichten, sodass gemeinsam super-intelligentes und
> super-sicheres Verhalten entsteht?
>
> Endziel: Ein einzelner Agent, der diesen Ansatz nutzt, navigiert sicher
> durch eine Welt, in der sich auch Menschen und andere Roboter (mit oder
> ohne denselben Ansatz) bewegen.

## Die Kernwette: Vereinfachung ist der Mechanismus, nicht der Kompromiss

Die Wette dieser Forschung: Wenn man die Welt *grob genug* diskretisiert,
wird tiefe strategische Interaktionssuche in Echtzeit bezahlbar — und was
an Präzision verloren geht, holt eine *feinere taktische Ebene desselben
Prinzips* zurück. **Ein interaktives Suchprinzip, teleskopiert über
Auflösungen.**

Das grenzt sich vom Stand der Technik ab: Heutige Stacks schichten zwar
auch (Behavior / Trajektorie / Regler), aber Interaktivität lebt höchstens
in einer Schicht — darunter werden andere Agenten als vorhergesagte
Hindernisse behandelt („predict-then-plan"). Genau daraus entsteht das
Frozen-Robot-Problem (Trautman & Krause): Wer den anderen als statische
Vorhersage sieht, findet in dichter Umgebung keinen zulässigen Pfad mehr.
SM-MCTS ist deshalb hier keine Implementierungsentscheidung, sondern die
Abgrenzung: Der andere wird auf jeder Ebene als *antwortfähig* modelliert.
Gegenüber kontinuierlichen Spieltheorie-Planern (iLQGames, ALGAMES), die
in ein lokales Equilibrium konvergieren, repräsentiert der Suchbaum
**multimodale** Entscheidungen (links vorbei / rechts vorbei) nativ — und
unsere Experimente zeigen, dass genau diese Multimodalität der zentrale
Fehlermodus dezentraler Koordination ist (messbar als Prediction
Consistency).

## Sicherheit als Dial, nicht als Schalter

Die Arbeit behandelt Sicherheit nicht als Ja/Nein-Architektur, sondern als
kontinuierlichen Regler zwischen zwei implementierten Endpunkten:

- **Harte Hülle** (SAFETY.md): der bewiesene Maximin-Filter — Kollision
  per Konstruktion unmöglich, unabhängig von Suchqualität.
- **Emergente Sicherheit**: Vorhersagegüte + hohe Replanning-Frequenz +
  feine Zeitauflösung tragen die Sicherheitsmasse; keine harte Garantie,
  dafür keinerlei Konservatismus.

Empirischer Zwischenstand, der die Dial-Sicht stützt: Der Filter machte in
unseren Benchmarks *nicht* passiv (Episoden wurden kürzer, Konsistenz
stieg), und hohe Replanning-Frequenz allein erwies sich als der stärkste
einzelne Sicherheitsfaktor (ASYNC.md). Die offene, quantifizierbare
Kernfrage: **Wie viel harte Hülle kann man abbauen, wenn die Intelligenz
wächst (Simulationsbudget, Replan-Rate, Auflösung) — bei konstantem
nachgewiesenem Risiko?** Nachweis kleiner Restrisiken (10^-6) erfordert
Rare-Event-Methoden (Importance Sampling auf Konfliktsituationen); die
JAX-Infrastruktur ist dafür ausgelegt. Als Zielarchitektur dient das
Luftfahrtprinzip: intelligente Koordination oben, eine dünne, selten
bindende zertifizierte Hülle unten (least-restrictive shielding).

## Die offene Welt: Absichten und Payoffs schätzen

Der Übergang von „Common Knowledge der Payoffs" (Annahme A3) zur offenen
Welt ist der entscheidende Schritt Richtung Realität: Wo will der andere
hin (Ziel), was treibt ihn (Payoff-Funktion), wie groß ist er (Radius)?
Der natürliche Fit in dieses Framework: ein **Posterior über (Ziel,
Payoff-Parameter, Ausdehnung)** je beobachtetem Agenten (Bayesian inverse
planning; für Menschen mit Boltzmann-/noisy-rational-Modellen), aus dem
die Suchwurzel sampelt — jede Simulation spielt gegen eine andere
Hypothese. „*Sicher* schätzen" heißt dabei: nicht Punktschätzung, sondern
risikosensitive Aggregation über Hypothesen (CVaR statt Mittelwert).
Filter-These und Lern-These treffen sich hier: *Der Filter macht
Fehlschätzungen ungefährlich; das Lernen macht ihn selten nötig.*

## Das Programm als Fragenhierarchie

| RQ | Frage | Status |
|---|---|---|
| RQ1 | Läuft interaktive Suche auf vereinfachten Welten in Echtzeit? | ✅ beantwortet: JAX/XLA, ms-Bereich, N Agenten, dynamische Umgebungen |
| RQ2 | Unter welchen Informationsbedingungen koordinieren sich unabhängige Planer (Timing, Commitments, Konventionen)? | 🔬 angebrochen: Beobachtbarkeit schlägt Timing (ASYNC.md); Messgröße Prediction Consistency |
| RQ3 | Wie verläuft die Trade-Kurve harte Hülle ↔ Effizienz bei wachsender Intelligenz? | 🧩 Endpunkte implementiert, Kurve offen |
| RQ4 | Schlägt strategisch-grob + taktisch-fein (dasselbe interaktive Prinzip) Einzelauflösung und predict-then-plan? | ⏳ die Kernwette, ungebaut |
| RQ5 | Ziel-/Payoff-/Radius-Inferenz und risikosensitive Planung gegenüber Nicht-Konformen (Menschen)? | ⏳ offen; Sim-Umgebung liefert das Testfeld |

## Neuheit und Sinnhaftigkeit

Die Bausteine existieren einzeln (hierarchisches MCTS, Shielding,
Intent-Inferenz, dezentrale Planer, asynchrones Planen). Neu ist die
**Vereinheitlichung**: ein interaktives Suchprinzip über alle Horizonte,
Sicherheit als kontinuierlicher Dial statt Architekturentscheidung, und
Koordinationsfähigkeit als gemessene Größe mit identifizierten
Minimalbedingungen (Commitment-Beobachtbarkeit). Sinnhaftigkeit:
Interaktionsbewusstsein ist der anerkannte Flaschenhals des Feldes, das
Frozen-Robot-Problem ist ungelöst, und Zertifizierung verlangt genau die
messbaren Sicherheitsaussagen, auf die dieses Programm zuläuft.

Realistischer Einsatzpfad: **geschlossene Areale** (Minen, Häfen,
Werksgelände — dort gelten die Annahmen) → **Drohnen/U-space**
(Intent-Broadcast wird ohnehin standardisiert) → **gemischte Räume mit
Menschen** (Inferenz + Dial).

## Rolle der interaktiven Simulationsumgebung

Die Web-Simulationsumgebung (`sm_mcts_web/`, docs/SIM_INTERFACE.md) macht
das Programm erlebbar und ist zugleich Forschungsinstrument: kontinuierliche
Welt mit Ackermann-Fahrzeugen, darunter der diskrete Echtzeit-Planer als
strategische Schicht und ein Regler als taktische Schicht — die erste
gebaute Instanz der RQ4-Architektur. Der Mensch fährt selbst mit; der
Planer behandelt ihn als Agenten mit geschätztem Ziel — die erste, bewusst
primitive Instanz von RQ5. Die Planner-Schnittstelle ist versioniert und
algorithmus-agnostisch, damit der Forschungsgegenstand (der Planer)
austauschbar bleibt, ohne die Umgebung anzufassen.
