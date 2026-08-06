# Motivazione scientifica dell’equivarianza alle traslazioni

## Obiettivo

La Dice loss misura la sovrapposizione tra predizione e ground truth, ma non
garantisce che il modello sia stabile rispetto a piccole variazioni del crop o
dell’allineamento. L’equivarianza introduce un segnale complementare: se
l’immagine viene traslata di due voxel, anche la segmentazione dovrebbe
traslare nello stesso modo.

\[
f(Tx) \approx T f(x),
\]

dove `f` è SwinUNETR e `T` è una traslazione esatta sulla griglia 3-D.

## Perché può essere utile

La posizione dell’ippocampo nel crop può cambiare leggermente senza che cambi
la sua identità anatomica. Una forte variazione della segmentazione dopo uno
spostamento minimo indica dipendenze posizionali o instabilità interne che la
sola supervisione voxel per voxel non penalizza esplicitamente.

Il vincolo è interessante perché:

- non impone una forma anatomica media uguale per tutti i soggetti;
- non richiede atlanti o registrazioni aggiuntive;
- produce un segnale differenziabile e denso;
- può migliorare la robustezza a crop, padding e piccoli disallineamenti;
- nelle analisi iniziali la violazione era associata a errori di segmentazione
  maggiori.

## Formulazione differenziabile

Si confrontano le probabilità `p` ottenute dall’immagine originale con le
probabilità `q` ottenute dall’immagine traslata e poi riportate nelle coordinate
originali. La regione di bordo persa dalla traslazione viene esclusa.

Per ciascuna delle due classi dell’ippocampo:

\[
S_c =
\frac{2\sum p_cq_c+\epsilon}
{\sum p_c^2+\sum q_c^2+\epsilon}.
\]

La soddisfazione è la media dei due valori e la loss è:

\[
L_{eq}=1-S_{eq}.
\]

Il denominatore quadratico rende la metrica riflessiva: due mappe soft
identiche hanno soddisfazione uno. Questo valore, chiamato `truth` nel codice,
deve essere la misura principale dell’equivarianza.

## Limite della metrica storica di adherence

Per compatibilità con l’analisi iniziale, il codice calcola anche una Dice con
denominatore lineare. Se le due mappe sono identiche, questa seconda metrica
può comunque essere inferiore a uno quando le probabilità non sono nette. Di
conseguenza misura contemporaneamente consistenza e confidenza.

La percentuale sopra la soglia `0.90` è utile solo come indicatore secondario e
non deve essere descritta come prova autonoma di equivarianza. Nell’evaluation
finale essa rappresenta la percentuale di coppie caso-traslazione aderenti, non
la percentuale di pazienti che soddisfano tutte le sei direzioni.

## Piano sperimentale

Il confronto minimo deve usare la stessa pipeline:

1. SwinUNETR con sola Dice (`constraint-set none`);
2. SwinUNETR con Dice più equivarianza (`constraint-set equivariance`).

Per entrambi gli esperimenti bisogna valutare tutte le sei traslazioni e
riportare:

- Dice hard finale e migliore Dice di validazione;
- soddisfazione quadratica media;
- risultati separati per classe e direzione;
- numero di casi migliorati e peggiorati;
- ripetizioni su più seed o fold.

Il miglioramento della sola adherence non è sufficiente. Il vincolo è utile se
aumenta la consistenza senza ridurre la qualità della segmentazione e se il
guadagno di Dice si ripete in confronti controllati.

## Ipotesi di ricerca

L’ipotesi è che la regolarizzazione di equivarianza riduca la sensibilità di
SwinUNETR alla posizione assoluta nel crop e migliori la segmentazione rispetto
alla sola Dice loss, fornendo allo stesso tempo una misura interpretabile della
stabilità del modello.
