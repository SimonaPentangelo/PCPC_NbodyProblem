# **N-body Problem**
Progetto di corso per l'esame di **Programmazione Concorrente e Parallela su Cloud 2020/21**.

- Studente: **Simona Pentangelo**
- Matricola: **0522501017**
- Data di consegna: **01/06/2021**  
___
## Sommario
- [**N-body Problem**](#n-body-problem)
  - [Sommario](#sommario)
  - [Introduzione](#introduzione)
  - [Soluzione proposta](#soluzione-proposta)
  - [Dettagli implementativi](#dettagli-implementativi)
    - [Definizione del tipo body](#definizione-del-tipo-body)
    - [Fasi preliminari](#fasi-preliminari)
    - [Update dei body](#update-dei-body)
    - [Fase di comunicazione](#fase-di-comunicazione)
  - [Istruzioni per l'esecuzione](#istruzioni-per-lesecuzione)
  - [Correttezza](#correttezza)
    - [Risultati](#risultati)
  - [Benchmarks](#benchmarks)
    - [Strong Scaling con 20000 body](#strong-scaling-con-20000-body)
    - [Strong Scaling con 30000 body](#strong-scaling-con-30000-body)
    - [Weak Scaling](#weak-scaling)
___
## Introduzione
Il problema degli N-body consiste nel conoscere la posizione e la velocità di un insieme di corpi, i quali si influenzano tra loro, in un intervallo di tempo. Questo genere di simulazioni è molto utilizzato in astrofisica per predirre il movimento di un gruppo di corpi celesti, in base alle influenze gravitazionali.

Per una simulazione sono necessari un insieme di corpi, per cui vengono fornite le coordinate all'interno dello spazio, la massa e le informazioni relative alla velocità di movimento sui tre assi. L'output consiste nell'insieme di particelle di input con posizioni e velocità aggiornate, in base alla durata della simulazione e all'influenza subita dagli altri corpi.
___
## Soluzione proposta
Per risolvere questo problema, è stata proposta la seguente soluzione:
  + Un programma in C per creare il file di partenza con il numero desiderato di body.
  + Un programma in C con le funzioni per:
    - Leggere e memorizzare i valori presenti nel file di partenza e randomizzarli tramite un seme;
    - Dividere i body in base al numero di processi;
    - Calcolare le informazioni necessarie per la comunicazione;
    - Calcolare e aggiornare le informazioni dei body;
    - Scrivere in un file i valori dei body.


Di seguito, sono state riportate le funzioni di maggiore importanza.
___
## Dettagli implementativi  

### Definizione del tipo body  

Per gestire agilmente l'insieme dei body all'interno del programma, è stata creata una struttura, così da poter allocare lo spazio necessario e scorrere facilmente gli elementi in fase di update.  

```c
typedef struct { float x, y, z, vx, vy, vz; } Body;
```  
Per poter inviare tali elementi usando **MPI**, è stato necessario creare il tipo `bodytype`, specificando i tipi corrispondenti dei sei elementi contenuti nella struttura (ovvero `MPI_FLOAT`).
```c
MPI_Datatype bodytype, oldtype[1]; //per definire il nuovo tipo
oldtype[0] = MPI_FLOAT;
int blockcount[1];
blockcount[0] = 6;
MPI_Aint offset[1];
offset[0] = 0;

// define structured type and commit it
MPI_Type_create_struct(1, blockcount, offset, oldtype, &bodytype);
MPI_Type_commit(&bodytype);
```
### Fasi preliminari  

Per avere un insieme di body di partenza, è stato utilizzato il programma **body_creation.c**, il quale genera un file contenente la quantità di body creata nella prima riga e per ogni riga, sei differenti valori (che serviranno a generare i valori dei sei campi del tipo `Body`).

```c
int getSize(FILE* fp) {
  int nBodies;
  fscanf(fp, "%d", &nBodies);
  return nBodies;
}

int getBodies(FILE* fp, int nBodies, Body *allBodies) {
  int i = 0;
    while(!feof(fp) && i < nBodies)
  {
        fscanf(fp, "%f %f %f %f %f %f", &allBodies[i].x, &allBodies[i].y, &allBodies[i].z, &allBodies[i].vx, &allBodies[i].vy, &allBodies[i].vz);
        i++;
  }
  return nBodies;
}
```  

Una volta ottenuti tutti i body, viene stabilita la quantità di body che dovrà essere gestita da un singolo core, basandosi sul numero di processori, sul numero di body e sull'eventuale resto.

```c
numPerProc = nBodies / world_size;
resto = nBodies % world_size;
if(myrank < resto) {
    numPerProc++;
}
```  

Per utilizzare permettere la comunicazione (ripresa più avanti), è stato necessario creare gli array `counts` e `displs` per rendere noto alla root quanti body vengono inviati da ogni processore ed il displacement necessario per combinare i dati in `allBodies`.

```c
void fillCounts(int counts[], int resto, int nBodies, int ws) {
    for(int i = 0; i < ws; i++) {
        if(i < resto) {
            counts[i] = (nBodies/ws) + 1;
        } else {
            counts[i] = nBodies/ws;
        }
    }
}

void fillDispls(int displs[], int counts[], int ws) {
    displs[0] = 0;
    for(int i = 1; i < ws; i++) {
        displs[i] = displs[i-1] + counts[i-1];
    }
}
```  
L'array `displs`, oltre che per la comunicazione, è stato anche utilizzato per assicurarsi che il puntatore `subBodies` facesse riferimento alla parte di array interessata dal processore (identificato tramite `myrank`).

```c
void getSubBodies(Body *subBodies, int numPerProc, Body *allBodies, int displs[], int myrank) {
  int j = 0;
    for(int i = displs[myrank]; i < displs[myrank] + numPerProc; i++) {
      subBodies[j] = allBodies[i];
      j++;
    }
}
```

### Update dei body

`bodyForce` è la funzione per calcolare e aggiornare i valori della velocità relativamente ai tre assi dei body.  
Poichè la funzione viene utilizzata in base al sottoinsieme di body assegnati ad un determinato processore, il puntatore `p` fa riferimento al sottinsieme, mentre `a` fa riferimento alla totalià dei body, i quali vengono aggiornati per ogni processre dopo ogni iterazione. Il valore `n` corrisponde al numero di body presenti in `p`, mentre `dim` è il numero totale di body.  
Questa funzione, per ogni nodo del sottinsieme, calcola i nuovi valori della velocità sui tre assi, considerando le posizioni dei corpi.
```c
void bodyForce(Body *p, float dt, int n, Body* a, int dim) {
    for (int i = 0; i < n; i++) { 
            float Fx = 0.0f; 
            float Fy = 0.0f; 
            float Fz = 0.0f;

        for (int j = 0; j < dim; j++) {
            float dx = a[j].x - p[i].x;
            float dy = a[j].y - p[i].y;
            float dz = a[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
    }
}
```
Una volta aggiornati i valori delle velocità, il sottinsieme di body, rappresentato dal puntatore `data`, utilizza le velocità aggiornate per calcolare le nuove posizioni.
```c
void updateBodies(Body *data, int size, float dt) {
    for (int i = 0 ; i < size; i++) { // integrate position
        data[i].x += data[i].vx*dt;
        data[i].y += data[i].vy*dt;
        data[i].z += data[i].vz*dt;
    }
}
```

### Fase di comunicazione

Ogni processore quindi, calcola le velocià e le posizioni dei propri body. Per poter proseguire alla prossima iterazione è necessario aggiornare `allBodies`, per fare ciò, viene utilizzata la funzione `MPI_Allgatherv`, la quale fa una gather dei `subBodies` e combina i dati in `allBodies`, rendendo le modifiche disponibili a tutti.
```c
for(int j = 0; j < nIters; j++) {
        bodyForce(subBodies, dt, numPerProc, allBodies, nBodies);
        updateBodies(subBodies, numPerProc, dt);
        MPI_Allgatherv(subBodies, numPerProc, bodytype, allBodies, counts, displs, bodytype, MPI_COMM_WORLD);
    } 
```
___
## Istruzioni per l'esecuzione  
Per eseguire correttamente la simulazione, è necessario compilare ed eseguire il programma **body_creation.c**, specificando la quantità di body che si intende utilizzare (se omesso, il valore di default è 30000). 
```bash
gcc body_creation.c -o body_creation.out  

./body_creation.c [numeroBodies]
```
Verrà generato un file il cui nome è nel formato **[numeroBodies]bodies.txt**, che sarà necessario per eseguire il programma **nbody.c**. Può essere anche inserito il valore del seme per la fase di randomizzazione dei bodies (se omesso, il valore di default è 3).
```bash
mpicc nbody.c -o nbody.out -lm

mpirun --allow-run-as-root --mca btl_vader_single_copy_mechanism none -np [numeroProcessi] nbody.out [nomeFile] [seed]
```
Durante l'esecuzione, verrà generato un file nel formato **[numeroBodies]inFile.txt** per visualizzare i body generati, mentre al termine verrà stampato il tempo impegato per le iterazioni sui body e verrà generato un file  **[numeroBodies]outFile.txt** per visualizzare i valori aggiornati dei body.
___
## Correttezza
Per dimostrare la correttezza dell'algoritmo, è stato utilizzato un file di partenza di soli dieci body, così da poter visualizzare facilmente i risultati ottenuti con uno, due o più processori.  
Le immagini sottostanti ripostano il file di partenza ottenuto in seguito alla randomizzazione (con seme di default) e file di output dopo le dieci iterazioni.  
Nonostante la variazione del numero di processi, avendo lo stesso file di input generato da **body_creation.c**, vengono prodotti gli stessi risultati.  

### Risultati

*File di input - np = 1*             |  *File di output - np = 1*
:-------------------------:|:-------------------------:
![inFile](risultati/10infile1proc.png)  |  ![outFile](risultati/10outfile1proc.png)

*File di input - np = 2*             |  *File di output - np = 2*
:-------------------------:|:-------------------------:
![inFile](risultati/10infile2proc.png)  |  ![outFile](risultati/10outfile2proc.png)

*File di input - np = 5*             |  *File di output - np = 5*
:-------------------------:|:-------------------------:
![inFile](risultati/10infile5proc.png)  |  ![outFile](risultati/10outfile5proc.png)


___
## Benchmarks  

Per osservare strong e weak scaling dell'algoritmo parallelo, è stato utilizzato un cluster di t2.xlarge.  
Per valutare lo strong scaling, sono state utilizzate fino a quattro istanze (utilizzando quindi da uno a sedici core) e sono stati fatti due test utilizzando un numero di body differenti (ventimila e trentamila).  
Per valutare il weak scaling, sono state usate quattro istanze, facendo in modo che ogni core dovesse lavorare su duemila body (quindi partendo da un core con duemila body fino a sedici core con trentaduemila body).

### Strong Scaling con 20000 body  

| vCPUs | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Tempo | 78,53 | 39,28 | 19,65 | 14,66 | 11,19 | 8,93 | 7,51 | 6,42 | 5,67 |
| Efficienza | 100,00% | 99,92% | 99,84% |89,04% | 89,30% | 88,47% | 89,04% | 85,79% | 87,00% |  

*Tempo medio di esecuzione (in secondi)*           |  *Efficienza (in percentuale)*
:-------------------------:|:-------------------------:
![StrongScaling](grafici/strongScaling20000.png)  |  ![StrongScaling](grafici/speedup20000.png)  

  
 ### Strong Scaling con 30000 body  

| vCPUs | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Tempo | 176,70 | 88,42 | 44,24 | 33,08 | 24,73 | 19,97 | 16,76 | 14,71 | 12,69 |
| Efficienza | 100,00% | 99,92% | 99,84% |89,04% | 89,30% | 88,47% | 89,04% | 85,79% | 87,00% |  

*Tempo medio di esecuzione (in secondi)*           |  *Efficienza (in percentuale)*
:-------------------------:|:-------------------------:
![StrongScaling](grafici/strongScaling30000.png)  |  ![StrongScaling](grafici/speedup30000.png) 
  

### Weak Scaling  

| vCPUs | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Tempo | 0,79 | 1,57 | 3,15 | 4,75 | 6,34 | 7,94 | 9,53 | 11,10 | 12,70 |
| Efficienza | 100,00% | 50,03% | 24,97% | 16,58% | 16,58% | 9,91% | 8,26% | 7,09% | 6,19% |
  

*Tempo medio di esecuzione (in secondi)*           |  *Efficienza (in percentuale)*
:-------------------------:|:-------------------------:
![WeakScaling](grafici/weakscaling1.png)  |  ![WeakScaling](grafici/weakscaling2.png) 
___