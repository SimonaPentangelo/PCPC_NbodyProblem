#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SOFTENING 1e-9f
typedef struct
{
  float x, y, z, vx, vy, vz;
} Body;

int getSize(FILE *fp)
{
  int nBodies;
  fscanf(fp, "%d", &nBodies);
  return nBodies;
}

int getBodies(FILE *fp, int nBodies, Body *allBodies)
{
  int i = 0;
  while (!feof(fp) && i < nBodies)
  {
    fscanf(fp, "%f %f %f %f %f %f", &allBodies[i].x, &allBodies[i].y, &allBodies[i].z, &allBodies[i].vx, &allBodies[i].vy, &allBodies[i].vz);
    i++;
  }
  return nBodies;
}

void getSubBodies(Body *subBodies, int numPerProc, Body *allBodies, int displs[], int myrank)
{
  int j = 0;
  for (int i = displs[myrank]; i < displs[myrank] + numPerProc; i++)
  {
    subBodies[j] = allBodies[i];
    j++;
  }
}

void bodyForce(Body *p, float dt, int n, Body *a, int dim)
{
  for (int i = 0; i < n; i++)
  {
    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;

    for (int j = 0; j < dim; j++)
    {
      float dx = a[j].x - p[i].x;
      float dy = a[j].y - p[i].y;
      float dz = a[j].z - p[i].z;
      float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3;
      Fy += dy * invDist3;
      Fz += dz * invDist3;
    }

    p[i].vx += dt * Fx;
    p[i].vy += dt * Fy;
    p[i].vz += dt * Fz;
  }
}

void fillCounts(int counts[], int resto, int nBodies, int ws)
{
  for (int i = 0; i < ws; i++)
  {
    if (i < resto)
    {
      counts[i] = (nBodies / ws) + 1;
    }
    else
    {
      counts[i] = nBodies / ws;
    }
  }
}

void fillDispls(int displs[], int counts[], int ws)
{
  displs[0] = 0;
  for (int i = 1; i < ws; i++)
  {
    displs[i] = displs[i - 1] + counts[i - 1];
  }
}

void printLog(Body *p, int n, char *arr)
{
  FILE *fp;
  fp = fopen(&arr[0], "w");
  fprintf(fp, "Body  x       y        z     ||    xv       yv       zv\n");
  for (int i = 0; i < n; i++)
  {
    fprintf(fp, "%f %f %f %f %f %f\n", p[i].x, p[i].y, p[i].z, p[i].vx, p[i].vy, p[i].vz);
  }
  fclose(fp);
}

void randomizeBodies(Body *data, int n, int seed)
{
  srand(seed);
  for (int i = 0; i < n; i++)
  {
    data[i].x = data[i].x * (rand() / (float)RAND_MAX) - 1.0f;
    data[i].y = data[i].y * (rand() / (float)RAND_MAX) - 1.0f;
    data[i].z = data[i].z * (rand() / (float)RAND_MAX) - 1.0f;
    data[i].vx = data[i].vx * (rand() / (float)RAND_MAX) - 1.0f;
    data[i].vy = data[i].vy * (rand() / (float)RAND_MAX) - 1.0f;
    data[i].vz = data[i].vz * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void updateBodies(Body *data, int size, float dt)
{
  for (int i = 0; i < size; i++)
  { // integrate position
    data[i].x += data[i].vx * dt;
    data[i].y += data[i].vy * dt;
    data[i].z += data[i].vz * dt;
  }
}

int main(int argc, char **argv)
{

  double start, end;

  int myrank;
  MPI_Status status;
  MPI_Request request = MPI_REQUEST_NULL;
  int world_size;

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations
  Body *allBodies;        //tutti i bodies
  Body *subBodies;        //sottogruppo di bodies
  int seed = 3;
  int nBodies = 10;
  int numPerProc; //numero di body per processo
  int resto;      //resto da spargere sui processi
  char nameFile[15];
  char inFile[15];
  char outFile[16];

  MPI_Datatype bodytype, oldtype[1]; //per definire il nuovo tipo
  oldtype[0] = MPI_FLOAT;
  int blockcount[1];
  blockcount[0] = 6;
  MPI_Aint offset[1];
  offset[0] = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  if (argc > 1)
    strcpy(nameFile, argv[1]);
  if (argc > 2)
    seed = atoi(argv[1]);
  allBodies = malloc(nBodies * sizeof(Body));
  // define structured type and commit it
  MPI_Type_create_struct(1, blockcount, offset, oldtype, &bodytype);
  MPI_Type_commit(&bodytype);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int counts[world_size];
  int displs[world_size];

  FILE *fp = fopen(nameFile, "r");
  nBodies = getSize(fp);
  allBodies = malloc(nBodies * sizeof(Body));
  getBodies(fp, nBodies, allBodies);
  fclose(fp);
  numPerProc = nBodies / world_size;
  resto = nBodies % world_size;
  if (myrank < resto)
  {
    numPerProc++;
  }
  fillCounts(counts, resto, nBodies, world_size);
  fillDispls(displs, counts, world_size);
  randomizeBodies(allBodies, nBodies, seed);
  if (myrank == 0)
  {
    sprintf(inFile, "%d", nBodies);
    strcat(inFile, "inFile.txt");
    printLog(allBodies, nBodies, inFile);
  }
  subBodies = malloc(numPerProc * sizeof(Body));
  getSubBodies(subBodies, numPerProc, allBodies, displs, myrank);
  start = MPI_Wtime();
  for (int j = 0; j < nIters; j++)
  {
    bodyForce(subBodies, dt, numPerProc, allBodies, nBodies);
    updateBodies(subBodies, numPerProc, dt);
    MPI_Allgatherv(subBodies, numPerProc, bodytype, allBodies, counts, displs, bodytype, MPI_COMM_WORLD);
  }
  end = MPI_Wtime();
  if (myrank == 0)
  {
    sprintf(outFile, "%d", nBodies);
    strcat(outFile, "outFile.txt");
    printLog(allBodies, nBodies, outFile);
    printf("Ime in ms = %f\n", end - start);
  }
  free(allBodies);
  free(subBodies);
  MPI_Finalize();
  return 0;
}