#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SOFTENING 1e-9f

typedef struct
{
  float x, y, z, vx, vy, vz;
} Body;

void randomizeBodies(float *data, int n)
{
  for (int i = 0; i < n; i++)
  {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void saveBodies(Body *p, int n)
{
  FILE *fp;
  char str[80];
  sprintf(str, "%d", n);
  strcat(str, "bodies.txt");
  fp = fopen(str, "w");
  fprintf(fp, "%d\n", n);
  for (int i = 0; i < n; i++)
  {
    fprintf(fp, "%f %f %f %f %f %f\n", p[i].x, p[i].y, p[i].z, p[i].vx, p[i].vy, p[i].vz);
  }
  fclose(fp);
}

int main(const int argc, const char **argv)
{

  int nBodies = 30000;
  if (argc > 1)
    nBodies = atoi(argv[1]);

  int bytes = nBodies * sizeof(Body);
  float *buf = (float *)malloc(bytes);
  Body *p = (Body *)buf;

  randomizeBodies(buf, 6 * nBodies); // Init pos / vel data
  saveBodies(p, nBodies);
  free(buf);
}